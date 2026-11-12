"""Data generator: schools first, then students; per-cell linear SCM in topological order."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .spec import ObservationScale
from .world import CompiledWorld


def seed_sequence(master: int, *key: int) -> np.random.SeedSequence:
    """Addressable stream: (master, condition, rep) -> independent SeedSequence (design §6.2)."""
    return np.random.SeedSequence(entropy=master, spawn_key=tuple(int(k) for k in key))


def _draw_categorical(rng: np.random.Generator, attr, assigns: dict[str, np.ndarray], n: int) -> np.ndarray:
    values = list(attr.values)
    if attr.depends_on and attr.table:
        parent = assigns[attr.depends_on]
        out = np.empty(n, dtype=object)
        u = rng.random(n)
        for pv, row in attr.table.items():
            m = parent == pv
            cum = np.cumsum([row[v] for v in values])
            idx = np.searchsorted(cum, u[m] * cum[-1], side="right")
            out[m] = np.array(values, dtype=object)[np.minimum(idx, len(values) - 1)]
        return out
    probs = np.array([attr.values[v] for v in values])
    idx = rng.choice(len(values), size=n, p=probs / probs.sum())
    return np.array(values, dtype=object)[idx]


def skewed(z: np.ndarray, skew: float) -> np.ndarray:
    """Standard normal draws mapped to a standardized gamma with this skewness (mean 0, SD 1), by quantile.

    Same uniform, same rank: a student's skewed noise sits where their normal noise would have been.
    """
    if skew == 0:
        return z
    k = 4.0 / skew**2  # gamma shape: skewness 2 / sqrt(k)
    u = np.clip(stats.norm.cdf(np.abs(skew) / skew * z), 1e-15, 1 - 1e-15)
    return np.sign(skew) * (stats.gamma.ppf(u, k) - k) / np.sqrt(k)


def observe(x: np.ndarray, scale: ObservationScale, mean: float, sd: float, rng: np.random.Generator) -> np.ndarray:
    """What the researcher records for latent values ``x`` (pooled latent ``mean`` and ``sd``)."""
    z = (x - mean) / sd
    if scale.type == "censored":
        lo = -np.inf if scale.low is None else scale.low
        hi = np.inf if scale.high is None else scale.high
        return mean + sd * np.clip(z, lo, hi)
    if scale.type == "count":
        rate = scale.mean * np.exp(scale.spread * z - scale.spread**2 / 2)
        return rng.poisson(rate).astype(float)
    return 1.0 + np.searchsorted(np.asarray(scale.cuts, float), z, side="right").astype(float)


def allocate(p: np.ndarray, n: int) -> np.ndarray:
    """Largest-remainder allocation of n units to strata with proportions p."""
    raw = np.asarray(p, float) / np.sum(p) * n
    base = np.floor(raw).astype(int)
    rest = n - base.sum()
    base[np.argsort(-(raw - base), kind="stable")[:rest]] += 1
    return base


def students_per_school(cw: CompiledWorld, rng: np.random.Generator, n_schools: int) -> np.ndarray:
    sps = cw.world.schools.students_per_school
    if sps.dist == "fixed":
        return np.full(n_schools, sps.n, dtype=int)
    if sps.dist == "uniform":
        return rng.integers(sps.min, sps.max + 1, size=n_schools)
    return np.clip(rng.poisson(sps.n, size=n_schools), sps.min, sps.max)


def generate(cw: CompiledWorld, seed: int | np.random.SeedSequence, *, n_schools: int | None = None,
             students: int | None = None) -> pd.DataFrame:
    """Draw one Dataset. Same (World, seed) gives an identical DataFrame.

    ``students`` overrides the per-school count (fixed), used by large oracle simulations.
    """
    rng = np.random.default_rng(seed)
    world = cw.world
    S = n_schools or world.schools.n_schools
    n_j = np.full(S, students, dtype=int) if students else students_per_school(cw, rng, S)
    N = int(n_j.sum())
    school_of = np.repeat(np.arange(S), n_j)

    # categorical attributes: school-level per school, student-level per student
    school_assign: dict[str, np.ndarray] = {}
    school_attrs = [a for a in cw.attr_order if world.attributes[a].level == "school"]
    if school_attrs and world.schools.allocation == "stratified":
        parts = {}
        for g, sk in enumerate(cw.school_key):
            parts.setdefault(int(sk), {a: cw.assigns[g][a] for a in school_attrs})
        from .world import school_factors
        counts = allocate(cw.pi_school * school_factors(cw), S)
        order = rng.permutation(np.repeat(np.arange(len(counts)), counts))
        for a in school_attrs:
            school_assign[a] = np.array([parts[int(k)][a] for k in order], dtype=object)
    else:
        for a in school_attrs:
            school_assign[a] = _draw_categorical(rng, world.attributes[a], school_assign, S)
    student_assign: dict[str, np.ndarray] = {a: v[school_of] for a, v in school_assign.items()}
    for a in cw.attr_order:
        attr = world.attributes[a]
        if attr.level == "student":
            student_assign[a] = _draw_categorical(rng, attr, student_assign, N)

    # cell index per student
    key_to_cell = {tuple(asg[a] for a in cw.attr_order): g for g, asg in enumerate(cw.assigns)}
    if cw.attr_order:
        keys = list(zip(*[student_assign[a] for a in cw.attr_order], strict=True))
        cell = np.fromiter((key_to_cell[k] for k in keys), dtype=int, count=N)
    else:
        cell = np.zeros(N, dtype=int)

    # continuous variables in topological order (obs:/pv: variables come last, so earlier draws and
    # common random numbers are unaffected by adding a measurement layer)
    p = cw.p
    v = np.zeros((N, p))
    sd_cell = np.sqrt(cw.psi_g)
    det: dict[int, np.ndarray] = {}
    for j in range(p):
        if cw.school_level[j]:
            eps = rng.normal(0.0, 1.0, size=S)[school_of] * np.sqrt(cw.psi[j])
        else:
            z = rng.normal(0.0, 1.0, size=N)
            node = world.nodes.get(cw.vars[j])
            if node is not None and node.noise is not None:
                z = skewed(z, node.noise.skew)
            eps = z * sd_cell[cell, j]
        parents = np.nonzero(np.any(cw.B[:, j, :] != 0, axis=0))[0]
        val = cw.alpha[cell, j].copy()
        for k in parents:
            val = val + cw.B[cell, j, k] * v[:, k]
        det[j] = val
        v[:, j] = val + eps

    data: dict[str, object] = {"school_id": school_of}
    for a in cw.attr_order:
        attr = world.attributes[a]
        data[a] = pd.Categorical(student_assign[a], categories=list(attr.values), ordered=attr.ordinal)
    for name in cw.visible:
        node = world.nodes[name]
        if not node.observed:
            continue
        if node.plausible_values is not None:
            j = cw.index[f"pv:{name}"]
            data[f"{name}_pv1"] = v[:, j]
            for m in range(2, node.plausible_values.n + 1):
                data[f"{name}_pv{m}"] = det[j] + rng.normal(0.0, 1.0, size=N) * sd_cell[cell, j]
        else:
            data[name] = v[:, cw.index[cw.data_var(name)]]
    sd = world.schools.sampling
    if sd is not None:
        data.update(design_weights(cw, school_of, school_assign, S, rng))
    scaled = [n for n in cw.visible if world.nodes[n].scale is not None and world.nodes[n].observed]
    if scaled:
        # last, from a child stream: every other draw (and so the latent data) matches the unscaled World
        from .truth import pooled_moments
        m, cov = pooled_moments(cw)
        seq = rng.bit_generator.seed_seq
        assert isinstance(seq, np.random.SeedSequence)
        srng = np.random.default_rng(seq.spawn(1)[0])
        for name in scaled:
            j = cw.index[name]
            scale = world.nodes[name].scale
            assert scale is not None
            data[name] = observe(v[:, j], scale, float(m[j]), float(np.sqrt(cov[j, j])), srng)
    return pd.DataFrame(data)


def design_weights(cw: CompiledWorld, school_of: np.ndarray, school_assign: dict[str, np.ndarray], S: int,
                   rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Final student weights (inverse of the oversampling) and Fay-BRR replicate weights.

    Schools are paired into variance zones within explicit strata (an odd school joins the last pair
    and keeps factor 1); replicate r multiplies one school of each zone by (2 - k) and the other by k,
    following a Hadamard pattern (PISA: k = 0.5, so factors 1.5 / 0.5, 80 replicates).
    """
    from scipy.linalg import hadamard

    from .world import school_factors
    sd = cw.world.schools.sampling
    assert sd is not None
    school_attrs = [a for a in cw.attr_order if cw.world.attributes[a].level == "school"]
    keys = [tuple(school_assign[a][s] for a in school_attrs) for s in range(S)]
    part_of = {}
    for g, sk in enumerate(cw.school_key):
        part_of.setdefault(tuple(cw.assigns[g][a] for a in school_attrs), int(sk))
    f = school_factors(cw)
    fs = np.array([f[part_of[k]] if k in part_of else 1.0 for k in keys])
    w_school = 1.0 / fs
    n_s = np.bincount(school_of, minlength=S).astype(float)
    total = float((w_school * n_s).sum())
    w_school = w_school * sd.population_students / total
    out: dict[str, np.ndarray] = {"W_FSTUWT": w_school[school_of]}
    if not sd.replicate_weights:
        return out
    R, k = sd.n_replicates, sd.fay_k
    order = rng.permutation(S)
    zones: list[list[int]] = []
    for key in sorted(set(keys), key=str):
        members = [s for s in order if keys[s] == key]
        pairs = [members[i:i + 2] for i in range(0, len(members) - 1, 2)]
        if len(members) % 2 == 1:
            if pairs:
                pairs[-1].append(members[-1])
            else:
                pairs.append([members[-1]])
        zones.extend(pairs)
    size = 1
    while size < max(R, len(zones)):
        size *= 2
    H = hadamard(size)
    rep = np.ones((S, R))
    for z, members in enumerate(zones):
        for r in range(R):
            h = H[z % size, r]
            if len(members) >= 2:
                rep[members[0], r] = (2 - k) if h > 0 else k
                rep[members[1], r] = k if h > 0 else (2 - k)
    for r in range(R):
        out[f"W_FSTURWT{r + 1}"] = (w_school * rep[:, r])[school_of]
    return out
