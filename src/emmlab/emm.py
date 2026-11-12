"""Exceptional Model Mining on the Researcher's Model via pysubgroup (research note R1).

Quality functions compare a subgroup's path coefficients with its complement (or the whole
population). ``current`` mirrors the researcher's pipeline (``SEMQFEntropyCIComplement``: natural-log
entropy of the group size times the number of paths whose intervals do not overlap, intervals from
student-level standard errors), ``ent_z`` is the recommended default, ``wald_z`` a test statistic,
``cook`` Cook's distance.
"""

from __future__ import annotations

import math
import warnings
from collections import namedtuple
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import pysubgroup as ps

from scipy import stats

from .spec import EMMConfig, SearchSpace
from .stages import AnyPrepared

# ---------------------------------------------------------------------------------------------
# Descriptions
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Condition:
    attr: str
    op: str  # "==", "<", ">=", "in[)"
    value: Any
    upper: Any = None
    quantile: float | None = None  # for numeric cuts: the quantile the cut came from
    upper_quantile: float | None = None

    def mask(self, df: pd.DataFrame) -> np.ndarray:
        col = df[self.attr]
        if self.op == "==":
            return (col.astype(str) == str(self.value)).to_numpy()
        x = col.to_numpy(dtype=float)
        if self.op == "<":
            return x < self.value
        if self.op == ">=":
            return x >= self.value
        return (x >= self.value) & (x < self.upper)

    def is_categorical(self) -> bool:
        return self.op == "=="

    def __str__(self) -> str:
        if self.op == "==":
            return f"{self.attr}={self.value}"
        if self.op == "in[)":
            return f"{self.attr}∈[{self.value:.3g},{self.upper:.3g})"
        return f"{self.attr}{self.op}{self.value:.3g}"

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass(frozen=True)
class Description:
    conditions: tuple[Condition, ...]

    def mask(self, df: pd.DataFrame) -> np.ndarray:
        m = np.ones(len(df), dtype=bool)
        for c in self.conditions:
            m &= c.mask(df)
        return m

    def categorical(self) -> dict[str, str] | None:
        if not all(c.is_categorical() for c in self.conditions):
            return None
        return {c.attr: str(c.value) for c in self.conditions}

    def attrs(self) -> set[str]:
        return {c.attr for c in self.conditions}

    def without(self, i: int) -> Description:
        return Description(tuple(c for j, c in enumerate(self.conditions) if j != i))

    def __str__(self) -> str:
        return " AND ".join(str(c) for c in self.conditions) or "∅"

    def key(self) -> tuple:
        return tuple(sorted((c.attr, c.op, str(c.value), str(c.upper)) for c in self.conditions))


def _selector_to_condition(sel, quantiles: dict[tuple[str, float], float]) -> Condition:
    if isinstance(sel, ps.EqualitySelector):
        return Condition(sel.attribute_name, "==", sel.attribute_value)
    lb, ub = sel.lower_bound, sel.upper_bound
    a = sel.attribute_name
    if np.isinf(lb):
        return Condition(a, "<", float(ub), quantile=quantiles.get((a, float(ub))))
    if np.isinf(ub):
        return Condition(a, ">=", float(lb), quantile=quantiles.get((a, float(lb))))
    return Condition(a, "in[)", float(lb), float(ub), quantiles.get((a, float(lb))), quantiles.get((a, float(ub))))


def subgroup_to_description(sg, quantiles) -> Description:
    sels = sg.selectors if hasattr(sg, "selectors") else []
    return Description(tuple(_selector_to_condition(s, quantiles) for s in sels))


# ---------------------------------------------------------------------------------------------
# Selectors and binning (D12)
# ---------------------------------------------------------------------------------------------


def numeric_cuts(values: np.ndarray, school: np.ndarray, n_bins: int, weighting: str) -> list[tuple[float, float]]:
    """Equal-frequency cut points (quantile, value), across students or one vote per school."""
    s = pd.Series(values).groupby(school).mean().to_numpy() if weighting == "schools" else values
    qs = [i / n_bins for i in range(1, n_bins)]
    cuts = []
    for q in qs:
        v = float(np.quantile(s, q))
        if not cuts or v > cuts[-1][1]:
            cuts.append((q, v))
    return cuts


def build_selectors(df: pd.DataFrame, search: SearchSpace, descriptors: list[str],
                    categorical: set[str]) -> tuple[list, dict[tuple[str, float], float]]:
    sels: list = []
    quantiles: dict[tuple[str, float], float] = {}
    school = df["school_id"].to_numpy()
    for a in descriptors:
        if a in categorical:
            for v in df[a].astype(str).unique():
                sels.append(ps.EqualitySelector(a, v))
            continue
        b = search.binning.get(a, search.default_binning)
        cuts = numeric_cuts(df[a].to_numpy(float), school, b.n_bins, b.weighting)
        for q, v in cuts:
            quantiles[(a, v)] = q
        if b.one_sided:
            for _, v in cuts:
                sels.append(ps.IntervalSelector(a, -np.inf, v))
                sels.append(ps.IntervalSelector(a, v, np.inf))
        else:
            edges = [-np.inf] + [v for _, v in cuts] + [np.inf]
            for lo, hi in zip(edges[:-1], edges[1:], strict=True):
                sels.append(ps.IntervalSelector(a, lo, hi))
    return sels, quantiles


# ---------------------------------------------------------------------------------------------
# Path statistics and quality functions
# ---------------------------------------------------------------------------------------------


@dataclass
class PathStat:
    child: str
    reg: str
    b_g: float
    se_g: float
    b_b: float  # baseline (complement or population)
    se_b: float
    lo_g: float = float("nan")
    hi_g: float = float("nan")
    lo_b: float = float("nan")
    hi_b: float = float("nan")

    @property
    def path(self) -> str:
        return f"{self.reg} -> {self.child}"

    @property
    def diff(self) -> float:
        return self.b_g - self.b_b

    @property
    def z(self) -> float:
        d = math.sqrt(self.se_g**2 + self.se_b**2)
        return self.diff / d if d > 0 else 0.0

    @property
    def disjoint(self) -> bool:
        return bool(self.hi_g < self.lo_b or self.hi_b < self.lo_g)

    def deviates(self, quality: str, z_crit: float = 1.959963984540054) -> bool:
        """Does the two sides' difference count as a deviation under this quality function?

        ``current`` / ``current_exact``: the intervals do not overlap (what the quality function counts).
        The others score on |z| or |Δb| with no threshold of their own; a path deviates when |z| >= z_crit.
        """
        qf = QUALITY_FUNCTIONS.get(quality)
        if qf is not None and qf.uses_intervals:
            return self.disjoint
        return bool(abs(self.z) >= z_crit)

    def to_dict(self) -> dict:
        return {"path": self.path, "b_g": self.b_g, "se_g": self.se_g, "b_b": self.b_b, "se_b": self.se_b,
                "diff": self.diff, "z": self.z, "ci_g": [self.lo_g, self.hi_g], "ci_b": [self.lo_b, self.hi_b],
                "disjoint": self.disjoint}


def entropy_weight(n: int, N: int, *, nats: bool = False) -> float:
    """Binary entropy of the group's share of the sample: bits by default, nats for the replica of `current`."""
    if n <= 0 or n >= N:
        return 0.0
    p = n / N
    nat = -p * math.log(p) - (1 - p) * math.log(1 - p)
    return float(nat if nats else nat / math.log(2))


@dataclass
class Evaluation:
    quality: float
    n: int
    n_c: int
    n_schools: int
    g_eff: int
    g_eff_c: int
    paths: list[PathStat] = field(default_factory=list)
    valid: bool = True

    @property
    def best(self) -> PathStat | None:
        if not self.paths:
            return None
        return max(self.paths, key=lambda p: abs(p.diff))


# ---------------------------------------------------------------------------------------------
# Quality functions: the one place to add or change one. Each entry is picked up by the spec
# validation (EMMConfig.quality), the Search & method dropdown (when in_menu) and the Results tab.


@dataclass(frozen=True)
class QualityFunction:
    """How a subgroup is scored, and when one of its paths counts as deviating.

    ``score(pq, ev)`` gets the PathQuality (``pq.N`` is the sample size) and the subgroup's Evaluation
    (``ev.n`` its size, ``ev.paths`` one PathStat per model path: ``diff``, ``z``, ``disjoint``, ...) and
    returns a number; higher is more exceptional, 0 is "nothing".
    """

    score: Callable[[PathQuality, Evaluation], float]
    description: str  # one line, shown in the app
    uses_intervals: bool = False  # True: a path deviates when its intervals do not overlap; False: when |z| >= 1.96
    in_menu: bool = True  # offered in the Search & method dropdown (the others work from YAML)
    bootstrap_intervals: bool = False  # replace the Wald intervals by a school bootstrap before scoring


def _n_disjoint(ev: Evaluation) -> float:
    return float(sum(p.disjoint for p in ev.paths))


QUALITY_FUNCTIONS: dict[str, QualityFunction] = {
    "current": QualityFunction(
        lambda pq, ev: entropy_weight(ev.n, pq.N, nats=True) * _n_disjoint(ev),
        "natural-log entropy of the group size × the number of paths whose subgroup and complement "
        "intervals do not overlap",
        uses_intervals=True),
    "current_exact": QualityFunction(
        lambda pq, ev: entropy_weight(ev.n, pq.N, nats=True) * _n_disjoint(ev),
        "like current, but with bootstrap intervals: entropy × the number of paths whose intervals do not overlap",
        uses_intervals=True, in_menu=False, bootstrap_intervals=True),
    "ent_z": QualityFunction(
        lambda pq, ev: entropy_weight(ev.n, pq.N) * max(abs(p.z) for p in ev.paths),
        "group-size entropy × the largest |z| of the path differences"),
    "cook": QualityFunction(
        lambda pq, ev: pq._cook(ev),
        "Cook's distance of the path coefficients (largest change of one whole equation)"),
    "ent_dbeta": QualityFunction(
        lambda pq, ev: entropy_weight(ev.n, pq.N) * max(abs(p.diff) for p in ev.paths),
        "group-size entropy × the largest |Δb| over the paths", in_menu=False),
    "wald_z": QualityFunction(
        lambda pq, ev: max(abs(p.z) for p in ev.paths),
        "the largest |z| of the path differences", in_menu=False),
}


def quality_catalog() -> list[dict[str, Any]]:
    """The quality functions as the app shows them (GET /api/qualities)."""
    return [{"name": k, "description": q.description, "uses_intervals": q.uses_intervals, "in_menu": q.in_menu}
            for k, q in QUALITY_FUNCTIONS.items()]


class PathQuality:
    """Evaluates any row mask on a Prepared dataset under EMM settings (cached by mask)."""

    def __init__(self, prep: AnyPrepared, emm: EMMConfig, search: SearchSpace, *, per_group: bool | None = None,
                 rng_seed: int = 0):
        self.prep, self.emm, self.search = prep, emm, search
        self.per_group = (prep.analysis.standardize == "per_group") if per_group is None else per_group
        self.N = prep.n
        self.paths = self._paths()
        self.z = stats.norm.ppf(0.5 + emm.interval_level / 2)
        self._cache: dict[bytes, Evaluation] = {}
        self.full = prep.fit(per_group=self.per_group)
        self.rng_seed = rng_seed
        self.evaluations = 0

    def _paths(self) -> list[tuple[str, int, str]]:
        out = []
        for c, regs in self.prep.equations.items():
            for i, r in enumerate(regs):
                if self.emm.paths == "all" or f"{r} -> {c}" in self.emm.paths:
                    out.append((c, i, r))
        return out

    def g_eff(self, mask: np.ndarray, min_members: int = 3) -> int:
        cnt = np.bincount(self.prep.codes[mask], minlength=self.prep.n_groups)
        return int((cnt >= min_members).sum())

    def evaluate(self, mask: np.ndarray) -> Evaluation:
        key = np.packbits(mask).tobytes()
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        self.evaluations += 1
        ev = self._evaluate(mask)
        self._cache[key] = ev
        return ev

    def _fits(self, mask: np.ndarray, complement: bool = False) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        # inside the search BRR is replaced by (weighted) cluster-robust SEs: 80 replicate fits per
        # candidate would be too slow; reported coefficients still get BRR (design note)
        cl = self.prep.analysis.se in ("cluster", "brr") and self.emm.interval_se == "analysis"
        return {c: self.prep.fit_eq(c, mask, self.per_group, cl, complement) for c in {p[0] for p in self.paths}}

    def path_stats(self, g: dict[str, tuple[np.ndarray, np.ndarray]],
                   b: dict[str, tuple[np.ndarray, np.ndarray]]) -> list[PathStat]:
        """Per-path statistics of side ``g`` against side ``b`` from their (coefficients, SEs) per equation."""
        z = float(self.z)
        out = []
        for c, i, r in self.paths:
            ps_ = PathStat(c, r, float(g[c][0][i]), float(g[c][1][i]), float(b[c][0][i]), float(b[c][1][i]))
            ps_.lo_g, ps_.hi_g = ps_.b_g - z * ps_.se_g, ps_.b_g + z * ps_.se_g
            ps_.lo_b, ps_.hi_b = ps_.b_b - z * ps_.se_b, ps_.b_b + z * ps_.se_b
            out.append(ps_)
        return out

    def fit_slice(self, mask: np.ndarray) -> dict[str, tuple[float, float]]:
        """Coefficient and standard error of every model path fitted on one row set (NaN when it cannot be fitted)."""
        g = self._fits(mask)
        return {f"{r} -> {c}": (float(g[c][0][i]), float(g[c][1][i])) for c, i, r in self.paths}

    def compare(self, mask_a: np.ndarray, mask_b: np.ndarray) -> list[PathStat]:
        """Fit every equation on two arbitrary row sets and compare them path by path (a = "g", b = "baseline")."""
        return self.path_stats(self._fits(mask_a), self._fits(mask_b))

    def _evaluate(self, mask: np.ndarray) -> Evaluation:
        n = int(mask.sum())
        n_c = self.N - n
        ge, gec = self.g_eff(mask), self.g_eff(~mask)
        ev = Evaluation(0.0, n, n_c, int(np.count_nonzero(np.bincount(self.prep.codes[mask],
                                                                       minlength=self.prep.n_groups))), ge, gec)
        s = self.search
        if n < s.min_n or n_c < s.min_n_complement or ge < s.min_schools or n_c == 0:
            ev.valid = False
            return ev
        g = self._fits(mask)
        if self.emm.baseline == "complement":
            b = self._fits(mask, complement=True)
        else:
            b = {c: (f.coef, f.se) for c, f in self.full.items()}
        ev.paths = self.path_stats(g, b)
        if any(not np.isfinite(p.diff) for p in ev.paths):
            ev.valid = False
            return ev
        if QUALITY_FUNCTIONS[self.emm.quality].bootstrap_intervals:
            self._bootstrap_intervals(mask, ev)
        ev.quality = self.quality(ev)
        return ev

    def quality(self, ev: Evaluation) -> float:
        qf = QUALITY_FUNCTIONS.get(self.emm.quality)
        if qf is None:
            raise ValueError(self.emm.quality)
        return float(qf.score(self, ev))

    def _cook(self, ev: Evaluation) -> float:
        """Cook's distance of each equation's path coefficients (subgroup vs whole data), max over eqs."""
        best = 0.0
        by_eq: dict[str, list[PathStat]] = {}
        for p in ev.paths:
            by_eq.setdefault(p.child, []).append(p)
        for c, ps_ in by_eq.items():
            full = self.full[c]
            regs = full.regs
            idx = [regs.index(p.reg) for p in ps_]
            X = self.prep.X[c]
            X1 = np.column_stack([np.ones(len(X)), X])
            if self.per_group:
                sx = X.std(0)
                X1 = np.column_stack([np.ones(len(X)), (X - X.mean(0)) / np.where(sx > 0, sx, 1)])
            inv = np.linalg.inv(X1.T @ X1)
            resid_var = (1 - full.r2) * (1.0 if self.per_group else float(self.prep.y[c].var()))
            cov = inv[1:, 1:][np.ix_(idx, idx)] * resid_var
            g = np.array([p.b_g for p in ps_])
            f = full.coef[idx]
            d = g - f
            best = max(best, float(d @ np.linalg.solve(cov, d)) / len(idx))
        return best

    def _bootstrap_intervals(self, mask: np.ndarray, ev: Evaluation) -> None:
        """current_exact: percentile intervals from a school bootstrap of subgroup and baseline."""
        R = self.emm.exact_bootstrap_reps
        rng = np.random.default_rng(self.rng_seed)
        G = self.prep.n_groups
        codes = self.prep.codes
        draws = rng.integers(0, G, size=(R, G))
        lo_a, hi_a = (1 - self.emm.interval_level) / 2, 1 - (1 - self.emm.interval_level) / 2
        sides = {"g": mask, "b": ~mask if self.emm.baseline == "complement" else np.ones_like(mask)}
        res: dict[str, dict[str, np.ndarray]] = {k: {} for k in sides}
        for side, m in sides.items():
            for c in {p[0] for p in self.paths}:
                X, y = self.prep.X[c][m], self.prep.y[c][m]
                cc = codes[m]
                bs = []
                for r in range(R):
                    mult = np.bincount(draws[r], minlength=G)[cc]
                    b = _wls(X, y, mult.astype(float), self.per_group)
                    bs.append(b)
                res[side][c] = np.array(bs)
        for p in ev.paths:
            i = self.prep.equations[p.child].index(p.reg)
            bg, bb = res["g"][p.child][:, i], res["b"][p.child][:, i]
            p.lo_g, p.hi_g = np.nanquantile(bg, lo_a), np.nanquantile(bg, hi_a)
            p.lo_b, p.hi_b = np.nanquantile(bb, lo_a), np.nanquantile(bb, hi_a)


def _wls(X: np.ndarray, y: np.ndarray, w: np.ndarray, standardize: bool) -> np.ndarray:
    sw = w.sum()
    if sw <= X.shape[1] + 1:
        return np.full(X.shape[1], np.nan)
    if standardize:
        mx = (w @ X) / sw
        sx = np.sqrt((w @ (X - mx) ** 2) / sw)
        my = (w @ y) / sw
        sy = math.sqrt((w @ (y - my) ** 2) / sw)
        X = (X - mx) / np.where(sx > 0, sx, 1)
        y = (y - my) / (sy if sy > 0 else 1)
    X1 = np.column_stack([np.ones(len(X)), X])
    xtx = (X1 * w[:, None]).T @ X1
    try:
        return np.linalg.solve(xtx, (X1 * w[:, None]).T @ y)[1:]
    except np.linalg.LinAlgError:
        return np.full(X.shape[1], np.nan)


# ---------------------------------------------------------------------------------------------
# pysubgroup adapter
# ---------------------------------------------------------------------------------------------


class _PSQF(ps.AbstractInterestingnessMeasure):
    tpl = namedtuple("PathQ", ["size_sg", "quality"])

    def __init__(self, pq: PathQuality):
        self.pq = pq
        self.has_constant_statistics = False
        self.required_stat_attrs = self.tpl._fields

    def calculate_constant_statistics(self, data, target):
        self.N = len(data)
        self.has_constant_statistics = True

    def calculate_statistics(self, subgroup, target, data, statistics=None):
        cover, size = ps.get_cover_array_and_size(subgroup, self.N, data)
        if isinstance(cover, slice):
            return self.tpl(self.N, 0.0)
        mask = np.asarray(cover, dtype=bool)
        if mask.dtype != bool or mask.shape != (self.N,):
            full = np.zeros(self.N, dtype=bool)
            full[np.asarray(cover)] = True
            mask = full
        ev = self.pq.evaluate(mask)
        return self.tpl(int(mask.sum()), ev.quality if ev.valid else 0.0)

    def evaluate(self, subgroup, target, data, statistics=None):
        stats_ = self.ensure_statistics(subgroup, target, data, statistics)
        return stats_.quality if stats_ is not None else 0.0

    def optimistic_estimate(self, subgroup, target, data, statistics=None):
        return np.inf


@dataclass
class Finding:
    description: Description
    quality: float
    evaluation: Evaluation
    mask: np.ndarray
    rank: int = 0

    def to_dict(self) -> dict:
        ev = self.evaluation
        best = ev.best
        return {"description": str(self.description),
                "conditions": [c.to_dict() for c in self.description.conditions],
                "quality": self.quality, "rank": self.rank, "n": ev.n, "n_c": ev.n_c, "n_schools": ev.n_schools,
                "g_eff": ev.g_eff, "g_eff_c": ev.g_eff_c,
                "best_path": best.path if best else None, "paths": [p.to_dict() for p in ev.paths]}


@dataclass
class EMMResult:
    findings: list[Finding]  # canonical, ranked
    raw: list[Finding]  # as returned by the search
    evaluations: int
    descriptors: list[str]
    # How far the search got: candidates scored above 0 / at 0 / too small to fit, and whether the list it kept
    # (``kept_limit`` = 3 x top_k) was still all above 0 when it ended, i.e. positive subgroups may be cut off.
    coverage: dict = field(default_factory=dict)
    # The top_k highest-quality canonical findings without step (d), graded next to ``findings`` to measure what the
    # cover-based selection costs. None for results cached before it existed.
    no_cover: list[Finding] | None = None
    # The top_k with steps (b)+(c) switched off, other steps as configured (equal to ``findings`` when they are off).
    no_prune: list[Finding] | None = None

    def to_dict(self) -> dict:
        return {"findings": [f.to_dict() for f in self.findings], "raw": [f.to_dict() for f in self.raw],
                "evaluations": self.evaluations, "descriptors": self.descriptors, "coverage": self.coverage,
                "no_cover": None if self.no_cover is None else [f.to_dict() for f in self.no_cover],
                "no_prune": None if self.no_prune is None else [f.to_dict() for f in self.no_prune]}


def descriptor_columns(df: pd.DataFrame, latents: list[str], search: SearchSpace) -> list[str]:
    skip = set(latents) | {"school_id"}
    cols = [c for c in df.columns if c not in skip and not c.startswith("_")]
    if search.attributes is not None:
        cols = [c for c in cols if c in search.attributes]
    if search.filter:
        cols = [c for c in cols if c not in search.filter]
    return cols


def run_emm(df: pd.DataFrame, prep: AnyPrepared, emm: EMMConfig, search: SearchSpace, *, seed: int = 0,
            canonical: bool = True) -> EMMResult:
    frame = df.loc[prep.frame].reset_index(drop=True)
    descriptors = descriptor_columns(frame, prep.model.latents, search)
    categorical = {c for c in descriptors if not pd.api.types.is_numeric_dtype(frame[c])}
    sels, quantiles = build_selectors(frame, search, descriptors, categorical)
    pq = PathQuality(prep, emm, search, rng_seed=seed)
    data = frame[descriptors].copy()
    for c in categorical:
        data[c] = data[c].astype(str)
    qf = _PSQF(pq)
    size = max(emm.top_k * 3, emm.top_k)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        task = ps.SubgroupDiscoveryTask(data, None, sels, qf=qf, result_set_size=size, depth=search.depth,
                                        constraints=[ps.MinSupportConstraint(search.min_n)])
        algo = (ps.BeamSearch(beam_width=max(emm.beam_width, 2 * size)) if emm.algorithm == "beam"
                else ps.SimpleSearch(show_progress=False))
        res = algo.execute(task)
    raw = []
    for q, sg, _ in res.results:
        d = subgroup_to_description(sg, quantiles)
        m = d.mask(frame)
        ev = pq.evaluate(m)
        if q > 0 and ev.valid:
            raw.append(Finding(d, float(q), ev, m))
    raw.sort(key=lambda f: -f.quality)
    scored = [e for e in pq._cache.values() if e.valid]
    kept_positive = sum(1 for q, _, _ in res.results if q > 0)
    coverage = {"scored": len(scored), "positive": sum(1 for e in scored if e.quality > 0),
                "zero": sum(1 for e in scored if e.quality <= 0), "too_small": len(pq._cache) - len(scored),
                "kept_limit": size, "kept_positive": kept_positive, "truncated": kept_positive >= size}
    for i, f in enumerate(raw):
        f.rank = i + 1
    if canonical:
        def select(cands: list[Finding]) -> list[Finding]:
            return cover_select(cands, frame, emm.top_k) if emm.cover_selection else cands[: emm.top_k]

        cands = canonical_candidates(raw, pq, frame, emm, prune=emm.prune_refinements)
        findings, no_cover = select(cands), cands[: emm.top_k]
        no_prune = select(canonical_candidates(raw, pq, frame, emm, prune=False)) if emm.prune_refinements \
            else findings
    else:
        findings = no_cover = no_prune = raw[: emm.top_k]
    return EMMResult(_ranked(findings), raw, pq.evaluations, descriptors, coverage, _ranked(no_cover),
                     _ranked(no_prune))


def _ranked(findings: list[Finding]) -> list[Finding]:
    """Copies ranked 1..k, so one Finding can sit in two lists with its own rank in each."""
    return [replace(f, rank=i + 1) for i, f in enumerate(findings)]


# ---------------------------------------------------------------------------------------------
# Canonicalization (system_design §7, R1 §3)
# ---------------------------------------------------------------------------------------------


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    u = np.count_nonzero(a | b)
    return np.count_nonzero(a & b) / u if u else 1.0


def canonical_candidates(raw: list[Finding], pq: PathQuality, frame: pd.DataFrame,
                         emm: EMMConfig, *, prune: bool = True) -> list[Finding]:
    """Steps (b) pruning, (c) generalization filter (both skipped when ``prune`` is off), dedupe and (a) complement
    merging; sorted by quality."""
    def q_of(desc: Description) -> tuple[float, Evaluation, np.ndarray]:
        m = desc.mask(frame)
        ev = pq.evaluate(m)
        return (ev.quality if ev.valid else 0.0), ev, m

    if not prune:
        return _dedupe_merge([Finding(f.description, f.quality, f.evaluation, f.mask) for f in raw], emm)
    # (b) dominance pruning: drop conditions that do not lower quality
    pruned: list[Finding] = []
    for f in raw:
        desc, q, ev, m = f.description, f.quality, f.evaluation, f.mask
        changed = True
        while changed and len(desc.conditions) > 1:
            changed = False
            for i in range(len(desc.conditions)):
                d2 = desc.without(i)
                q2, ev2, m2 = q_of(d2)
                if q2 >= q - 1e-12:
                    desc, q, ev, m = d2, q2, ev2, m2
                    changed = True
                    break
        pruned.append(Finding(desc, q, ev, m))
    # (c) generalization-aware filter: a refinement must beat all of its generalizations
    kept: list[Finding] = []
    for f in pruned:
        conds = f.description.conditions
        if len(conds) > 1:
            gen_best = max(q_of(f.description.without(i))[0] for i in range(len(conds)))
            if f.quality - gen_best <= 0:
                continue
        kept.append(f)
    return _dedupe_merge(kept, emm)


def _dedupe_merge(kept: list[Finding], emm: EMMConfig) -> list[Finding]:
    # dedupe identical descriptions
    uniq: dict[tuple, Finding] = {}
    for f in kept:
        k = f.description.key()
        if k not in uniq or uniq[k].quality < f.quality:
            uniq[k] = f
    cands = sorted(uniq.values(), key=lambda f: -f.quality)
    # (a) merge subgroup/complement pairs (identical split under complement baselines)
    if emm.baseline == "complement":
        merged: list[Finding] = []
        for f in cands:
            if any(jaccard(f.mask, ~g.mask) >= 0.99 for g in merged):
                continue
            merged.append(f)
        cands = merged
    return cands


def cover_select(cands: list[Finding], frame: pd.DataFrame, top_k: int, alpha: float = 0.9) -> list[Finding]:
    """(d) cover-based selection (multiplicative weighted covering, van Leeuwen & Knobbe 2012)."""
    selected: list[Finding] = []
    counts = np.zeros(len(frame), dtype=float)
    pool = list(cands)
    while pool and len(selected) < top_k:
        def score(f: Finding) -> float:
            n = f.mask.sum()
            omega = float((alpha ** counts[f.mask]).sum() / n) if n else 0.0
            return omega * f.quality
        best = max(pool, key=score)
        if score(best) <= 0:
            break
        selected.append(best)
        counts[best.mask] += 1
        pool.remove(best)
    return selected
