"""M1 acceptance: spec, validation, compile, truth engine (implementation_plan §3.1)."""

from __future__ import annotations

import itertools
import math
import time

import networkx as nx
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from emmlab import truth
from emmlab.estimand import EstimandEngine
from emmlab.spec import AnalysisConfig, ResearcherModel, World, dump_yaml, spec_hash, to_plain
from emmlab.world import (
    InfeasibleTargetError,
    WorldValidationError,
    compile_world,
    twist_coding_warnings,
    validate_world,
)
from tests.worlds import PRESETS, world_a, world_b, world_pcase, world_simpson

crit = pytest.mark.criterion


# --------------------------------------------------------------------------- S1–S4: specs


@crit("S1")
@pytest.mark.parametrize("path", sorted(PRESETS.glob("*.yaml")), ids=lambda p: p.stem)
def test_preset_roundtrip_lossless(path):
    from emmlab.spec import load_world
    w = load_world(path)
    again = World.model_validate(load_yaml_text(dump_yaml(to_plain(w))))
    assert to_plain(again) == to_plain(w)


@crit("S1")
def test_yaml_comments_preserved_on_update():
    from emmlab.yamldoc import update_document
    text = (PRESETS / "minimal.yaml").read_text()
    w = world_a()
    d = to_plain(w)
    d["arrows"][0]["slope"] = 0.35
    new_text = update_document(text, d)
    assert "# World A from research/04-path-model-math.md" in new_text
    assert World.model_validate(load_yaml_text(new_text)).arrows[0].slope == 0.35


def load_yaml_text(text: str):
    import io

    from emmlab.spec import _plainify, _yaml
    return _plainify(_yaml().load(io.StringIO(text)))


@crit("S2")
def test_hash_stable_and_sensitive():
    w = world_a()
    d = to_plain(w)
    reordered = dict(reversed(list(d.items())))
    assert spec_hash(d) == spec_hash(reordered) == spec_hash(w)
    w2 = w.model_copy(deep=True)
    w2.arrows[0].slope = 0.31
    assert spec_hash(w2) != spec_hash(w)


def _issues(d: dict) -> str:
    return " | ".join(str(i) for i in validate_world(World.model_validate(d)))


@crit("S3")
@pytest.mark.parametrize("mutate,expect", [
    (lambda d: d["arrows"].append({"from": "achievement", "to": "feedback", "slope": 0.1}), "cycle"),
    (lambda d: d["nodes"]["feedback"].update(root={"mean": 0, "sd": 1}), "give a recipe, not a root"),
    (lambda d: d["nodes"]["selfeff"].pop("recipe"), "has no recipe"),
    (lambda d: d["twists"].append({"arrow": "selfeff -> feedback", "when": {"ses": "rich"}, "delta": 1}),
     "existing arrow"),
    (lambda d: (d["nodes"].update(size={"kind": "attribute", "root": {"mean": 300, "sd": 100}}),
                d["twists"].append({"arrow": "feedback -> selfeff", "when": {"size": "big"}, "delta": 1})),
     "only use categorical"),
    (lambda d: (d["nodes"].update(clim={"level": "school", "recipe": {"noise_sd": 1}}),
                d["arrows"].append({"from": "feedback", "to": "clim", "slope": 1})), "school-level parents"),
    (lambda d: d["attributes"].update(
        a={"values": {"x": .5, "y": .5}, "depends_on": "b", "table": {"x": {"x": .5, "y": .5}, "y": {"x": .5, "y": .5}}},
        b={"values": {"x": .5, "y": .5}, "depends_on": "a", "table": {"x": {"x": .5, "y": .5}, "y": {"x": .5, "y": .5}}}),
     "attribute dependencies form a cycle"),
    (lambda d: d["attributes"]["ses"].update(values={"poor": 0.5, "rich": 0.6}), "sum to 1.1"),
])
def test_validation_rejections(mutate, expect):
    d = to_plain(world_a())
    mutate(d)
    msg = _issues(d)
    assert expect in msg, msg


@crit("S3")
def test_compile_raises_with_element_names():
    d = to_plain(world_a())
    d["arrows"].append({"from": "achievement", "to": "feedback", "slope": 0.1})
    with pytest.raises(WorldValidationError) as e:
        compile_world(World.model_validate(d))
    assert "arrows" in str(e.value)


@crit("S4")
def test_target_sd_solver_world_a():
    d = to_plain(world_a())
    d["nodes"]["achievement"] = {"recipe": {"intercept": 0.0}, "target": {"sd": 1.0}}
    cw = compile_world(World.model_validate(d))
    assert cw.solved["achievement"]["noise_sd"] == pytest.approx(0.927, abs=1e-3)
    _, sig = truth.pooled_moments(cw)
    assert math.sqrt(sig[cw.index["achievement"], cw.index["achievement"]]) == pytest.approx(1.0, abs=1e-12)
    d["nodes"]["achievement"]["target"] = {"sd": 0.37}
    with pytest.raises(InfeasibleTargetError) as e:
        compile_world(World.model_validate(d))
    assert "t_min = 0.376" in str(e.value)


@crit("S4")
def test_target_r2_and_icc():
    d = to_plain(world_a())
    d["nodes"]["selfeff"] = {"recipe": {}, "target": {"r2": 0.10, "icc": 0.05}}
    cw = compile_world(World.model_validate(d))
    r2 = truth.node_r2(cw)["selfeff"]
    icc = truth.node_icc(cw)["selfeff"]
    assert r2 == pytest.approx(0.10, abs=1e-9)
    assert icc == pytest.approx(0.05, abs=1e-9)


# --------------------------------------------------------------------------- TR1–TR5: reference numbers


@crit("TR1")
def test_world_a_cells_and_pooled():
    cw = compile_world(world_a())
    mu, sig, _ = truth.cell_moments(cw)
    rich = [i for i, a in enumerate(cw.assigns) if a["ses"] == "rich"][0]
    poor = 1 - rich
    np.testing.assert_allclose(mu[rich], [0.600, 0.180, 0.425], atol=1e-3)
    np.testing.assert_allclose(np.sqrt(np.diag(sig[poor])), [1.000, 0.949, 0.875], atol=1e-3)
    np.testing.assert_allclose(np.sqrt(np.diag(sig[rich])), [1.000, 0.949, 0.840], atol=1e-3)
    T = truth.total_effects(cw)
    F, Y = cw.index["feedback"], cw.index["achievement"]
    assert T[poor, Y, F] == pytest.approx(0.275, abs=1e-3)
    assert T[rich, Y, F] == pytest.approx(-0.125, abs=1e-3)
    sdp, sdr = np.sqrt(np.diag(sig[poor])), np.sqrt(np.diag(sig[rich]))
    assert cw.B[poor, Y, F] * sdp[F] / sdp[Y] == pytest.approx(0.228, abs=1e-3)
    assert cw.B[rich, Y, F] * sdr[F] / sdr[Y] == pytest.approx(-0.238, abs=1e-3)
    _, s = truth.pooled_moments(cw)
    np.testing.assert_allclose(np.sqrt(np.diag(s)), [1.044, 0.953, 0.884], atol=1e-3)


@crit("TR2")
def test_world_a_step_estimands_and_ovb():
    cw = compile_world(world_a())
    pop = truth.population(cw, "none")
    long_ = truth.ols(pop, "achievement", ["feedback", "selfeff", "ses=rich"])
    short = truth.ols(pop, "achievement", ["feedback", "selfeff"])
    assert short.coef["feedback"] == pytest.approx(0.0523, abs=5e-4)
    assert short.coef["selfeff"] == pytest.approx(0.250, abs=5e-4)
    assert truth.ols(pop, "achievement", ["feedback"]).coef["feedback"] == pytest.approx(0.1273, abs=5e-4)
    assert long_.coef["feedback"] == pytest.approx(0.0, abs=5e-4)
    assert long_.coef["ses=rich"] == pytest.approx(0.380, abs=5e-4)
    aux = truth.ols(pop, "z", []) if False else None  # noqa: F841
    # omitted-variable decomposition: short = long + gamma * aux-slope of the dummy on (F, S)
    from emmlab.truth import Population
    M = np.zeros((4, 3))
    M[:3, :3] = np.eye(3)
    dummy_pop = pop.linear(np.eye(3), names=pop.vars)
    rich = np.array([a["ses"] == "rich" for a in pop.assigns], float)
    ext = Population(pop.vars + ["rich"], pop.w, np.column_stack([pop.mean, rich]),
                     _extend_second(pop, rich), pop.assigns)
    aux = truth.ols(ext, "rich", ["feedback", "selfeff"])
    assert dummy_pop is not None
    recon = long_.coef["feedback"] + long_.coef["ses=rich"] * aux.coef["feedback"]
    assert recon == pytest.approx(short.coef["feedback"], abs=1e-10)


def _extend_second(pop, d):
    G, q = pop.mean.shape
    out = np.zeros((G, q + 1, q + 1))
    out[:, :q, :q] = pop.second
    out[:, :q, q] = pop.mean * d[:, None]
    out[:, q, :q] = pop.mean * d[:, None]
    out[:, q, q] = d * d
    return out


@crit("TR3")
def test_simpson():
    cw = compile_world(world_simpson())
    pop = truth.population(cw, "none")
    assert truth.ols(pop, "achievement", ["feedback"]).coef["feedback"] == pytest.approx(0.30, abs=1e-3)
    for g in ("poor", "rich"):
        sub = pop.restrict(cw.cell_mask({"ses": g}))
        assert truth.ols(sub, "achievement", ["feedback"]).coef["feedback"] == pytest.approx(-0.20, abs=1e-3)


@crit("TR4")
def test_world_b_centered_estimand():
    cw = compile_world(world_b())
    pooled = truth.ols(truth.population(cw, "none"), "achievement", ["feedback", "selfeff"])
    assert pooled.coef["feedback"] == pytest.approx(0.109, abs=5e-4)
    cen = truth.ols(truth.population(cw, "centered", n=None), "achievement", ["feedback", "selfeff"])
    assert cen.coef["feedback"] == pytest.approx(-0.0005, abs=5e-4)
    assert cen.coef["selfeff"] == pytest.approx(0.3267, abs=5e-4)
    # naive Sigma_W = sum pi_g Sigma_W,g (drops group-mean and cross-level terms) gives the wrong 0.325
    mu, sig, A = truth.cell_moments(cw)
    psi_w = np.where(cw.school_level, 0.0, cw.psi)
    naive = np.einsum("g,gij->ij", cw.pi, np.einsum("gij,j,gkj->gik", A, psi_w, A))
    i = [cw.index[v] for v in ("feedback", "selfeff")]
    y = cw.index["achievement"]
    b = np.linalg.solve(naive[np.ix_(i, i)], naive[i, y])
    assert b[1] == pytest.approx(0.325, abs=1e-3)


@crit("TR5")
def test_residualization_total_effect_vs_simple():
    cw = compile_world(world_pcase())
    m = ResearcherModel.from_world(world_pcase())
    res = {}
    for meth in ("total_effect", "simple"):
        an = AnalysisConfig(normalisation="none", residualize_on=["feedback"], residualize_method=meth)
        e = EstimandEngine(cw, m, an, n=None, scale=False)
        res[meth] = (e.coefs[("feedback", "selfeff")], dict(zip(e.full["achievement"].regs,
                                                                e.full["achievement"].coef, strict=True)))
    assert res["total_effect"][0] == pytest.approx(0.300, abs=5e-4)
    assert res["total_effect"][1]["feedback"] == pytest.approx(0.275, abs=5e-4)
    assert res["simple"][0] == pytest.approx(0.460, abs=5e-4)
    assert res["simple"][1]["feedback"] == pytest.approx(0.315, abs=5e-4)


@crit("TR5")
def test_residualization_triangle_and_chain_r2():
    w = world_a()
    cw = compile_world(w)
    tri = ResearcherModel(latents=["feedback", "selfeff", "achievement"],
                          arrows=["feedback -> selfeff", "selfeff -> achievement", "feedback -> achievement"])
    chain = ResearcherModel(latents=tri.latents, arrows=["feedback -> selfeff", "selfeff -> achievement"])
    base = AnalysisConfig(normalisation="none")
    click = AnalysisConfig(normalisation="none", residualize_on=["feedback"])
    r_before = EstimandEngine(cw, tri, base, n=None, scale=False).full["achievement"].r2
    r_after = EstimandEngine(cw, tri, click, n=None, scale=False).full["achievement"].r2
    assert r_after == pytest.approx(r_before, abs=1e-10)
    assert r_before == pytest.approx(0.0874, abs=1e-4)
    c_before = EstimandEngine(cw, chain, base, n=None, scale=False).full["achievement"].r2
    c_after = EstimandEngine(cw, chain, click, n=None, scale=False).full["achievement"].r2
    assert c_before == pytest.approx(0.0840, abs=1e-4)
    assert c_after == pytest.approx(0.0648, abs=1e-4)
    # chain drop = squared semipartial = R2(F,S) - R2(F)
    pop = truth.population(cw, "none")
    r2_fs = truth.ols(pop, "achievement", ["feedback", "selfeff"]).r2
    r2_f = truth.ols(pop, "achievement", ["feedback"]).r2
    assert c_after == pytest.approx(r2_fs - r2_f, abs=1e-10)


# --------------------------------------------------------------------------- TR6–TR10: properties, oracle


@st.composite
def random_world(draw, max_nodes=8, with_attrs=True):
    k = draw(st.integers(2, max_nodes))
    names = [f"v{i}" for i in range(k)]
    edges = [(names[i], names[j]) for i, j in itertools.combinations(range(k), 2)
             if draw(st.booleans())]
    attrs = {}
    if with_attrs:
        for a in range(draw(st.integers(0, 3))):
            nv = draw(st.integers(2, 3))
            ps = np.array([draw(st.floats(0.1, 1.0)) for _ in range(nv)])
            ps = ps / ps.sum()
            vals = {f"a{a}v{i}": float(p) for i, p in enumerate(ps)}
            vals[list(vals)[-1]] = 1.0 - sum(list(vals.values())[:-1])
            attrs[f"a{a}"] = {"level": draw(st.sampled_from(["school", "student"])), "values": vals}
    shifts, twists = [], []
    for a, spec in attrs.items():
        for v in list(spec["values"])[1:]:
            if draw(st.booleans()):
                shifts.append({"when": {a: v}, "target": draw(st.sampled_from(names)),
                               "delta": draw(st.floats(-1, 1))})
            if edges and draw(st.booleans()):
                p, c = draw(st.sampled_from(edges))
                twists.append({"arrow": f"{p} -> {c}", "when": {a: v}, "delta": draw(st.floats(-0.5, 0.5))})
    has_in = {c for _, c in edges} | {s["target"] for s in shifts}
    nodes = {}
    for n in names:
        if n in has_in:
            nodes[n] = {"recipe": {"intercept": draw(st.floats(-1, 1)), "noise_sd": draw(st.floats(0.3, 1.5))},
                        "school_sd": draw(st.sampled_from([0.0, 0.0, 0.3]))}
        else:
            nodes[n] = {"root": {"mean": draw(st.floats(-1, 1)), "sd": draw(st.floats(0.3, 1.5))}}
    arrows = [{"from": p, "to": c, "slope": draw(st.floats(-0.8, 0.8))} for p, c in edges]
    return World.model_validate({"name": "rand", "attributes": attrs, "nodes": nodes, "arrows": arrows,
                                 "shifts": shifts, "twists": twists,
                                 "schools": {"n_schools": 400, "students_per_school": {"n": 25}}})


@crit("TR6")
@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(random_world(max_nodes=12))
def test_total_effects_equal_path_enumeration(w):
    cw = compile_world(w)
    T = truth.total_effects(cw)
    g = nx.DiGraph()
    g.add_edges_from((a.parent, a.child) for a in w.arrows)
    for gi in range(len(cw.assigns)):
        B = cw.B[gi]
        for s in w.nodes:
            for d in w.nodes:
                if s == d:
                    continue
                tot = 0.0
                if s in g and d in g:
                    for path in nx.all_simple_paths(g, s, d):
                        prod = 1.0
                        for a, b in itertools.pairwise(path):
                            prod *= B[cw.index[b], cw.index[a]]
                        tot += prod
                assert T[gi, cw.index[d], cw.index[s]] == pytest.approx(tot, abs=1e-10)


def _batch_z(est_batches: np.ndarray, analytic: np.ndarray) -> np.ndarray:
    m = est_batches.mean(0)
    se = est_batches.std(0, ddof=1) / math.sqrt(len(est_batches))
    return (m - analytic) / np.where(se > 0, se, np.inf)


@crit("TR7")
@pytest.mark.slow
@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(random_world(max_nodes=6))
def test_analytic_moments_match_simulation(w):
    from emmlab.generate import generate
    cw = compile_world(w)
    n_schools, per = 20_000, 50  # 10^6 students
    df = generate(cw, 12345, n_schools=n_schools, students=per)
    vis = cw.visible
    X = df[vis].to_numpy()
    batches = np.array_split(np.arange(n_schools), 50)
    sid = df["school_id"].to_numpy()
    means, covs = [], []
    for b in batches:
        rows = (sid >= b[0]) & (sid <= b[-1])
        means.append(X[rows].mean(0))
        covs.append(np.cov(X[rows], rowvar=False, bias=True)[np.triu_indices(len(vis))])
    mu, sig = truth.pooled_moments(cw)
    idx = [cw.index[v] for v in vis]
    zm = _batch_z(np.array(means), mu[idx])
    zc = _batch_z(np.array(covs), sig[np.ix_(idx, idx)][np.triu_indices(len(vis))])
    z = np.abs(np.concatenate([zm, zc]))
    assert (z < 4).mean() >= 0.99 or len(z) < 20 and (z < 4).all(), z
    assert (z < 6).all(), z


@crit("TR8")
@pytest.mark.oracle
@settings(max_examples=10, deadline=None)
@given(random_world(max_nodes=7, with_attrs=False))
def test_population_covariance_matches_sempler(w):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import sempler
    cw = compile_world(w)
    if any(v.startswith("u:") for v in cw.vars):  # school intercepts add correlated noise; skip
        return
    W = cw.B[0].T
    means = cw.alpha[0]
    lg = sempler.LGANM(W, means, cw.psi)
    d = lg.sample(population=True)
    _, sig = truth.pooled_moments(cw)
    np.testing.assert_allclose(sig, d.covariance, atol=1e-8)
    np.testing.assert_allclose(truth.pooled_moments(cw)[0], d.mean, atol=1e-8)


@crit("TR9")
def test_twist_coding_equivalence_and_warning():
    a = to_plain(world_a())
    # recode: baseline = rich slope (-0.20), twist on poor (+0.40) -> identical cell slopes
    b = to_plain(world_a())
    b["arrows"][2]["slope"] = -0.20
    b["twists"] = [{"arrow": "feedback -> achievement", "when": {"ses": "poor"}, "delta": 0.40}]
    ca, cb = compile_world(World.model_validate(a)), compile_world(World.model_validate(b))
    np.testing.assert_allclose(ca.B, cb.B, atol=1e-15)
    c = to_plain(world_a())
    c["twists"].append({"arrow": "feedback -> achievement", "when": {"ses": "poor"}, "delta": 0.1})
    assert twist_coding_warnings(World.model_validate(c))


@crit("TR10")
def test_truth_latency_30_nodes_50_cells():
    nodes = {f"v{i}": ({"root": {"mean": 0, "sd": 1}} if i == 0 else {"recipe": {"noise_sd": 1}})
             for i in range(30)}
    arrows = [{"from": f"v{i}", "to": f"v{i + 1}", "slope": 0.3} for i in range(29)]
    arrows += [{"from": f"v{i}", "to": f"v{i + 2}", "slope": 0.1} for i in range(28)]
    attrs = {"a": {"values": {f"x{i}": 0.2 for i in range(5)}},
             "b": {"level": "school", "values": {f"y{i}": 0.1 for i in range(10)}}}
    twists = [{"arrow": "v3 -> v4", "when": {"a": "x1"}, "delta": 0.2},
              {"arrow": "v10 -> v11", "when": {"b": "y3"}, "delta": -0.2}]
    w = World.model_validate({"name": "big", "nodes": nodes, "arrows": arrows, "attributes": attrs,
                              "twists": twists})
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        cw = compile_world(w)
        truth.pooled_moments(cw)
        truth.total_effects(cw)
        times.append(time.perf_counter() - t0)
    assert len(cw.assigns) == 50
    assert np.percentile(times, 95) < 0.020, np.percentile(times, 95)


@crit("S2")
def test_layout_direction_is_presentation_only():
    w = world_a()
    d = w.model_dump(by_alias=True)
    d["layout_direction"] = "RIGHT"
    w2 = World.model_validate(d)
    assert spec_hash(w2) == spec_hash(w) and to_plain(w2)["layout_direction"] == "RIGHT"
    assert "layout_direction" not in to_plain(w)  # unset fields stay out of YAML and hashes of existing Worlds
