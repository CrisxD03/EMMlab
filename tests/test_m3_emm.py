"""M3 acceptance: EMM, presets, canonicalization, grading, flags, bootstrap (implementation_plan §3.3)."""

from __future__ import annotations

import inspect
import math
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from emmlab import emm as emm_mod
from emmlab.bootstrap import recompute_description, resample_schools, school_bootstrap, strata_of
from emmlab.emm import (
    _PSQF,
    Condition,
    Description,
    Evaluation,
    Finding,
    PathQuality,
    PathStat,
    entropy_weight,
    run_emm,
)
from emmlab.estimand import EstimandEngine
from emmlab.experiment import aggregate, execute
from emmlab.generate import generate, seed_sequence
from emmlab.grade import TruthProvider, _oracle_data, _oracle_prepared, classify, grade
from emmlab.run import cluster_tier, effective_schools, execute_run
from emmlab.spec import (
    AnalysisConfig,
    BootstrapConfig,
    EMMConfig,
    ExperimentSpec,
    GradingConfig,
    ResearcherModel,
    RunSpec,
    SearchSpace,
    World,
    load_world,
    preset,
    to_plain,
)
from emmlab.stages import ols_fit, prepare
from emmlab.world import compile_world
from tests.worlds import PRESETS, world_a

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import pysubgroup as ps

crit = pytest.mark.criterion


def _small():
    rng = np.random.default_rng(0)
    n = 600
    g = np.where(np.arange(n) % 3 == 0, "f", "m")
    x = rng.normal(size=n)
    y = np.where(g == "f", 0.5, 0.1) * x + rng.normal(size=n)
    df = pd.DataFrame({"school_id": np.arange(n) // 20, "gender": g, "x": x, "y": y})
    model = ResearcherModel(latents=["x", "y"], arrows=["x -> y"])
    prep = prepare(df, model, AnalysisConfig(normalisation="none"))
    return df, prep


@crit("E1")
def test_quality_functions_hand_computed():
    df, prep = _small()
    mask = (df["gender"] == "f").to_numpy()
    X, y, sch = prep.X["y"], prep.y["y"], prep.school
    bg, _, seg, _, _ = ols_fit(X[mask], y[mask], sch[mask])
    bc, _, sec, _, _ = ols_fit(X[~mask], y[~mask], sch[~mask])
    H = -(1 / 3) * math.log2(1 / 3) - (2 / 3) * math.log2(2 / 3)
    ss = SearchSpace(min_n=10, min_n_complement=10)
    z = abs(bg[0] - bc[0]) / math.sqrt(seg[0] ** 2 + sec[0] ** 2)
    for q, expect in [("ent_dbeta", H * abs(bg[0] - bc[0])), ("wald_z", z), ("ent_z", H * z)]:
        pq = PathQuality(prep, EMMConfig(quality=q), ss)
        assert pq.evaluate(mask).quality == pytest.approx(expect, abs=1e-10)
    z = stats.norm.ppf(0.975)
    disjoint = (bg[0] + z * seg[0] < bc[0] - z * sec[0]) or (bc[0] + z * sec[0] < bg[0] - z * seg[0])
    pq = PathQuality(prep, EMMConfig(quality="current"), ss, per_group=False)
    # the researcher's quality function: natural-log entropy x number of disjoint paths (here one path)
    assert pq.evaluate(mask).quality == pytest.approx(H * math.log(2) * float(disjoint), abs=1e-10)
    assert entropy_weight(300, 600) == pytest.approx(1.0)


@crit("E1")
def test_cook_quality_hand_computed():
    df, prep = _small()
    mask = (df["gender"] == "f").to_numpy()
    pq = PathQuality(prep, EMMConfig(quality="cook", baseline="population"), SearchSpace(min_n=10, min_n_complement=10),
                     per_group=False)
    ev = pq.evaluate(mask)
    X1 = np.column_stack([np.ones(prep.n), prep.X["y"]])
    inv = np.linalg.inv(X1.T @ X1)
    full = prep.full["y"]
    s2 = (1 - full.r2) * prep.y["y"].var()
    bg, *_ = ols_fit(prep.X["y"][mask], prep.y["y"][mask], prep.school[mask])
    d = bg[0] - full.coef[0]
    assert ev.quality == pytest.approx(d * d / (inv[1, 1] * s2), rel=1e-10)


@crit("E2")
@pytest.mark.parametrize("algo", ["beam", "simple", "dfs"])
def test_adapter_handles_cover_types(algo):
    df, prep = _small()
    pq = PathQuality(prep, EMMConfig(quality="ent_dbeta"), SearchSpace(min_n=10, min_n_complement=10))
    data = df[["gender"]].astype(str)
    task = ps.SubgroupDiscoveryTask(data, None, [ps.EqualitySelector("gender", "f"),
                                                 ps.EqualitySelector("gender", "m")],
                                    qf=_PSQF(pq), result_set_size=2, depth=1)
    a = {"beam": ps.BeamSearch(beam_width=4), "simple": ps.SimpleSearch(show_progress=False),
         "dfs": ps.DFS()}[algo]
    res = a.execute(task)
    assert res.results and res.results[0][0] > 0


@crit("E2")
def test_generalization_aware_never_with_beam():
    assert "GeneralizationAwareQF" not in inspect.getsource(emm_mod)


@crit("E3")
def test_min_sizes_enforced_inside_qf():
    df, prep = _small()
    mask = (df["gender"] == "f").to_numpy()
    for ss in (SearchSpace(min_n=250), SearchSpace(min_n=10, min_n_complement=500),
               SearchSpace(min_n=10, min_n_complement=10, min_schools=31)):
        ev = PathQuality(prep, EMMConfig(), ss).evaluate(mask)
        assert not ev.valid and ev.quality == 0.0


def _twist_world(delta: float = 0.4, share: float = 0.5) -> World:
    return World.model_validate({
        "name": "strong", "schools": {"n_schools": 200, "students_per_school": {"n": 35}},
        "attributes": {"gender": {"values": {"m": 1 - share, "f": share}},
                       "football": {"role": "decoy", "values": {"no": 0.5, "yes": 0.5}},
                       "school_type": {"level": "school", "values": {"a": 0.5, "b": 0.5}}},
        "nodes": {"feedback": {"root": {"mean": 0, "sd": 1}},
                  "achievement": {"recipe": {"noise_sd": math.sqrt(0.91)}}},
        "arrows": [{"feedback -> achievement": 0.3}],
        "twists": [{"arrow": "feedback -> achievement", "when": {"gender": "f"}, "delta": delta, "label": "girls"}]})


@crit("E4")
@pytest.mark.slow
def test_strong_signal_recovery():
    w = _twist_world(0.4)
    spec = ExperimentSpec(name="E4", base=RunSpec(world=w, model=ResearcherModel.from_world(w),
                                                  method=preset("recommended")),
                          presets=["recommended"], n_sim=200, include_null_world=False)
    agg = aggregate(spec, execute(spec))
    hr = agg["conditions"][0]["presets"]["recommended"]["twists"]["girls"]["hit_rate"]
    assert hr["value"] >= 0.95, hr


@crit("E5")
def test_variance_artifact_truth():
    w = load_world(PRESETS / "variance_decoy.yaml")
    cw = compile_world(w)
    m = ResearcherModel.from_world(w)
    tp = TruthProvider(cw, m, preset("current").analysis, None, GradingConfig(oracle_students=2_000_000))
    df, prep = tp.oracle
    mask = df["ses"].to_numpy() >= stats.norm.ppf(0.75)
    g = prep.fit(mask, per_group=True, cluster=False)["achievement"].coef[0]
    c = prep.fit(~mask, per_group=True, cluster=False)["achievement"].coef[0]
    assert g == pytest.approx(0.20, abs=0.02) and c == pytest.approx(0.28, abs=0.02)
    rg = prep.fit(mask, per_group=False, cluster=False)["achievement"].coef[0]
    rc = prep.fit(~mask, per_group=False, cluster=False)["achievement"].coef[0]
    assert rg == pytest.approx(rc, abs=0.005)


@crit("E5")
@pytest.mark.slow
def test_variance_decoy_false_alarm_rates():
    w = load_world(PRESETS / "variance_decoy.yaml")
    pure = to_plain(w)
    pure["arrows"] = [{"from": "feedback", "to": "achievement", "slope": 0.40}]
    pure["nodes"]["feedback"] = {"root": {"mean": 0, "sd": 1}}
    spec = ExperimentSpec(name="E5", base=RunSpec(world=w, model=ResearcherModel.from_world(w),
                                                  method=preset("recommended")),
                          presets=["current", "recommended"], n_sim=200, null_world=World.model_validate(pure))
    agg = aggregate(spec, execute(spec))["conditions"][0]["presets"]
    rec, cur = agg["recommended"], agg["current"]
    fa, fwer = rec["planted_any_significant"], rec["fwer_null"]
    assert fa["value"] <= fwer["value"] + 2 * math.sqrt(fa["mcse"] ** 2 + fwer["mcse"] ** 2), (fa, fwer)
    assert cur["planted_any_significant"]["value"] > fa["value"]


@crit("E6")
@pytest.mark.parametrize("n_bins,one_sided,expect", [(5, False, 0.97), (10, False, 0.48), (5, True, 0.97)])
def test_binning_sensitivity(n_bins, one_sided, expect):
    from emmlab.emm import jaccard
    from emmlab.spec import Binning
    rng = np.random.default_rng(1)
    n = 20000
    ses, x = rng.normal(size=n), rng.normal(size=n)
    tw = ses > 0.8
    y = np.where(tw, 0.3, -0.1) * x + rng.normal(size=n)
    df = pd.DataFrame({"school_id": np.arange(n) // 25, "ses": ses, "x": x, "y": y})
    model = ResearcherModel(latents=["x", "y"], arrows=["x -> y"])
    prep = prepare(df, model, AnalysisConfig(normalisation="none", se="naive"))
    ss = SearchSpace(depth=1, default_binning=Binning(n_bins=n_bins, one_sided=one_sided))
    f = run_emm(df, prep, EMMConfig(quality="ent_dbeta", top_k=1), ss, canonical=False).findings[0]
    j = max(jaccard(f.mask, tw), jaccard(~f.mask, tw))
    assert j == pytest.approx(expect, abs=0.05)
    if one_sided:
        assert j >= 0.9


@crit("E7")
@pytest.mark.slow
def test_baseline_lesson_level_shifts():
    w = load_world(PRESETS / "level_shift_only.yaml")
    res = {}
    for base in ("complement", "population"):
        method = preset("recommended")
        method.analysis.normalisation = "none"
        method.emm.baseline = base
        # depth 1: the vision's lesson is about single-attribute subgroups (a conjunction's complement
        # mixes groups with different levels, so depth-2 complements leak; see docs/findings.md)
        spec = ExperimentSpec(name=f"E7{base}", base=RunSpec(world=w, model=ResearcherModel.from_world(w),
                                                             method=method, search=SearchSpace(depth=1)),
                              presets=["recommended"], n_sim=200, include_null_world=False)
        res[base] = aggregate(spec, execute(spec))["conditions"][0]["presets"]["recommended"]["top1_attrs"]
    st_pop = res["population"]["school_type"]
    st_comp = res["complement"].get("school_type", {"value": 0.0, "mcse": 0.0})
    assert st_pop["value"] >= 0.90
    assert st_comp["value"] <= 0.5  # chance level among three attributes, not systematic
    diff = st_pop["value"] - st_comp["value"]
    assert diff > 4 * math.sqrt(st_pop["mcse"] ** 2 + st_comp["mcse"] ** 2)
    # the analytic truth: no slope difference vs the complement, a real one vs the confounded population
    cw = compile_world(w)
    an = AnalysisConfig(normalisation="none")
    e = EstimandEngine(cw, ResearcherModel.from_world(w), an)
    m = e.mask({"school_type": "academic"})
    g, c, full = e.fit(m)["achievement"].coef[0], e.fit(~m)["achievement"].coef[0], e.full["achievement"].coef[0]
    assert abs(g - c) < 1e-10 and abs(g - full) > 0.1


@crit("E8")
def test_preset_fidelity():
    cur = preset("current")
    assert cur.analysis.normalisation == "none"  # both presets start pooled (Oct 2026); centering is selectable
    assert preset("recommended").analysis.normalisation == "none"
    assert cur.analysis.standardize == "per_group"
    assert cur.analysis.residualize_method == "simple"
    assert cur.emm.quality == "current" and cur.emm.baseline == "complement"
    assert cur.emm.nominal_min_schools == 25
    assert SearchSpace().default_binning.weighting == "students"
    rec = preset("recommended")
    assert (rec.analysis.standardize, rec.analysis.residualize_method, rec.emm.quality) == \
           ("global_once", "total_effect", "ent_z")


@crit("C1")
def test_canonicalization():
    w = load_world(PRESETS / "pisa_like.yaml")
    m = ResearcherModel.from_world(w)
    df = generate(compile_world(w), seed_sequence(3))
    method = preset("recommended")
    method.emm.algorithm = "exhaustive"
    prep = prepare(df, m, method.analysis)
    res = run_emm(df, prep, method.emm, SearchSpace(depth=2))
    raw_desc = [str(f.description) for f in res.raw]
    can = [str(f.description) for f in res.findings]
    assert any("gender" in d and "AND" in d for d in raw_desc)  # free riders present before
    assert not any("gender" in d and "AND" in d for d in can)  # ...gone after
    assert sum(d.startswith("gender=") for d in can) == 1  # f/m pair merged
    assert len(can) <= method.emm.top_k


def _fake(desc: Description, df: pd.DataFrame, path: str, diff: float, quality: float = 1.0,
          rank: int = 1) -> Finding:
    r, c = [s.strip() for s in path.split("->")]
    ev = Evaluation(quality, 0, 0, 0, 0, 0, [PathStat(c, r, diff, 0.01, 0.0, 0.01)])
    return Finding(desc, quality, ev, desc.mask(df), rank)


@crit("GR1")
def test_hit_rule():
    w = world_a()
    cw = compile_world(w)
    m = ResearcherModel.from_world(w)
    df = generate(cw, 1)
    tp = TruthProvider(cw, m, preset("recommended").analysis, None, GradingConfig())
    rich = Description((Condition("ses", "==", "rich"),))
    poor = Description((Condition("ses", "==", "poor"),))
    path = "feedback -> achievement"
    from emmlab.emm import EMMResult
    good = grade(w, m, EMMResult([_fake(rich, df, path, -0.4)], [], 0, []), df, tp, GradingConfig(), "complement")
    assert len(good.hits) == 1
    flipped = grade(w, m, EMMResult([_fake(poor, df, path, 0.4)], [], 0, []), df, tp, GradingConfig(), "complement")
    assert len(flipped.hits) == 1  # complement orientation: poor & +0.4 == rich & -0.4
    wrong = grade(w, m, EMMResult([_fake(rich, df, path, 0.4)], [], 0, []), df, tp, GradingConfig(), "complement")
    assert not wrong.hits and wrong.misses
    two = grade(w, m, EMMResult([_fake(rich, df, path, -0.4), _fake(poor, df, path, 0.4, rank=2)], [], 0, []),
                df, tp, GradingConfig(), "complement")
    assert len(two.hits) == 1 and len(two.redundant) == 1 and not two.false_positives


def _classify_world(world: dict, desc: Description, analysis: AnalysisConfig, baseline: str = "complement",
                    diff: float = 0.1, path: str = "feedback -> achievement") -> str:
    w = World.model_validate(world)
    cw = compile_world(w)
    m = ResearcherModel.from_world(w)
    df = generate(cw, 5)
    tp = TruthProvider(cw, m, analysis, None, GradingConfig(oracle_students=1_000_000))
    from emmlab.grade import planted_twists
    planted = planted_twists(w, m, tp, df, baseline)
    import networkx as nx
    decoys = {n for n, a in w.attributes.items() if a.role == "decoy"}
    return classify(_fake(desc, df, path, diff), planted, decoys, nx.DiGraph(), tp, GradingConfig(), baseline)


BASE = {"name": "c", "nodes": {"feedback": {"root": {"mean": 0, "sd": 1}},
                               "achievement": {"recipe": {"noise_sd": 1}}},
        "arrows": [{"feedback -> achievement": 0.3}]}


@crit("GR2")
def test_false_positive_taxonomy():
    an = AnalysisConfig(normalisation="none")
    g = {**BASE, "attributes": {"gender": {"values": {"m": .5, "f": .5}},
                                "football": {"role": "decoy", "values": {"no": .5, "yes": .5}}}}
    assert _classify_world(g, Description((Condition("gender", "==", "f"),)), an) == "null"
    assert _classify_world(g, Description((Condition("football", "==", "yes"),)), an) == "decoy"
    tw = {**g, "twists": [{"arrow": "feedback -> achievement", "when": {"gender": "f"}, "delta": 0.3}]}
    assert _classify_world(tw, Description((Condition("gender", "==", "f"),)), an, diff=-0.3) == \
        "phantom_downstream"
    vd = to_plain(load_world(PRESETS / "variance_decoy.yaml"))
    assert _classify_world(vd, Description((Condition("ses", ">=", float(stats.norm.ppf(.75)), quantile=.75),)),
                           preset("current").analysis) == "variance_artifact"
    ls = to_plain(load_world(PRESETS / "level_shift_only.yaml"))
    assert _classify_world(ls, Description((Condition("school_type", "==", "academic"),)), an,
                           baseline="population") == "level_shift_leak"
    px = {**BASE, "attributes": {
        "school_type": {"level": "school", "values": {"a": .5, "b": .5}},
        "track": {"values": {"x": .5, "y": .5}, "depends_on": "school_type",
                  "table": {"a": {"x": .8, "y": .2}, "b": {"x": .2, "y": .8}}}},
        "twists": [{"arrow": "feedback -> achievement", "when": {"school_type": "a"}, "delta": 0.3}]}
    assert _classify_world(px, Description((Condition("track", "==", "x"),)), an) == "proxy"


@crit("GR3")
def test_threshold_findings_use_cached_oracle():
    w = load_world(PRESETS / "variance_decoy.yaml")
    cw = compile_world(w)
    m = ResearcherModel.from_world(w)
    _oracle_data.cache_clear()
    _oracle_prepared.cache_clear()
    d = Description((Condition("ses", ">=", 0.5, quantile=0.7),))
    for _ in range(2):
        tp = TruthProvider(cw, m, preset("recommended").analysis, None, GradingConfig(oracle_students=200_000))
        assert tp.contrast(d, "complement")["source"] == "oracle"
    # simulated once, prepared once: the second provider reuses the prepared oracle
    assert _oracle_data.cache_info().misses == 1
    info = _oracle_prepared.cache_info()
    assert info.misses == 1 and info.hits >= 1


@crit("GR4")
def test_scorecard_schema():
    w = _twist_world(0.4)
    spec = ExperimentSpec(name="GR4", base=RunSpec(world=w, model=ResearcherModel.from_world(w),
                                                   method=preset("recommended")),
                          presets=["recommended"], n_sim=12)
    agg = aggregate(spec, execute(spec, workers=1))
    p = agg["conditions"][0]["presets"]["recommended"]
    for k in ("twists", "fdp", "decoy_rate", "fwer_null", "estimation", "threshold"):
        assert k in p
    t = p["twists"]["girls"]
    for k in ("hit_rate", "mean_rank", "est_diff", "true_diff", "winners_curse", "unstable"):
        assert k in t
    assert t["hit_rate"]["mcse"] is not None and p["fwer_null"]["mcse"] is not None
    row = p["estimation"][0]
    for k in ("bias", "emp_se", "rmse", "coverage", "rel_bias", "flags"):
        assert k in row
    assert set(p["fdp"]) >= {"null", "decoy", "phantom_downstream", "variance_artifact", "level_shift_leak",
                             "proxy"}


@crit("F1")
def test_flags():
    codes = np.repeat(np.arange(60), 10)
    student_mask = np.zeros(600, bool)
    student_mask[np.arange(0, 600, 10)] = True  # one member per school -> no school with >= 3
    student_mask[:30] = True  # 3 schools fully
    e = effective_schools(codes, student_mask, 60)
    assert e["g_eff"] == 3 and e["schools"] == 60 and e["tier"] == "unreliable"
    school_mask = codes < 30
    assert effective_schools(codes, school_mask, 60)["g_eff"] == 30
    assert cluster_tier(19) == "unreliable" and cluster_tier(20) == "fragile" and cluster_tier(50) == "ok"
    w = load_world(PRESETS / "pisa_like.yaml")
    spec = RunSpec(world=w, model=ResearcherModel.from_world(w), method=preset("current"), seed=2)
    r = execute_run(spec)
    for f in r["flags"]:
        assert "current_rule_fragile" in f
        assert f["g_eff"] <= min(f["subgroup"]["g_eff"], f["complement"]["g_eff"])


@crit("B1")
def test_stratified_resampling_and_recompute():
    w = load_world(PRESETS / "pisa_like.yaml")
    df = generate(compile_world(w), 9)
    strata = strata_of(df, ["school_type"])
    rng = np.random.default_rng(0)
    for _ in range(50):
        b = resample_schools(df, strata, rng)
        assert b.groupby("school_id")["school_type"].first().value_counts().to_dict() == \
            strata.value_counts().to_dict()
    d = Description((Condition("school_size", ">=", 700.0, quantile=0.6),))
    b = resample_schools(df, strata, rng)
    dd = recompute_description(d, b)
    assert dd.conditions[0].value == pytest.approx(np.quantile(b["school_size"], 0.6))


@crit("B2")
@pytest.mark.slow
def test_bootstrap_coverage():
    from emmlab.parallel import pmap
    covered = pmap(_boot_rep, range(500))
    rate = np.mean(covered)
    assert 0.92 <= rate <= 0.98, rate


def _boot_rep(k: int) -> bool:
    w = world_a()
    cw = compile_world(w)
    m = ResearcherModel.from_world(w)
    an = AnalysisConfig(normalisation="centered")
    t3 = EstimandEngine(cw, m, an).full["achievement"]
    df = generate(cw, seed_sequence(4242, k))
    b = school_bootstrap(df, m, an, None, ["ses"], [], "complement", BootstrapConfig(reps=399), seed_sequence(1, k))
    lo, hi = b.full["feedback -> achievement"]["ci"]
    return bool(lo <= t3.coef[t3.regs.index("feedback")] <= hi)


@crit("B3")
@pytest.mark.slow
def test_current_vs_current_exact_rankings():
    from emmlab.parallel import pmap
    taus = pmap(_b3_rep, range(50))
    assert np.mean(taus) >= 0.8, taus


def _b3_rep(k: int) -> float:
    w = load_world(PRESETS / "pisa_like.yaml")
    m = ResearcherModel.from_world(w)
    df = generate(compile_world(w), seed_sequence(31, k))
    method = preset("current")
    prep = prepare(df, m, method.analysis)
    ss = SearchSpace(depth=1)
    fast = run_emm(df, prep, method.emm, ss, canonical=False)
    exact_cfg = method.emm.model_copy(update={"quality": "current_exact"})
    pq = PathQuality(prep, exact_cfg, ss, rng_seed=k)
    top = fast.raw[:5]
    if len(top) < 2:
        return 1.0
    q_fast = [f.quality for f in top]
    q_exact = [pq.evaluate(f.mask).quality for f in top]
    if len(set(q_fast)) == 1 and len(set(q_exact)) == 1:
        return 1.0
    tau = stats.kendalltau(q_fast, q_exact).statistic
    return 1.0 if np.isnan(tau) else float(tau)


def _two_path_data():
    rng = np.random.default_rng(3)
    n = 4000
    g = np.where(np.arange(n) % 4 == 0, "f", "m")
    x = rng.normal(size=n)
    y = np.where(g == "f", 0.8, 0.1) * x + rng.normal(size=n)
    z = np.where(g == "f", -0.7, 0.2) * x + rng.normal(size=n)
    w = 0.3 * x + rng.normal(size=n)  # same slope in both groups
    df = pd.DataFrame({"school_id": np.arange(n) // 20, "gender": g, "x": x, "y": y, "z": z, "w": w})
    model = ResearcherModel(latents=["x", "y", "z", "w"], arrows=["x -> y", "x -> z", "x -> w"])
    return df, prepare(df, model, AnalysisConfig(normalisation="none", standardize="global_once"))


@crit("E9")
def test_current_is_entropy_times_number_of_disjoint_paths():
    """The researcher's SEMQFEntropyCIComplement: ln-entropy x (number of paths with disjoint intervals)."""
    df, prep = _two_path_data()
    mask = (df["gender"] == "f").to_numpy()
    ss = SearchSpace(min_n=10, min_n_complement=10)
    pq = PathQuality(prep, EMMConfig(quality="current", interval_se="naive"), ss, per_group=False)
    ev = pq.evaluate(mask)
    z = stats.norm.ppf(0.975)
    n_disjoint = 0
    for child in ("y", "z", "w"):
        X, y_, sch = prep.X[child], prep.y[child], prep.school
        bg, _, _, seg, _ = ols_fit(X[mask], y_[mask], sch[mask])  # naive SE is the 4th return value
        bc, _, _, sec, _ = ols_fit(X[~mask], y_[~mask], sch[~mask])
        lo_g, hi_g, lo_b, hi_b = bg[0] - z * seg[0], bg[0] + z * seg[0], bc[0] - z * sec[0], bc[0] + z * sec[0]
        n_disjoint += int(hi_g < lo_b or hi_b < lo_g)
    assert n_disjoint >= 2  # y and z are planted; a count differs from the old 0/1 flag
    p = mask.mean()
    h_nat = -p * math.log(p) - (1 - p) * math.log(1 - p)
    assert ev.quality == pytest.approx(h_nat * n_disjoint, abs=1e-10)
    assert sum(ps_.disjoint for ps_ in ev.paths) == n_disjoint


@crit("E9")
def test_current_preset_uses_student_level_intervals_inside_the_search():
    """plspm's inner-model intervals ignore school clustering; the replica does the same, `recommended` does not."""
    assert preset("current").emm.interval_se == "naive"
    assert preset("recommended").emm.interval_se == "analysis"
    df, prep = _two_path_data()
    mask = (df["gender"] == "f").to_numpy()
    ss = SearchSpace(min_n=10, min_n_complement=10)
    naive = PathQuality(prep, EMMConfig(quality="current", interval_se="naive"), ss, per_group=False).evaluate(mask)
    robust = PathQuality(prep, EMMConfig(quality="current", interval_se="analysis"), ss, per_group=False).evaluate(mask)
    assert prep.analysis.se == "cluster"
    assert [p.se_g for p in naive.paths] != [p.se_g for p in robust.paths]
    for pn, pr in zip(naive.paths, robust.paths, strict=True):
        assert pn.b_g == pytest.approx(pr.b_g, abs=1e-12)  # same coefficients, different standard errors


@crit("E10")
def test_search_coverage_says_whether_the_kept_list_reached_zero():
    df, prep = _small()
    data = df.assign(club=np.where(np.arange(len(df)) % 2 == 0, "a", "b"))
    # few candidates, many slots: the kept list is not full, nothing positive was cut off
    res = run_emm(data, prep, EMMConfig(quality="current", top_k=10), SearchSpace(min_n=10, min_n_complement=10, depth=1))
    cov = res.coverage
    assert cov["kept_limit"] == 30 and cov["truncated"] is False
    assert cov["scored"] == cov["positive"] + cov["zero"] and cov["scored"] + cov["too_small"] >= 1
    # one slot and a quality that is (almost) never 0: the kept list is full of positives -> truncated
    res2 = run_emm(data, prep, EMMConfig(quality="ent_z", top_k=1), SearchSpace(min_n=10, min_n_complement=10, depth=2))
    assert res2.coverage["kept_limit"] == 3 and res2.coverage["truncated"] is True
    assert res2.to_dict()["coverage"] == res2.coverage


@crit("E11")
def test_cover_selection_can_be_switched_off_and_both_lists_are_kept():
    w = load_world(PRESETS / "pisa_like.yaml")
    df = generate(compile_world(w), 0)
    method = preset("recommended")
    prep = prepare(df, ResearcherModel.from_world(w), method.analysis)
    on = run_emm(df, prep, method.emm, SearchSpace())
    # without step (d): the top k by quality, ranked 1..k in its own list
    q = [f.quality for f in on.no_cover]
    assert on.no_cover is not None and q == sorted(q, reverse=True) and len(q) <= method.emm.top_k
    assert [f.rank for f in on.no_cover] == list(range(1, len(q) + 1))
    assert [f.rank for f in on.findings] == list(range(1, len(on.findings) + 1))
    # with (d) on (the default) the reported list differs from it on this draw
    assert [str(f.description) for f in on.findings] != [str(f.description) for f in on.no_cover]
    off = run_emm(df, prep, method.emm.model_copy(update={"cover_selection": False}), SearchSpace())
    assert [str(f.description) for f in off.findings] == [str(f.description) for f in on.no_cover]
    assert on.to_dict()["no_cover"] == [f.to_dict() for f in on.no_cover]
    # steps (b)+(c) off: refinements stay as found, so the list keeps two-condition subgroups the pruned one absorbs
    assert on.no_prune is not None
    depth = [len(f.description.conditions) for f in on.no_prune]
    assert max(depth) == 2 and [f.rank for f in on.no_prune] == list(range(1, len(depth) + 1))
    assert sum(d == 2 for d in depth) > sum(len(f.description.conditions) == 2 for f in on.findings)
    np_off = run_emm(df, prep, method.emm.model_copy(update={"prune_refinements": False}), SearchSpace())
    assert [str(f.description) for f in np_off.findings] == [str(f.description) for f in on.no_prune]
    assert np_off.no_prune is not None and [str(f.description) for f in np_off.no_prune] == \
        [str(f.description) for f in np_off.findings]


@crit("E12")
def test_a_new_quality_function_is_one_entry_in_the_registry(tmp_path, monkeypatch):
    # the example of the user manual: score = the largest |Δb| over the paths, no size weight
    new = emm_mod.QualityFunction(lambda pq, ev: max(abs(p.diff) for p in ev.paths), "the largest |Δb|")
    monkeypatch.setitem(emm_mod.QUALITY_FUNCTIONS, "max_dbeta", new)
    df, prep = _small()
    mask = (df["gender"] == "f").to_numpy()
    # the spec accepts it, the search scores with it, and a path deviates by the |z| rule (no intervals)
    cfg = EMMConfig(quality="max_dbeta")
    ev = PathQuality(prep, cfg, SearchSpace(min_n=10, min_n_complement=10), per_group=False).evaluate(mask)
    assert ev.valid and ev.quality == pytest.approx(max(abs(p.diff) for p in ev.paths), abs=1e-12)
    assert ev.paths[0].deviates("max_dbeta") == (abs(ev.paths[0].z) >= 1.959963984540054)
    res = run_emm(df, prep, cfg, SearchSpace(min_n=10, min_n_complement=10, depth=1))
    assert res.findings and res.findings[0].quality == pytest.approx(ev.quality, abs=1e-12)
    # an unknown name is refused with the list of known ones
    with pytest.raises(ValueError, match="unknown quality function 'nope'"):
        EMMConfig(quality="nope")
    # the app's menu comes from the same registry
    monkeypatch.setenv("EMMLAB_DATA", str(tmp_path))
    from fastapi.testclient import TestClient

    import emmlab.server.app as app_mod
    listed = {q["name"]: q for q in TestClient(app_mod.app).get("/api/qualities").json()}
    assert listed["max_dbeta"] == {"name": "max_dbeta", "description": "the largest |Δb|", "uses_intervals": False,
                                   "in_menu": True}
    assert [k for k, q in listed.items() if q["in_menu"] and k != "max_dbeta"] == ["current", "ent_z", "cook"]
    assert listed["current"]["uses_intervals"] and not listed["wald_z"]["in_menu"]


@crit("GR5")
def test_scale_artifact_class_and_oracle_truth():
    # no twist; group b scores higher, so more of it sits at the exam's maximum and its observed slope flattens
    w = {**BASE, "attributes": {"g": {"values": {"a": .5, "b": .5}}},
         "shifts": [{"when": {"g": "b"}, "target": "achievement", "delta": 1.0}]}
    w["nodes"] = {**BASE["nodes"], "achievement": {"recipe": {"noise_sd": 1}, "scale": {"type": "censored",
                                                                                      "high": 0.5}}}
    an = AnalysisConfig(normalisation="none")
    world = World.model_validate(w)
    tp = TruthProvider(compile_world(world), ResearcherModel.from_world(world), an, None, GradingConfig())
    d = Description((Condition("g", "==", "b"),))
    assert tp.engine is None and tp.contrast(d, "complement")["source"] == "oracle"
    assert tp.contrast(d, "complement")["common"]["feedback -> achievement"] < -0.05
    latent = tp.true_score_contrast(d, "complement")
    assert latent is not None and abs(latent["feedback -> achievement"]) < 1e-9
    assert _classify_world(w, d, an, diff=-0.1) == "scale_artifact"
    # without the scale the same group is a plain chance finding
    plain = {**w, "nodes": BASE["nodes"]}
    assert _classify_world(plain, d, an, diff=-0.1) == "null"

