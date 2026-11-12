"""M6 acceptance: later-scope features (implementation_plan §3.7)."""

from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd
import pytest

from emmlab import truth
from emmlab.emm import Condition, Description, PathQuality
from emmlab.estimand import EstimandEngine
from emmlab.experiment import aggregate, conditions, execute, run_adaptive
from emmlab.generate import generate, seed_sequence
from emmlab.grade import TruthProvider, classify, planted_twists
from emmlab.loader import RealDataSpec, load_real
from emmlab.parallel import pmap
from emmlab.run import execute_real
from emmlab.spec import (
    AnalysisConfig,
    EMMConfig,
    ExperimentSpec,
    Factor,
    GradingConfig,
    MethodPreset,
    ResearcherModel,
    RunSpec,
    SearchSpace,
    World,
    load_world,
    preset,
    to_plain,
)
from emmlab.stages import ols_fit, prepare, rubin
from emmlab.store import dumps_json
from emmlab.validation import benjamini_hochberg, holdout, permutation_null
from emmlab.world import compile_world
from tests.test_m3_emm import _fake
from tests.worlds import PRESETS, world_a

crit = pytest.mark.criterion


# --------------------------------------------------------------------------- L1 Cook


@crit("L1")
def test_cook_restricted_to_paths():
    w = world_a()
    df = generate(compile_world(w), 3)
    m = ResearcherModel.from_world(w)
    prep = prepare(df, m, AnalysisConfig(normalisation="none"))
    mask = (df["ses"] == "rich").to_numpy()
    cfg = EMMConfig(quality="cook", baseline="population", paths=["feedback -> achievement"])
    ev = PathQuality(prep, cfg, SearchSpace(min_n=10, min_n_complement=10), per_group=False).evaluate(mask)
    X1 = np.column_stack([np.ones(prep.n), prep.X["achievement"]])
    inv = np.linalg.inv(X1.T @ X1)
    full = prep.full["achievement"]
    s2 = (1 - full.r2) * prep.y["achievement"].var()
    i = full.regs.index("feedback")
    bg, *_ = ols_fit(prep.X["achievement"][mask], prep.y["achievement"][mask], prep.school[mask])
    d = bg[i] - full.coef[i]
    assert ev.quality == pytest.approx(d * d / (inv[1 + i, 1 + i] * s2), rel=1e-10)


# --------------------------------------------------------------------------- L2 factorial / adaptive


@crit("L2")
def test_full_factorial_and_adaptive_stopping():
    w = world_a()
    spec = ExperimentSpec(name="L2", base=RunSpec(world=w, model=ResearcherModel.from_world(w),
                                                  method=preset("recommended")),
                          factors=[Factor(path="search.depth", levels=[1, 2]),
                                   Factor(path="world.twists.0.delta", levels=[-0.4, -0.2])],
                          n_sim=20, target_mcse=0.06, max_sim=200)
    assert len(conditions(spec)) == 4
    res = run_adaptive(spec, workers=4, batch=20)
    agg = aggregate(spec, res)
    worst = max(v["mcse"] for c in agg["conditions"] for p in c["presets"].values()
                for v in p["headline"].values() if v["mcse"] is not None)
    n_used = agg["conditions"][0]["presets"]["recommended"]["n_planted"]
    assert worst <= 0.06 or n_used >= 200
    again = run_adaptive(spec, workers=4, batch=20)
    assert dumps_json(aggregate(spec, again)) == dumps_json(agg)  # reproducible from (master_seed, rep)


# --------------------------------------------------------------------------- L3 measurement layer


def _meas_world(rel_x=0.8, rel_y=0.9, rule=None) -> World:
    d = {"name": "meas", "schools": {"n_schools": 400, "students_per_school": {"n": 50}},
         "attributes": {"gender": {"values": {"m": 0.5, "f": 0.5}}},
         "nodes": {"x": {"root": {"mean": 0, "sd": 1}, "reliability": rel_x},
                   "y": {"recipe": {"noise_sd": 0.8}, "reliability": rel_y}},
         "arrows": [{"x -> y": 0.6}]}
    if rule:
        d["nodes"]["x"]["reliability_when"] = [rule]
    return World.model_validate(d)


@crit("L3")
def test_attenuation_analytic_and_simulated():
    w = _meas_world()
    cw = compile_world(w)
    m = ResearcherModel.from_world(w)
    an = AnalysisConfig(normalisation="none")
    obs = EstimandEngine(cw, m, an).full["y"].coef[0]
    lat = EstimandEngine(cw, m, an, true_scores=True).full["y"].coef[0]
    assert obs == pytest.approx(lat * math.sqrt(0.8 * 0.9), abs=0.005)
    ests = [prepare(generate(cw, seed_sequence(5, k)), m, an).full["y"].coef[0] for k in range(30)]
    z = (np.mean(ests) - obs) / (np.std(ests, ddof=1) / math.sqrt(30))
    assert abs(z) < 4


@crit("L3")
def test_group_specific_reliability_is_variance_artifact():
    w = _meas_world(rel_x=0.9, rule={"when": {"gender": "f"}, "reliability": 0.5})
    cw = compile_world(w)
    m = ResearcherModel.from_world(w)
    an = AnalysisConfig(normalisation="none")
    df = generate(cw, 1)
    tp = TruthProvider(cw, m, an, None, GradingConfig())
    desc = Description((Condition("gender", "==", "f"),))
    planted = planted_twists(w, m, tp, df, "complement")
    import networkx as nx
    cls = classify(_fake(desc, df, "x -> y", -0.2), planted, set(), nx.DiGraph(), tp, GradingConfig(), "complement")
    assert cls == "variance_artifact"


# --------------------------------------------------------------------------- L4 plausible values


def _pv_world(conditioning: list[str]) -> World:
    return World.model_validate({
        "name": "pv", "schools": {"n_schools": 200, "students_per_school": {"n": 35}},
        "attributes": {"gender": {"values": {"m": 0.5, "f": 0.5}}},
        "nodes": {"feedback": {"root": {"mean": 0, "sd": 1}},
                  "achievement": {"recipe": {"noise_sd": 0.9},
                                  "plausible_values": {"n": 10, "reliability": 0.8, "conditioning": conditioning}}},
        "arrows": [{"feedback -> achievement": 0.2}],
        "twists": [{"arrow": "feedback -> achievement", "when": {"gender": "f"}, "delta": 0.3}]})


@crit("L4")
def test_rubin_formula():
    coefs = np.array([[0.1, 0.5], [0.2, 0.4], [0.15, 0.6]])
    ses = np.array([[0.05, 0.1], [0.06, 0.1], [0.04, 0.1]])
    est, se = rubin(coefs, ses)
    np.testing.assert_allclose(est, coefs.mean(0))
    expect = np.sqrt((ses**2).mean(0) + (1 + 1 / 3) * coefs.var(0, ddof=1))
    np.testing.assert_allclose(se, expect, rtol=1e-12)


@crit("L4")
def test_pv_shortcuts_and_conditioning_attenuation():
    w = _pv_world(["gender", "feedback"])
    cw = compile_world(w)
    m = ResearcherModel.from_world(w)
    df = generate(cw, 2)
    pvs = df[[f"achievement_pv{i}" for i in range(1, 11)]]
    # averaging PVs underestimates the SD (an EAP-like score): compare with any single PV
    sd_mean = pvs.mean(axis=1).std()
    sd_single = pvs.std().mean()
    assert sd_mean < sd_single
    an = AnalysisConfig(normalisation="none", standardize="global_once")
    mask = (df["gender"] == "f").to_numpy()
    reps = []
    for k in range(40):
        d = generate(cw, seed_sequence(8, k))
        mk = (d["gender"] == "f").to_numpy()
        p = prepare(d, m, an)
        reps.append(p.fit(mk)["achievement"].coef[0] - p.fit(~mk)["achievement"].coef[0])
    eng_pv = EstimandEngine(cw, m, an)
    eng_true = EstimandEngine(cw, m, an, true_scores=True)
    f, b = eng_pv.mask({"gender": "f"}), eng_pv.mask({"gender": "m"})
    diff_pv = eng_pv.fit(f)["achievement"].coef[0] - eng_pv.fit(b)["achievement"].coef[0]
    diff_true = eng_true.fit(f)["achievement"].coef[0] - eng_true.fit(b)["achievement"].coef[0]
    assert abs(diff_pv) < abs(diff_true)  # interaction absent from conditioning -> pulled toward pooled
    mcse = np.std(reps, ddof=1) / math.sqrt(len(reps))
    assert (abs(diff_true) - np.mean(reps)) > 4 * mcse
    assert abs(np.mean(reps) - diff_pv) < 4 * mcse  # PV estimates centre on the PV estimand
    assert mask.sum() > 0


# --------------------------------------------------------------------------- L5 weights + BRR


def _weighted_world() -> World:
    return World.model_validate({
        "name": "weighted", "schools": {"n_schools": 160, "students_per_school": {"n": 30},
                                        "sampling": {"oversample": [{"when": {"stype": "voc"}, "factor": 3.0}]}},
        "attributes": {"stype": {"level": "school", "values": {"gen": 0.8, "voc": 0.2}}},
        "nodes": {"feedback": {"recipe": {"noise_sd": 1.0}}, "achievement": {"recipe": {"noise_sd": 1.0}}},
        "arrows": [{"feedback -> achievement": 0.3}],
        "shifts": [{"when": {"stype": "voc"}, "target": "feedback", "delta": 0.5}],
        "twists": [{"arrow": "feedback -> achievement", "when": {"stype": "voc"}, "delta": -0.4}]})


@crit("L5")
def test_replicate_factors_and_brr_formula():
    import statsmodels.api as sm
    w = _weighted_world()
    df = generate(compile_world(w), 4)
    rcols = [f"W_FSTURWT{r}" for r in range(1, 81)]
    ratios = set(np.round((df[rcols].to_numpy() / df[["W_FSTUWT"]].to_numpy()).ravel(), 6))
    assert ratios <= {0.5, 1.0, 1.5} and {0.5, 1.5} <= ratios
    m = ResearcherModel.from_world(w)
    an = AnalysisConfig(normalisation="none", weights="final", se="brr")
    prep = prepare(df, m, an)
    f = prep.full["achievement"]
    # direct implementation with statsmodels WLS on the same standardized arrays
    X = sm.add_constant(prep.X["achievement"])
    y = prep.y["achievement"]
    theta = sm.WLS(y, X, weights=df["W_FSTUWT"]).fit().params[1:]
    draws = np.array([sm.WLS(y, X, weights=df[c]).fit().params[1:] for c in rcols])
    se_direct = np.sqrt(((draws - theta) ** 2).sum(0) / 20)
    np.testing.assert_allclose(f.coef, theta, rtol=1e-8)
    np.testing.assert_allclose(f.se, se_direct, rtol=1e-8)


@crit("L5")
def test_weighted_vs_unweighted_lesson():
    w = _weighted_world()
    cw = compile_world(w)
    m = ResearcherModel.from_world(w)
    t3_pop = EstimandEngine(cw, m, AnalysisConfig(normalisation="none", weights="final")).full["achievement"].coef[0]
    t3_smp = EstimandEngine(cw, m, AnalysisConfig(normalisation="none")).full["achievement"].coef[0]
    est_w, est_u = [], []
    for k in range(60):
        d = generate(cw, seed_sequence(6, k))
        est_w.append(prepare(d, m, AnalysisConfig(normalisation="none", weights="final")).full["achievement"].coef[0])
        est_u.append(prepare(d, m, AnalysisConfig(normalisation="none")).full["achievement"].coef[0])
    mc_w = np.std(est_w, ddof=1) / math.sqrt(60)
    mc_u = np.std(est_u, ddof=1) / math.sqrt(60)
    assert abs(np.mean(est_w) - t3_pop) < 4 * mc_w  # weighted recovers the population estimand
    assert abs(np.mean(est_u) - t3_smp) < 4 * mc_u  # unweighted recovers the sample-composition one
    assert abs(np.mean(est_u) - t3_pop) > 4 * mc_u  # ...which is biased for the population


# --------------------------------------------------------------------------- L6 permutation null


def _null_rep(k: int) -> float:
    w = World.model_validate({
        "name": "nullperm", "schools": {"n_schools": 60, "students_per_school": {"n": 20}},
        "attributes": {"g": {"values": {"a": 0.5, "b": 0.5}}, "h": {"values": {"a": 0.7, "b": 0.3}},
                       "s": {"level": "school", "values": {"a": 0.5, "b": 0.5}}},
        "nodes": {"x": {"root": {"mean": 0, "sd": 1}}, "y": {"recipe": {"noise_sd": 1.0}}},
        "arrows": [{"x -> y": 0.3}]})
    cw = compile_world(w)
    m = ResearcherModel.from_world(w)
    df = generate(cw, seed_sequence(21, k))
    method = preset("recommended")
    method.analysis.normalisation = "none"
    ss = SearchSpace(depth=1, min_n=50, min_n_complement=50)
    prep = prepare(df, m, method.analysis)
    pn = permutation_null(df, prep, method.emm, ss, 99, seed_sequence(22, k))
    return float(pn.observed_top > pn.threshold)


@crit("L6")
@pytest.mark.slow
def test_permutation_null_calibrated():
    exceed = pmap(_null_rep, range(200))
    rate = float(np.mean(exceed))
    mcse = math.sqrt(0.05 * 0.95 / 200)
    assert abs(rate - 0.05) <= 2 * mcse + 1 / 100, rate  # + quantile granularity of 99 permutations


# --------------------------------------------------------------------------- L7 holdout


def _holdout_rep(k: int) -> float:
    w = load_world(PRESETS / "pisa_like.yaml").without_twists()
    cw = compile_world(w)
    m = ResearcherModel.from_world(w)
    df = generate(cw, seed_sequence(31, k))
    method = preset("recommended")
    r = holdout(df, m, method.analysis, method.emm, SearchSpace(depth=1), ["school_type"], seed_sequence(32, k))
    return float(any(f["confirmed"] for f in r["findings"]))


@crit("L7")
def test_benjamini_hochberg():
    # thresholds q*i/n = .0125, .025, .0375, .05
    assert benjamini_hochberg(np.array([0.01, 0.04, 0.03, 0.2]), 0.05).tolist() == [True, False, False, False]
    assert benjamini_hochberg(np.array([0.01, 0.03, 0.02, 0.2]), 0.05).tolist() == [True, True, True, False]
    # step-up: a later p below its threshold rescues earlier ones
    assert benjamini_hochberg(np.array([0.04, 0.045, 0.03, 0.049]), 0.05).tolist() == [True, True, True, True]
    assert benjamini_hochberg(np.array([0.02, 0.9]), 0.05).tolist() == [True, False]


@crit("L7")
@pytest.mark.slow
def test_holdout_fdr_on_null_worlds():
    hits = pmap(_holdout_rep, range(200))
    rate = float(np.mean(hits))
    assert rate <= 0.05 + 2 * math.sqrt(0.05 * 0.95 / 200), rate


# --------------------------------------------------------------------------- L8 real-data loader


def _pisa_fixture(tmp_path) -> str:
    w = to_plain(load_world(PRESETS / "pisa_like.yaml"))
    w["nodes"]["achievement"]["plausible_values"] = {"n": 10, "reliability": 0.9,
                                                     "conditioning": ["gender", "ses_q", "school_type"]}
    w["schools"]["sampling"] = {"oversample": [{"when": {"school_type": "vocational"}, "factor": 2.0}]}
    df = generate(compile_world(World.model_validate(w)), 77)
    out = pd.DataFrame({"CNT": "XYZ", "CNTSCHID": 9_000_000 + df["school_id"],
                        "ST004D01T": df["gender"].map({"f": 1, "m": 2}).astype(int),
                        "SC013Q01TA": df["school_type"].astype(str), "ESCS_Q": df["ses_q"].astype(str),
                        "PERFEED": df["feedback"], "SCIEEFF": df["selfeff"], "JOYSCIE": df["enjoyment"],
                        "INTBRSCI": df["interest"]})
    for i in range(1, 11):
        out[f"PV{i}SCIE"] = 500 + 100 * df[f"achievement_pv{i}"]
    out["W_FSTUWT"] = df["W_FSTUWT"]
    for r in range(1, 81):
        out[f"W_FSTURWT{r}"] = df[f"W_FSTURWT{r}"]
    out.loc[3, "PERFEED"] = np.nan  # one incomplete row
    p = tmp_path / "pisa_fixture.csv"
    out.to_csv(p, index=False)
    return str(p)


PISA_MAP = dict(latents={"feedback": "PERFEED", "selfeff": "SCIEEFF", "enjoyment": "JOYSCIE",
                         "interest": "INTBRSCI"},
                plausible_values={"achievement": "PV{m}SCIE"},
                attributes={"gender": "ST004D01T", "school_type": "SC013Q01TA", "ses_q": "ESCS_Q"},
                recode={"gender": {"1": "f", "2": "m"}})


@crit("L8")
def test_real_data_loader_and_pipeline(tmp_path):
    spec = RealDataSpec(path=_pisa_fixture(tmp_path), filter={"CNT": "XYZ"}, **PISA_MAP)
    df, frame = load_real(spec)
    assert frame["rows_after_listwise"] == frame["rows_after_filter"] - 1
    assert {"achievement_pv1", "achievement_pv10", "W_FSTUWT", "W_FSTURWT80"} <= set(df.columns)
    assert set(df["gender"].astype(str)) == {"f", "m"}
    model = ResearcherModel(latents=["feedback", "selfeff", "enjoyment", "interest", "achievement"],
                            arrows=["feedback -> selfeff", "feedback -> enjoyment", "selfeff -> enjoyment",
                                    "feedback -> interest", "enjoyment -> interest", "feedback -> achievement",
                                    "selfeff -> achievement", "enjoyment -> achievement",
                                    "interest -> achievement"])
    method = preset("recommended")
    method.analysis.weights = "final"
    method.analysis.se = "brr"
    r = execute_real(df, frame, model, method, SearchSpace(depth=1))
    assert r["origin"] == "real" and r["pv_pooling"] == "rubin" and r["se_method"] == "brr"
    assert r["emm"]["findings"] and all(f["tier"] in ("ok", "fragile", "unreliable") for f in r["flags"])
    ach = r["fits"]["achievement"]
    assert all(v > 0 for v in ach["se"].values())


@crit("L8")
@pytest.mark.skipif(not os.environ.get("EMMLAB_PISA_FILE"), reason="set EMMLAB_PISA_FILE to a PISA student file")
def test_real_pisa_file():  # pragma: no cover - needs the user's data
    spec = RealDataSpec(path=os.environ["EMMLAB_PISA_FILE"], filter={"CNT": os.environ.get("EMMLAB_PISA_CNT", "GRC")},
                        **PISA_MAP)
    df, frame = load_real(spec)
    assert frame["rows_after_listwise"] > 1000


__all__ = ["truth", "MethodPreset", "execute"]
