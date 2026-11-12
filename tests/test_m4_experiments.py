"""M4 acceptance: experiments, reports, reproducibility (implementation_plan §3.4 and G4)."""

from __future__ import annotations

import json
import math
import threading
import time

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from emmlab import truth
from emmlab.cli import app as cli
from emmlab.experiment import (
    aggregate,
    candidate_rows,
    conditions,
    execute,
    experiment_rows,
    run_spec_for,
    units,
)
from emmlab.generate import generate, seed_sequence
from emmlab.report import experiment_report, run_report
from emmlab.run import execute_run
from emmlab.spec import (
    AnalysisConfig,
    EMMConfig,
    ExperimentSpec,
    Factor,
    MethodPreset,
    ResearcherModel,
    RunSpec,
    SearchSpace,
    World,
    load_world,
    preset,
    to_plain,
)
from emmlab.store import Store, dumps_json
from emmlab.world import compile_world
from tests.worlds import PRESETS, world_a

crit = pytest.mark.criterion


def _exp(world: World, **kw) -> ExperimentSpec:
    base = RunSpec(world=world, model=ResearcherModel.from_world(world), method=kw.pop("method", preset("recommended")),
                   search=kw.pop("search", SearchSpace()))
    return ExperimentSpec(name=kw.pop("name", "t"), base=base, **kw)


@crit("X1")
def test_common_random_numbers():
    w = load_world(PRESETS / "pisa_like.yaml")
    spec = _exp(w, presets=["current", "recommended"],
                factors=[Factor(path="search.depth", levels=[1, 2])], n_sim=3)
    frames = []
    for cond in range(2):
        for p in ("current", "recommended"):
            rs = run_spec_for(spec, [{"search.depth": 1}, {"search.depth": 2}][cond], p, "planted")
            frames.append(generate(compile_world(rs.world), seed_sequence(rs.seed, 2)))
    for f in frames[1:]:
        pd.testing.assert_frame_equal(frames[0], f)
    # null variant shares the random stream: identical attributes and noise, only twisted slopes differ
    rs_null = run_spec_for(spec, {"search.depth": 1}, "recommended", "null")
    nf = generate(compile_world(rs_null.world), seed_sequence(rs_null.seed, 2))
    pd.testing.assert_series_equal(nf["gender"], frames[0]["gender"])
    pd.testing.assert_series_equal(nf["feedback"], frames[0]["feedback"])


@crit("X2")
def test_null_world_runs_automatically():
    w = world_a()
    spec = _exp(w, n_sim=12)
    assert {u.variant for u in units(spec)} == {"planted", "null"}
    agg = aggregate(spec, execute(spec, workers=1))
    p = agg["conditions"][0]["presets"]["recommended"]
    assert p["n_null"] == 12 and p["fwer_null"]["mcse"] is not None


def _power_world(n_schools: int, extra_decoys: int = 0) -> World:
    attrs = {"grp": {"values": {"a": 0.9, "b": 0.1}}}
    for i in range(extra_decoys):
        attrs[f"d{i}"] = {"role": "decoy", "values": {"x": 0.5, "y": 0.5}}
    return World.model_validate({
        "name": "power", "schools": {"n_schools": n_schools, "students_per_school": {"n": 25}},
        "attributes": attrs,
        "nodes": {"x": {"root": {"mean": 0, "sd": 1}}, "y": {"recipe": {"noise_sd": math.sqrt(0.91)}}},
        "arrows": [{"x -> y": 0.3}],
        "twists": [{"arrow": "x -> y", "when": {"grp": "b"}, "delta": 0.2, "label": "grp"}]})


def _power_spec(search_attrs: list[str] | None, decoys: int, n_sim: int, name: str) -> ExperimentSpec:
    method = MethodPreset(name="custom", analysis=AnalysisConfig(normalisation="none", se="naive"),
                          emm=EMMConfig(quality="wald_z", top_k=5))
    w = _power_world(80, decoys)
    levels = [20, 40, 60, 80, 120, 200]  # n_G = 0.1 * 25 * n_schools = 50 ... 500
    return _exp(w, name=name, method=method, presets=["recommended"], n_sim=n_sim,
                search=SearchSpace(attributes=search_attrs, depth=1 if search_attrs else 2,
                                   min_n=20, min_n_complement=20),
                factors=[Factor(path="world.schools.n_schools", levels=levels)])


def _analytic_n80() -> float:
    cw = compile_world(_power_world(80))
    _, sig = truth.pooled_moments(cw)
    var_y = sig[cw.index["y"], cw.index["y"]]
    # common metric: x and y standardized once, so the slope difference and residual variance scale by sd(y)
    return truth.required_n_g(p=0.1, delta=0.2 / math.sqrt(var_y), sigma2=0.91 / var_y)


@crit("X3")
@pytest.mark.slow
def test_power_calibration_prespecified():
    spec = _power_spec(["grp"], 0, 400, "X3")
    agg = aggregate(spec, execute(spec))
    n80 = agg["sweep"]["recommended:grp"]["significant"]["n80"]
    expect = _analytic_n80()
    assert abs(n80 - expect) / expect <= 0.20, (n80, expect)


@crit("X4")
@pytest.mark.slow
def test_search_penalty_visible():
    pre = aggregate(s := _power_spec(["grp"], 0, 300, "X4pre"), execute(s))
    full = aggregate(s2 := _power_spec(None, 6, 300, "X4full"), execute(s2))
    n80_pre = pre["sweep"]["recommended:grp"]["significant"]["n80"]
    n80_full = full["sweep"]["recommended:grp"]["significant"]["n80"]
    assert n80_full is None or n80_full > n80_pre
    # at the level closest to the pre-specified n80, the search hit rate is lower with
    # non-overlapping 95% Monte Carlo intervals
    pts_pre = pre["sweep"]["recommended:grp"]["significant"]["points"]
    pts_full = {p["n_g"]: p for p in full["sweep"]["recommended:grp"]["significant"]["points"]}
    near = min(pts_pre, key=lambda p: abs(p["n_g"] - n80_pre))
    other = min(pts_full.values(), key=lambda p: abs(p["n_g"] - near["n_g"]))
    assert other["hit_rate"] + 1.96 * other["mcse"] < near["hit_rate"] - 1.96 * near["mcse"], (near, other)


@crit("X5")
def test_flagship_experiment_runs_end_to_end(tmp_path):
    w = load_world(PRESETS / "pisa_like.yaml")
    spec = _exp(w, name="flagship", presets=["current", "recommended"], n_sim=10,
                factors=[Factor(path="world.twists.0.delta", levels=[0.1, 0.2])])
    agg = aggregate(spec, execute(spec, workers=4))
    html = experiment_report(agg)
    for p in ("current", "recommended"):
        d = agg["conditions"][1]["presets"][p]
        assert "feedback helps girls more" in d["twists"]
        assert d["fdp"]["all"]["value"] is not None and d["fwer_null"]["value"] is not None
    assert "null-World FWER" in html and "flagship" in html
    (tmp_path / "r.html").write_text(html)


@crit("X6")
def test_single_run_under_5s():
    w = load_world(PRESETS / "pisa_like.yaml")
    spec = RunSpec(world=w, model=ResearcherModel.from_world(w), method=preset("recommended"), seed=5)
    t0 = time.perf_counter()
    execute_run(spec)
    assert time.perf_counter() - t0 < 5.0


@crit("X6")
@pytest.mark.slow
def test_experiment_throughput_under_15min():
    w = load_world(PRESETS / "pisa_like.yaml")
    spec = _exp(w, name="X6", presets=["current", "recommended"], n_sim=200,
                factors=[Factor(path="world.twists.0.delta", levels=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3])])
    t0 = time.perf_counter()
    res = execute(spec)
    dt = time.perf_counter() - t0
    assert len(res) == 200 * 2 * 6 + 200 * 2  # the twist sweep shares one null World per preset
    assert dt < 15 * 60, dt


@crit("X7")
def test_cancel_and_resume_identical():
    w = world_a()
    spec = _exp(w, n_sim=10, presets=["current", "recommended"])
    full = execute(spec, workers=1)
    cancel = threading.Event()
    seen: list[str] = []

    def on_result(r):
        seen.append(r["unit"])
        if len(seen) >= 7:
            cancel.set()

    partial = execute(spec, workers=1, on_result=on_result, cancel=cancel)
    assert 7 <= len(partial) < len(full)
    resumed_new: list[str] = []
    resumed = execute(spec, workers=1, existing=partial, on_result=lambda r: resumed_new.append(r["unit"]))
    assert set(resumed_new).isdisjoint(partial)  # only the missing units were run
    assert dumps_json(aggregate(spec, resumed)) == dumps_json(aggregate(spec, full))


@crit("X8")
def test_reports_label_and_mcse():
    w = world_a()
    r = execute_run(RunSpec(world=w, model=ResearcherModel.from_world(w), method=preset("recommended"), seed=1))
    assert "one draw — not evidence" in run_report(r)
    spec = _exp(w, n_sim=10)
    agg = aggregate(spec, execute(spec, workers=1))
    html = experiment_report(agg)
    for c in agg["conditions"]:
        for d in c["presets"].values():
            for t in d["twists"].values():
                assert t["hit_rate"]["mcse"] is not None
    assert html.count("±") >= 6


@crit("G4")
def test_reproduce_run_and_experiment(tmp_path):
    w = world_a()
    spec = RunSpec(world=w, model=ResearcherModel.from_world(w), method=preset("recommended"), seed=4)
    s = Store(tmp_path / "artifacts")
    r = execute_run(spec, s)
    man = s.path(r["run_key"]) / "manifest.json"
    res = CliRunner().invoke(cli, ["reproduce", str(man)])
    assert res.exit_code == 0 and "identical" in res.output, res.output
    exp = _exp(w, name="g4", n_sim=6)
    d = tmp_path / "exp"
    d.mkdir()
    agg = aggregate(exp, execute(exp, workers=1))
    (d / "aggregate.json").write_text(dumps_json(agg))
    (d / "spec.json").write_text(dumps_json(to_plain(exp)))
    (d / "manifest.json").write_text("{}")
    res = CliRunner().invoke(cli, ["reproduce", str(d / "manifest.json")])
    assert res.exit_code == 0 and "identical" in res.output, res.output


@crit("G4")
def test_dataset_parquet_byte_identical(tmp_path):
    w = world_a()
    spec = RunSpec(world=w, model=ResearcherModel.from_world(w), method=preset("recommended"), seed=9)
    a, b = Store(tmp_path / "a"), Store(tmp_path / "b")
    ra, rb = execute_run(spec, a), execute_run(spec, b)
    pa = next((tmp_path / "a").rglob("dataset.parquet")).read_bytes()
    pb = next((tmp_path / "b").rglob("dataset.parquet")).read_bytes()
    assert pa == pb and ra["run_key"] == rb["run_key"]
    ja = next((tmp_path / "a").rglob("run.json")).read_text()
    jb = next((tmp_path / "b").rglob("run.json")).read_text()
    assert json.loads(ja) == json.loads(jb)


@crit("X3")
def test_logistic_n_recovers_known_curve():
    from emmlab.experiment import logistic_n
    a, b = -10.0, 2.0
    ns = np.array([50, 100, 200, 400, 800], float)
    p = 1 / (1 + np.exp(-(a + b * np.log(ns))))
    fit = logistic_n(ns, p, np.full(5, 10_000))
    assert fit["n50"] == pytest.approx(math.exp(5), rel=1e-3)
    assert fit["n80"] == pytest.approx(math.exp((math.log(4) + 10) / 2), rel=1e-3)


@crit("X9")
def test_null_runs_shared_across_twist_levels_and_both_lists_graded():
    w = world_a()
    spec = _exp(w, n_sim=3, presets=["recommended"],
                factors=[Factor(path="world.twists.0.delta", levels=[0.2, 0.4]),
                         Factor(path="world.schools.n_schools", levels=[60, 80])])
    us = units(spec)
    # 4 planted conditions, but only 2 distinct null Worlds (the twist size does not change a World without twists)
    assert sum(u.variant == "planted" for u in us) == 4 * 3
    assert sorted({u.cond for u in us if u.variant == "null"}) == [0, 1]
    res = execute(spec, workers=1)
    assert len(res) == len(us)
    # the shared null runs equal what each condition would have run itself
    alone = run_spec_for(spec, conditions(spec)[2], "recommended", "null")
    assert alone == run_spec_for(spec, conditions(spec)[0], "recommended", "null")
    agg = aggregate(spec, res)
    for c in agg["conditions"]:
        p = c["presets"]["recommended"]
        assert p["n_null"] == 3
        tw = next(iter(p["twists"].values()))
        assert tw["hit_rate_no_cover"]["n"] == 3 and "fdp_no_cover" in p
        assert tw["recovery_rate_no_prune"]["n"] == 3 and "fdp_no_prune" in p
        # recovered = a hit whose twisted path also deviates, so it can never exceed the hit rate
        assert tw["recovery_rate"]["value"] <= tw["hit_rate"]["value"]
    for r in res.values():
        if r["variant"] == "planted":
            assert "hits_no_cover" in r and "false_positives_no_cover" in r and "hits_no_prune" in r
            for h in r["hits"]:
                assert isinstance(h["path_deviates"], bool)
                assert h["recovered_rank"] is None or h["recovered_rank"] >= 1


@crit("X10")
def test_linked_factor_levels_and_flat_result_rows():
    w = world_a().model_copy(deep=True)
    w.twists.append(w.twists[0].model_copy(update={"label": "second"}))
    spec = _exp(w, n_sim=2, presets=["recommended"],
                factors=[Factor(path="world.twists.0.delta", linked=["world.twists.1.delta"], levels=[0.2, 0.4])])
    conds = conditions(spec)
    # linked paths take the same level instead of being crossed
    assert conds == [{"world.twists.0.delta": 0.2, "world.twists.1.delta": 0.2},
                     {"world.twists.0.delta": 0.4, "world.twists.1.delta": 0.4}]
    res = execute(spec, workers=1)
    runs, nulls = experiment_rows(spec, res)
    # one row per planted run x graded list x twist; one per null run, without the (shared) twist columns
    assert len(runs) == 2 * 2 * 3 * 2 and len(nulls) == 2
    assert {r["list"] for r in runs} == {"reported", "no_prune", "no_cover"}
    assert all(not k.startswith("world.twists") for r in nulls for k in r)
    for r in runs:
        assert r["recovered"] <= r["hit"] and (r["recovered_rank"] is None) == (not r["recovered"])
        assert r["path"] == f"{w.twists[0].parent} -> {w.twists[0].child}"
        if r["recovered"]:
            assert r["recovered_quality"] > 0 and r["jaccard"] >= spec.base.grading.tau


@crit("X11")
def test_model_factor_grades_a_missing_twisted_arrow_where_it_lands():
    w = world_a()
    true_model = ResearcherModel.from_world(w)
    arrow = f"{w.twists[0].parent} -> {w.twists[0].child}"
    wrong = true_model.model_copy(update={"name": "without_twisted_arrow",
                                          "arrows": [a for a in true_model.arrows if a != arrow]})
    spec = _exp(w, n_sim=1, presets=["recommended"],
                factors=[Factor(path="model", levels=[to_plain(true_model), to_plain(wrong)])])
    # each model gets its own null runs (the chance threshold depends on the model)
    assert sorted({u.cond for u in units(spec) if u.variant == "null"}) == [0, 1]
    runs, nulls = experiment_rows(spec, execute(spec, workers=1))
    assert len(nulls) == 2
    by_model = {r["model"]["name"]: r["path"] for r in runs if r["list"] == "reported"}
    assert by_model[true_model.name] == arrow
    # without the twisted arrow the twist is graded on the path into its child that its contrast moves most
    landed = by_model["without_twisted_arrow"]
    assert landed != arrow and landed.endswith(f"-> {w.twists[0].child}")


def _recovered_from(cands: list[dict], tau: float) -> dict[tuple, int | None]:
    """Recovered rank per (cond, rep, list, twist) recomputed from candidate rows at overlap bar ``tau``."""
    out: dict[tuple, int | None] = {}
    for c in cands:
        k = (c["cond"], c["rep"], c["list"], c["twist"])
        ok = c["jaccard"] >= tau and c["sign_ok"] and c["path_deviates"]
        prev = out.get(k)
        out[k] = (c["rank"] if prev is None else min(prev, c["rank"])) if ok else prev
    return out


@crit("X12")
def test_candidate_rows_reproduce_recovery_at_any_tau():
    w = world_a()
    spec = _exp(w, n_sim=3, presets=["recommended"], factors=[Factor(path="world.twists.0.delta", levels=[0.2, 0.4])])
    res = execute(spec, workers=1)
    runs, _ = experiment_rows(spec, res)
    cands = candidate_rows(spec, res)
    g = spec.base.grading
    assert cands and {c["list"] for c in cands} == {"reported", "no_prune", "no_cover"}
    assert all(c["jaccard"] >= min(g.tau, g.proxy_min_jaccard) for c in cands)
    # at the spec's tau the candidate rows give exactly the graded recovery and recovered rank
    rec = _recovered_from(cands, g.tau)
    for r in runs:
        assert rec.get((r["cond"], r["rep"], r["list"], r["twist"])) == r["recovered_rank"]
    # a lower bar can only recover more
    low = _recovered_from(cands, g.proxy_min_jaccard)
    assert all(low.get((r["cond"], r["rep"], r["list"], r["twist"])) is not None for r in runs if r["recovered"])

