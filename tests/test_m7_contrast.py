"""M7 acceptance: contrastive analysis of a found conjunction (implementation_plan §3.7a, CA1-CA6)."""

from __future__ import annotations

import importlib
import math

import numpy as np
import pytest
from scipy import stats

from emmlab.contrast import contrast_finding
from emmlab.emm import Condition, Description, PathQuality, PathStat
from emmlab.generate import generate
from emmlab.spec import EMMConfig, ResearcherModel, RunSpec, SearchSpace, World, preset, to_plain
from emmlab.stages import ols_fit, prepare
from emmlab.world import compile_world

crit = pytest.mark.criterion

PATH = "feedback -> achievement"


def _world(twists: list[dict], name: str = "ca") -> World:
    return World.model_validate({
        "name": name,
        "schools": {"n_schools": 160, "students_per_school": {"dist": "fixed", "n": 40}},
        "attributes": {"a": {"level": "student", "values": {"x": 0.5, "y": 0.5}},
                       "b": {"level": "student", "values": {"u": 0.5, "v": 0.5}}},
        "nodes": {"feedback": {"root": {"mean": 0, "sd": 1}}, "achievement": {"recipe": {"noise_sd": 1.0}}},
        "arrows": [{"feedback -> achievement": 0.1}],
        "twists": twists,
    })


def _setup(twists: list[dict], quality: str = "ent_z", seed: int = 4):
    w = _world(twists)
    df = generate(compile_world(w), seed)
    model = ResearcherModel.from_world(w)
    method = preset("recommended")
    emm = EMMConfig(quality=quality)
    prep = prepare(df, model, method.analysis)
    frame = df.loc[prep.frame].reset_index(drop=True)
    pq = PathQuality(prep, emm, SearchSpace(min_n=50, min_n_complement=50), rng_seed=seed)
    desc = Description((Condition("a", "==", "y"), Condition("b", "==", "u")))
    return df, frame, prep, pq, desc


def _by_key(res: dict) -> dict:
    return {s["key"]: s for s in res["slices"]}


@crit("CA1")
def test_slices_partition_the_data_and_match_the_descriptions():
    _, frame, prep, pq, desc = _setup([])
    res = contrast_finding(frame, prep, desc, pq)
    s = _by_key(res)
    a, b = (frame["a"] == "y").to_numpy(), (frame["b"] == "u").to_numpy()
    expected = {"finding": a & b, "flip:1": a & ~b, "flip:0": ~a & b, "cell:00": ~a & ~b, "complement": ~(a & b)}
    assert set(s) == set(expected)
    for k, m in expected.items():
        assert s[k]["n"] == int(m.sum()), k
        assert s[k]["schools"] == len(np.unique(prep.school[m]))
    # the four cells of the 2 x 2 grid cover every student exactly once
    assert sum(s[k]["n"] for k in ("finding", "flip:1", "flip:0", "cell:00")) == len(frame)
    assert s["finding"]["label"] == "a=y AND b=u"
    assert s["flip:1"]["label"] == "a=y AND NOT b=u"
    assert s["cell:00"]["label"] == "NOT a=y AND NOT b=u"
    assert s["complement"]["label"] == "NOT (a=y AND b=u)"
    assert [x["condition"] for x in res["singles"]] == ["a=y", "b=u"]
    assert res["singles"][0]["alone"]["n"] == int(a.sum()) and res["singles"][0]["rest"]["n"] == int((~a).sum())


@crit("CA2")
@pytest.mark.parametrize(("twists", "kind", "needed"), [
    ([{"arrow": PATH, "when": {"a": "y"}, "delta": 0.6}], "carried_by", ["a=y"]),
    ([{"arrow": PATH, "when": {"b": "u"}, "delta": 0.6}], "carried_by", ["b=u"]),
    ([{"arrow": PATH, "when": {"a": "y"}, "delta": 0.5}, {"arrow": PATH, "when": {"b": "u"}, "delta": 0.5}],
     "conjunction_specific", ["a=y", "b=u"]),
])
def test_verdict_names_the_condition_that_carries_the_deviation(twists, kind, needed):
    _, frame, prep, pq, desc = _setup(twists)
    res = contrast_finding(frame, prep, desc, pq)
    assert PATH in res["focal_paths"]
    v = res["verdicts"][PATH]
    assert v["kind"] == kind
    assert v["needed"] == needed
    assert v["text"]


@crit("CA2")
@pytest.mark.parametrize(("conds", "deviates_when_flipped", "kind", "needed"), [
    (["a=y", "b=u"], [True, True], "conjunction_specific", ["a=y", "b=u"]),
    (["a=y", "b=u"], [True, False], "carried_by", ["a=y"]),
    (["a=y", "b=u"], [False, True], "carried_by", ["b=u"]),
    (["a=y", "b=u"], [False, False], "shared", []),
    (["a=y", "b=u", "c=p"], [True, False, True], "carried_by", ["a=y", "c=p"]),
    (["a=y", "b=u", "c=p"], [True, True, True], "conjunction_specific", ["a=y", "b=u", "c=p"]),
    (["a=y", "b=u", "c=p"], [False, False, False], "shared", []),
    (["a=y"], [], "single", []),
])
def test_verdict_rule(conds, deviates_when_flipped, kind, needed):
    """A condition is needed when flipping it makes the deviation go away; the verdict is read off that set."""
    from emmlab.contrast import verdict_for
    v = verdict_for(conds, deviates_when_flipped, low_power=False)
    assert (v["kind"], v["needed"]) == (kind, needed)
    assert v["text"] and v["low_power"] is False
    assert verdict_for(conds, deviates_when_flipped, low_power=True)["low_power"] is True


@crit("CA2")
@pytest.mark.parametrize(("kind", "needed", "alone", "expect_note"), [
    ("carried_by", ["a=y"], {"a=y": True, "b=u": False}, False),
    ("carried_by", ["a=y"], {"a=y": False, "b=u": False}, True),
    ("conjunction_specific", ["a=y", "b=u"], {"a=y": False, "b=u": True}, False),
    ("conjunction_specific", ["a=y", "b=u"], {"a=y": False, "b=u": False}, True),
    ("shared", [], {"a=y": False, "b=u": False}, False),
    ("single", [], {}, False),
])
def test_verdict_cautions_when_the_needed_conditions_do_not_deviate_alone(kind, needed, alone, expect_note):
    from emmlab.contrast import alone_note
    assert (alone_note(kind, needed, alone) is not None) is expect_note


@crit("CA2")
def test_planted_effect_is_supported_by_the_condition_alone():
    _, frame, prep, pq, desc = _setup([{"arrow": PATH, "when": {"a": "y"}, "delta": 0.6}])
    v = contrast_finding(frame, prep, desc, pq)["verdicts"][PATH]
    assert v["alone_deviates"]["a=y"] is True and v["note"] is None


@crit("CA2")
def test_null_world_has_no_deviating_path_to_explain():
    _, frame, prep, pq, desc = _setup([])
    res = contrast_finding(frame, prep, desc, pq)
    assert res["focal_paths"] == [] and res["verdicts"] == {}


@crit("CA3")
def test_deviation_rule_follows_the_quality_function():
    ps = PathStat("y", "x", b_g=0.0, se_g=1.0, b_b=3.5, se_b=1.0)
    ps.lo_g, ps.hi_g, ps.lo_b, ps.hi_b = -1.96, 1.96, 3.5 - 1.96, 3.5 + 1.96
    assert ps.disjoint is False and abs(ps.z) > 1.96
    assert ps.deviates("current") is False  # intervals overlap
    assert ps.deviates("current_exact") is False
    assert ps.deviates("ent_z") is True  # |z| = 2.47
    assert ps.deviates("wald_z") is True
    far = PathStat("y", "x", b_g=0.0, se_g=1.0, b_b=4.5, se_b=1.0)
    far.lo_g, far.hi_g, far.lo_b, far.hi_b = -1.96, 1.96, 4.5 - 1.96, 4.5 + 1.96
    assert far.deviates("current") is True

    twists = [{"arrow": PATH, "when": {"a": "y"}, "delta": 0.6}]
    for quality in ("current", "ent_z"):
        _, frame, prep, pq, desc = _setup(twists, quality=quality)
        res = contrast_finding(frame, prep, desc, pq)
        for sl in res["slices"]:
            for path, c in (sl["vs_finding"] or {}).items():
                assert c["deviating"] == (c["disjoint"] if quality == "current" else abs(c["z"]) >= 1.96), (sl["key"], path)


@crit("CA4")
def test_slice_numbers_equal_direct_regressions_and_intervals_are_wald():
    _, frame, prep, pq, desc = _setup([{"arrow": PATH, "when": {"a": "y"}, "delta": 0.6}])
    res = contrast_finding(frame, prep, desc, pq)
    a, b = (frame["a"] == "y").to_numpy(), (frame["b"] == "u").to_numpy()
    z = stats.norm.ppf(0.975)
    X, y, sch = prep.X["achievement"], prep.y["achievement"], prep.school
    for key, m in (("finding", a & b), ("flip:1", a & ~b), ("cell:00", ~a & ~b), ("complement", ~(a & b))):
        slope, _, se, _, _ = ols_fit(X[m], y[m], sch[m])  # standardized once, cluster-robust: `recommended`
        f = _by_key(res)[key]["fits"][PATH]
        assert f["b"] == pytest.approx(slope[0], abs=1e-10)
        assert f["se"] == pytest.approx(se[0], rel=1e-9)
        assert f["ci"] == pytest.approx([slope[0] - z * se[0], slope[0] + z * se[0]], abs=1e-10)
    cmp_ = _by_key(res)["flip:1"]["vs_finding"][PATH]
    f0, f1 = _by_key(res)["finding"]["fits"][PATH], _by_key(res)["flip:1"]["fits"][PATH]
    assert cmp_["diff"] == pytest.approx(f0["b"] - f1["b"], abs=1e-12)
    assert cmp_["z"] == pytest.approx((f0["b"] - f1["b"]) / math.hypot(f0["se"], f1["se"]), rel=1e-9)
    assert res["interval_method"] == "wald" and res["interval_level"] == 0.95


@crit("CA5")
def test_single_condition_finding_has_nothing_to_decompose():
    _, frame, prep, pq, _ = _setup([{"arrow": PATH, "when": {"a": "y"}, "delta": 0.6}])
    res = contrast_finding(frame, prep, Description((Condition("a", "==", "y"),)), pq)
    assert {s["key"] for s in res["slices"]} == {"finding", "complement"}
    assert res["singles"] == []
    assert res["verdicts"][PATH]["kind"] == "single"


@crit("CA6")
def test_current_preset_uses_its_own_intervals_and_slices_too_small_to_fit_are_flagged():
    w = _world([{"arrow": PATH, "when": {"a": "y"}, "delta": 0.6}])
    df = generate(compile_world(w), 4)
    method = preset("current")
    prep = prepare(df, ResearcherModel.from_world(w), method.analysis)
    frame = df.loc[prep.frame].reset_index(drop=True)
    pq = PathQuality(prep, method.emm, SearchSpace(min_n=50, min_n_complement=50))
    res = contrast_finding(frame, prep, Description((Condition("a", "==", "y"), Condition("b", "==", "u"))), pq)
    assert res["interval_se"] == "naive" and res["quality"] == "current"
    # a condition that matches 2 students cannot be fitted: reported as unreliable, not as a number
    tiny = frame.copy()
    tiny["b"] = tiny["b"].astype(str)
    tiny.loc[tiny.index[tiny["a"] == "y"][:2], "b"] = "w"
    pq2 = PathQuality(prep, method.emm, SearchSpace(min_n=1, min_n_complement=1))
    res2 = contrast_finding(tiny, prep, Description((Condition("a", "==", "y"), Condition("b", "==", "w"))), pq2)
    fin = _by_key(res2)["finding"]
    assert fin["n"] == 2
    assert fin["reliable"] is False and fin["fits"][PATH] is None


@crit("CA6")
def test_api_endpoint_returns_the_contrast_of_a_stored_run(tmp_path, monkeypatch):
    monkeypatch.setenv("EMMLAB_DATA", str(tmp_path))
    from fastapi.testclient import TestClient

    import emmlab.server.app as app_mod
    app_mod = importlib.reload(app_mod)
    client = TestClient(app_mod.app)
    w = _world([{"arrow": PATH, "when": {"a": "y"}, "delta": 0.6}])
    spec = RunSpec(world=w, model=ResearcherModel.from_world(w), method=preset("recommended"), seed=4,
                   search=SearchSpace(depth=2, min_n=100, min_n_complement=100))
    run = client.post("/api/run", json={"spec": to_plain(spec)})
    assert run.status_code == 200, run.text
    key = run.json()["run_key"]
    findings = run.json()["emm"]["findings"]
    assert findings
    r = client.get(f"/api/run/{key}/contrast", params={"rank": findings[0]["rank"]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["description"] == findings[0]["description"]
    assert body["slices"][0]["key"] == "finding" and body["paths"] == [PATH]
    assert client.get(f"/api/run/{key}/contrast", params={"rank": 99}).status_code == 404
    assert client.get("/api/run/doesnotexist/contrast", params={"rank": 1}).status_code == 404



@crit("CA7")
def test_every_pair_of_cells_is_compared_and_the_interaction_tells_additive_from_combined():
    add = [{"arrow": PATH, "when": {"a": "y"}, "delta": 0.4}, {"arrow": PATH, "when": {"b": "u"}, "delta": 0.4}]
    _, frame, prep, pq, desc = _setup(add)
    res = contrast_finding(frame, prep, desc, pq)
    pairs = res["pairs"]
    assert len(pairs) == 6  # 2 x 2 grid: two rows, two columns, two diagonals
    assert {frozenset((p["a"], p["b"])) for p in pairs} == {
        frozenset(x) for x in [("finding", "flip:1"), ("finding", "flip:0"), ("finding", "cell:00"),
                               ("flip:1", "flip:0"), ("flip:1", "cell:00"), ("flip:0", "cell:00")]}
    s = _by_key(res)
    f = {k: s[k]["fits"][PATH] for k in ("finding", "flip:1", "flip:0", "cell:00")}
    it = res["interaction"][PATH]
    expected = (f["finding"]["b"] - f["flip:1"]["b"]) - (f["flip:0"]["b"] - f["cell:00"]["b"])
    assert it["estimate"] == pytest.approx(expected, abs=1e-12)
    assert it["se"] == pytest.approx(math.sqrt(sum(x["se"] ** 2 for x in f.values())), rel=1e-12)
    assert abs(it["estimate"]) < 3 * it["se"] and it["additive"] is True  # twists that add up
    # a twist that exists only for the combination: a clear interaction
    _, frame, prep, pq, desc = _setup([{"arrow": PATH, "when": {"a": "y", "b": "u"}, "delta": 0.6}])
    it2 = contrast_finding(frame, prep, desc, pq)["interaction"][PATH]
    assert it2["estimate"] > 0.3 and it2["additive"] is False
    # one condition: nothing to pair beyond finding vs complement, no interaction
    _, frame, prep, pq, _ = _setup(add)
    one = contrast_finding(frame, prep, Description((Condition("a", "==", "y"),)), pq)
    assert one["pairs"] == [] and one["interaction"] == {}
