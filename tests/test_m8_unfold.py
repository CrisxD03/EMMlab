"""M8 acceptance: generating equations unfolded to primitives (implementation_plan §3.7b, EQ1-EQ6)."""

from __future__ import annotations

import numpy as np
import pytest

from emmlab.spec import World, load_world
from emmlab.truth import cell_moments
from emmlab.unfold import evaluate_cell, unfold_world
from emmlab.world import compile_world
from tests.worlds import PRESETS, world_a, world_b, world_pcase

crit = pytest.mark.criterion

PRESET_FILES = sorted(PRESETS.glob("*.yaml"))


def _poly(eq_part: dict) -> dict[frozenset, float]:
    return {frozenset(map(tuple, t["when"])): t["coef"] for t in eq_part["poly"]}


def _noise(eq: dict, symbol: str) -> dict[frozenset, float]:
    return _poly(next(n for n in eq["noise"] if n["symbol"] == symbol))


def _const(eq: dict) -> dict[frozenset, float]:
    return {frozenset(map(tuple, t["when"])): t["coef"] for t in eq["const"]}


def _internally_standardized() -> World:
    d = world_b().model_dump(by_alias=True)
    d["name"] = "world_b_std"
    d["generation"] = "internally_standardized"
    return World.model_validate(d)


WORLDS = [pytest.param(load_world(p), id=p.stem) for p in PRESET_FILES] + [
    pytest.param(world_a(), id="world_a"), pytest.param(world_b(), id="world_b"),
    pytest.param(world_pcase(), id="pcase"), pytest.param(_internally_standardized(), id="internally_standardized"),
]


@crit("EQ1")
@pytest.mark.parametrize("world", WORLDS)
def test_unfolded_equations_reproduce_the_per_cell_moments_exactly(world):
    """Evaluated in every cell, the unfolded equation of every node has the mean and variance the truth engine computes."""
    cw = compile_world(world)
    res = unfold_world(cw)
    mu, sig, _ = cell_moments(cw)
    assert res["unavailable"] == {}
    for v in cw.visible:
        eq = res["equations"]["roots"][v]
        j = cw.index[v]
        for g, assign in enumerate(cw.assigns):
            mean, var = evaluate_cell(eq, assign)
            assert mean == pytest.approx(mu[g, j], abs=1e-9), (v, assign)
            assert var == pytest.approx(sig[g][j, j], abs=1e-9), (v, assign)


@crit("EQ2")
def test_world_a_unfolds_to_the_hand_derivation():
    """feedback = 0.6 R + eF; selfeff = 0.18 R + 0.3 eF + eS; achievement = 0.425 R + (0.275 - 0.4 R) eF + 0.25 eS + eA."""
    res = unfold_world(compile_world(world_a()))["equations"]["roots"]
    R = frozenset({("ses", "rich")})
    none = frozenset()
    assert _const(res["feedback"]) == pytest.approx({R: 0.6})
    assert _noise(res["feedback"], "e:feedback") == pytest.approx({none: 1.0})
    assert _const(res["selfeff"]) == pytest.approx({R: 0.18})
    assert _noise(res["selfeff"], "e:feedback") == pytest.approx({none: 0.3})
    a = res["achievement"]
    assert _const(a) == pytest.approx({R: 0.425})  # 0.5 + 0.25*0.18 + 0.2*0.6 - 0.4*0.6 (R*R = R)
    assert _noise(a, "e:feedback") == pytest.approx({none: 0.275, R: -0.4})  # poor 0.275, rich -0.125 = total effects
    assert _noise(a, "e:selfeff") == pytest.approx({none: 0.25})
    assert _noise(a, "e:achievement") == pytest.approx({none: 1.0})
    sds = {n["symbol"]: n["sd"] for n in a["noise"]}
    assert sds["e:feedback"] == pytest.approx(1.0) and sds["e:selfeff"] == pytest.approx(0.9)
    assert sds["e:achievement"] == pytest.approx(0.8)


@crit("EQ3")
@pytest.mark.parametrize("world", WORLDS)
def test_every_factor_appears_once_and_values_of_one_attribute_never_mix(world):
    cw = compile_world(world)
    res = unfold_world(cw)
    for form in ("roots", "direct"):
        for eq in res["equations"][form].values():
            parts = [eq["const"], *[p["poly"] for p in eq.get("vars", [])], *[n["poly"] for n in eq["noise"]]]
            for poly in parts:
                keys = [tuple(sorted(map(tuple, t["when"]))) for t in poly]
                assert len(keys) == len(set(keys)), "a monomial is listed twice"
                for when in keys:
                    attrs = [a for a, _ in when]
                    assert len(attrs) == len(set(attrs)), "two values of one attribute in one term"
                assert all(abs(t["coef"]) > 1e-12 for t in poly), "a zero term was kept"
            symbols = [n["symbol"] for n in eq["noise"]]
            assert len(symbols) == len(set(symbols)), "a noise source is listed twice"
    # a root-form equation contains no other node: only attributes and noise sources
    assert all("vars" not in eq or eq["vars"] == [] for eq in res["equations"]["roots"].values())


@crit("EQ4")
def test_gender_twist_example_from_the_review_meeting():
    """selfeff = -0.0622 f + (0.097 + 0.1 f) eFeedback + eSelfeff, with f^2 = f."""
    d = load_world(PRESETS / "katsantonis_s0.yaml").model_dump(by_alias=True)
    d["twists"].append({"arrow": "feedback -> selfeff", "when": {"gender": "f"}, "delta": 0.1})  # as in the meeting
    cw = compile_world(World.model_validate(d))
    s = unfold_world(cw)["equations"]["roots"]["selfeff"]
    f = frozenset({("gender", "f")})
    none = frozenset()
    assert _const(s) == pytest.approx({f: 0.097 * -0.3158 + 0.1 * -0.3158}, abs=1e-12)
    assert _noise(s, "e:feedback") == pytest.approx({none: 0.097, f: 0.1})
    # school_type and qcsc appear nowhere in this path's equation: no subgroup built from them can move it
    mentioned = {a for poly in (_const(s), *[_poly(n) for n in s["noise"]]) for m in poly for a, _ in m}
    assert mentioned == {"gender"}


@crit("EQ5")
def test_direct_form_is_the_compact_equation_of_the_spec():
    res = unfold_world(compile_world(world_a()))["equations"]["direct"]
    R = frozenset({("ses", "rich")})
    none = frozenset()
    a = res["achievement"]
    by_var = {v["name"]: _poly(v) for v in a["vars"]}
    assert by_var["selfeff"] == pytest.approx({none: 0.25})
    assert by_var["feedback"] == pytest.approx({none: 0.2, R: -0.4})  # baseline slope and the twist
    assert _const(a) == pytest.approx({R: 0.5})  # the level shift
    assert [n["symbol"] for n in a["noise"]] == ["e:achievement"]


@crit("EQ6")
def test_primitives_and_the_world_truth_endpoint():
    res = unfold_world(compile_world(load_world(PRESETS / "katsantonis_s0.yaml")))
    prim = res["primitives"]
    assert {a["name"] for a in prim["attributes"]} == {"gender", "school_type", "qcsc"}
    qcsc = next(a for a in prim["attributes"] if a["name"] == "qcsc")
    assert qcsc["depends_on"] == "school_type" and qcsc["level"] == "student"
    syms = {n["symbol"]: n for n in prim["noise"]}
    assert "e:feedback" in syms and syms["e:feedback"]["sd"] > 0
    assert syms["u:achievement"]["sd"] > 0  # the school random intercept behind its ICC is a primitive too
    from fastapi.testclient import TestClient

    from emmlab.server.app import app
    from emmlab.spec import to_plain
    r = TestClient(app).post("/api/world/truth", json={"world": to_plain(load_world(PRESETS / "minimal.yaml"))}).json()
    assert r["ok"] and set(r["unfolded"]["equations"]) == {"roots", "direct"}
    assert np.isclose(
        next(t["coef"] for t in r["unfolded"]["equations"]["roots"]["feedback"]["const"] if t["when"]), 0.6)


@crit("EQ6")
def test_run_result_carries_the_generating_equations_of_the_world_that_made_the_data():
    from emmlab.run import execute_run
    from emmlab.spec import ResearcherModel, RunSpec, preset
    w = world_a()
    spec = RunSpec(world=w, model=ResearcherModel.from_world(w), method=preset("recommended"), seed=1)
    r = execute_run(spec, store=None, with_bootstrap=False)
    gen = r["generating"]
    assert set(gen["equations"]["roots"]) == {"feedback", "selfeff", "achievement"}
    assert gen["equations"] == unfold_world(compile_world(w))["equations"]
    assert execute_run(spec, store=None, with_bootstrap=False, light=True)["generating"] is None


@crit("EQ7")
def test_fitted_equations_unfold_to_the_model_implied_total_effects():
    """Unfolding a fitted model gives, on each starting latent and residual, the model-implied total effect."""
    from emmlab.unfold import unfold_fitted
    latents = ["feedback", "sport", "motivation", "extra", "grade"]
    coefs = {"motivation": {"feedback": 0.25}, "extra": {"sport": 0.297, "motivation": 0.197},
             "grade": {"feedback": 0.106, "motivation": 0.303, "extra": 0.102}}
    eqs = unfold_fitted(latents, coefs)
    idx = {v: i for i, v in enumerate(latents)}
    B = np.zeros((5, 5))
    for c, regs in coefs.items():
        for r, b in regs.items():
            B[idx[c], idx[r]] = b
    T = np.linalg.inv(np.eye(5) - B)  # total effects incl. the identity (own residual)
    g = eqs["grade"]
    by_var = {v["name"]: v["poly"][0]["coef"] for v in g["vars"]}
    by_noise = {n["node"]: n["poly"][0]["coef"] for n in g["noise"]}
    assert set(by_var) == {"feedback", "sport"}  # the model's starting latents stay as variables
    for root in ("feedback", "sport"):
        assert by_var[root] == pytest.approx(T[idx["grade"], idx[root]], abs=1e-12)
    for m in ("motivation", "extra", "grade"):
        assert by_noise[m] == pytest.approx(T[idx["grade"], idx[m]], abs=1e-12)
    assert eqs["feedback"]["vars"] == [{"name": "feedback", "poly": [{"coef": 1.0, "when": []}]}]
    assert all(t["when"] == [] for e in eqs.values() for p in e["vars"] + e["noise"] for t in p["poly"])


@crit("EQ7")
def test_run_carries_fitted_unfolded_equations_for_the_full_sample_and_each_finding():
    from emmlab.run import execute_run
    from emmlab.spec import ResearcherModel, RunSpec, preset
    w = world_a()
    r = execute_run(RunSpec(world=w, model=ResearcherModel.from_world(w), method=preset("recommended"), seed=1),
                    store=None, with_bootstrap=False)
    fu = r["fitted_unfolded"]
    assert set(fu["full"]) == {"feedback", "selfeff", "achievement"}
    f1 = r["emm"]["findings"][0]
    sub = fu["findings"][str(f1["rank"])]
    b = {p["path"]: p["b_g"] for p in f1["paths"]}
    coef = {v["name"]: v["poly"][0]["coef"] for v in sub["subgroup"]["achievement"]["vars"]}
    expected = b["feedback -> achievement"] + b["feedback -> selfeff"] * b["selfeff -> achievement"]
    assert coef["feedback"] == pytest.approx(expected, abs=1e-12)
