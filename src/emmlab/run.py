"""One Run: World + seed -> Dataset -> stages -> EMM -> canonical findings -> grade (+ bootstrap).

Every stage is cached in the artifact store under hash(stage, inputs); the Run's manifest records
everything needed to regenerate any number (spec hashes, seed path, package versions, preset).
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from . import truth
from .bootstrap import school_bootstrap
from .emm import Description, run_emm
from .generate import generate, seed_sequence
from .grade import TruthProvider, estimation_block, grade
from .spec import MethodPreset, RunSpec, preset, spec_hash, to_plain
from .stages import prepare
from .store import Store, dumps_json, stage_key, versions
from .unfold import unfold_fitted, unfold_world
from .world import compile_world

TIER_UNRELIABLE, TIER_FRAGILE = 20, 50


def cluster_tier(g: float) -> str:
    if g < TIER_UNRELIABLE:
        return "unreliable"
    if g < TIER_FRAGILE:
        return "fragile"
    return "ok"


def effective_schools(codes: np.ndarray, mask: np.ndarray, n_groups: int, min_members: int = 3) -> dict:
    cnt = np.bincount(codes[mask], minlength=n_groups)
    contrib = cnt[cnt >= min_members].astype(float)
    g = int(len(contrib))
    bal = float(contrib.sum() ** 2 / (contrib**2).sum()) if g else 0.0
    return {"schools": int(np.count_nonzero(cnt)), "g_eff": g, "g_eff_balanced": bal,
            "tier": cluster_tier(min(g, bal))}


def preset_overrides(method: MethodPreset) -> dict:
    """Fields that differ from the named preset (recorded in the manifest)."""
    if method.name == "custom":
        return to_plain(method)
    base = to_plain(preset(method.name))
    cur = to_plain(method)
    out = {}
    for sec in ("analysis", "emm"):
        for k, v in cur[sec].items():
            if base[sec].get(k) != v:
                out[f"{sec}.{k}"] = v
    return out


def dataset_for(spec: RunSpec, store: Store | None, seed_key: tuple[int, ...]) -> tuple[pd.DataFrame, str]:
    cw = compile_world(spec.world)
    inputs = {"world": spec_hash(spec.world), "seed": [spec.seed, *seed_key]}

    def make() -> pd.DataFrame:
        return generate(cw, seed_sequence(spec.seed, *seed_key))

    if store is None:
        return make(), "mem"
    df, key = store.cached("dataset", inputs, make, kind="parquet")
    for a, at in spec.world.attributes.items():  # parquet round-trip keeps category order explicit
        df[a] = pd.Categorical(df[a].astype(str), categories=list(at.values), ordered=at.ordinal)
    return df, key


def execute_run(spec: RunSpec, store: Store | None = None, seed_key: tuple[int, ...] = (),
                *, threshold: float | None = None, with_bootstrap: bool = True,
                df: pd.DataFrame | None = None, light: bool = False) -> dict[str, Any]:
    """``light`` skips display-only extras (per-finding truth contrasts) inside Experiments."""
    world, model, method = spec.world, spec.model, spec.method
    filt = spec.search.filter
    d_inputs = {"world": spec_hash(world), "seed": [spec.seed, *seed_key]}
    dkey = stage_key("dataset", d_inputs) if df is None else "external"
    a_inputs = {"dataset": dkey, "model": spec_hash(model), "analysis": spec_hash(method.analysis),
                "filter": filt}
    e_inputs = {**a_inputs, "search": spec_hash(spec.search), "emm": spec_hash(method.emm), "seed": spec.seed}
    g_inputs = {**e_inputs, "world": spec_hash(world), "grading": spec_hash(spec.grading), "threshold": threshold}
    b_inputs = {**e_inputs, "bootstrap": spec_hash(spec.bootstrap) if spec.bootstrap else None,
                "with_bootstrap": with_bootstrap}
    keys = {"dataset": dkey, "fits": stage_key("fits", a_inputs), "emm": stage_key("emm", e_inputs),
            "grade": stage_key("grade", g_inputs), "bootstrap": stage_key("bootstrap", b_inputs)}
    manifest = {
        "world": spec_hash(world), "model": spec_hash(model), "method": spec_hash(method),
        "method_preset": method.name, "overrides": preset_overrides(method), "search": spec_hash(spec.search),
        "grading": spec_hash(spec.grading), "bootstrap": spec_hash(spec.bootstrap) if spec.bootstrap else None,
        "seed_path": [spec.seed, *seed_key], "stages": keys,
        "validation": {"permutation_reps": spec.permutation_reps, "holdout": spec.holdout},
        "label": "one draw — not evidence" if not seed_key else "replication",
    }
    r_inputs = {"manifest": manifest, "threshold": threshold}

    def compute() -> dict[str, Any]:
        r = _compute_run(spec, store, seed_key, threshold, with_bootstrap, df, manifest,
                         (d_inputs, a_inputs, e_inputs, g_inputs, b_inputs), light)
        return json.loads(dumps_json(r))  # plain JSON: identical whether computed or loaded from cache

    if store is None:
        result = compute()
    else:
        result, rkey = store.cached("run", r_inputs, compute)
        store.write_manifest(rkey, {**manifest, "versions": versions(), "run": rkey})
        result = {**result, "run_key": rkey}
    return result


def _compute_run(spec, store, seed_key, threshold, with_bootstrap, df, manifest, inputs,
                 light: bool = False) -> dict[str, Any]:
    d_inputs, a_inputs, e_inputs, g_inputs, b_inputs = inputs
    world, model, method = spec.world, spec.model, spec.method
    filt = spec.search.filter
    cw = compile_world(world)
    if df is None:
        df, _ = dataset_for(spec, store, seed_key)
    prep = prepare(df, model, method.analysis, filt)
    frame_df = df.loc[prep.frame].reset_index(drop=True)

    def cached(stage, inp, fn):
        return fn() if store is None else store.cached(stage, inp, fn)[0]

    fits = cached("fits", a_inputs, lambda: {c: f.as_dict() for c, f in prep.full.items()})
    holder: dict[str, Any] = {}

    def do_emm() -> dict:
        res = run_emm(df, prep, method.emm, spec.search, seed=spec.seed)
        holder["res"] = res
        return res.to_dict()

    emm_dict = cached("emm", e_inputs, do_emm)
    res = holder.get("res") or rebuild_emm(emm_dict, df, prep, method.emm, spec.search, spec.seed)
    tp = TruthProvider(cw, model, method.analysis, filt, spec.grading)
    z_crit = float(stats.norm.ppf(0.5 + method.emm.interval_level / 2))

    def graded(findings) -> dict:
        return grade(world, model, replace(res, findings=findings), frame_df, tp, spec.grading, method.emm.baseline,
                     threshold, quality=method.emm.quality, z_crit=z_crit).to_dict()

    gr = cached("grade", g_inputs, lambda: graded(res.findings))
    # the same search graded on the top_k without cover-based selection (step d) and without steps (b)+(c):
    # what each filter costs
    no_cover, no_prune = res.no_cover, res.no_prune
    gr_nc = None if no_cover is None else cached("grade_no_cover", g_inputs, lambda: graded(no_cover))
    gr_np = None if no_prune is None else cached("grade_no_prune", g_inputs, lambda: graded(no_prune))
    est = estimation_block(prep.full, tp, method.emm.interval_level)
    flags = []
    for f in res.findings:
        sub = effective_schools(prep.codes, f.mask, prep.n_groups)
        comp = effective_schools(prep.codes, ~f.mask, prep.n_groups)
        g = min(sub["g_eff"], sub["g_eff_balanced"]) if method.emm.baseline == "population" else \
            min(sub["g_eff"], sub["g_eff_balanced"], comp["g_eff"], comp["g_eff_balanced"])
        flag = {"description": str(f.description), "subgroup": sub, "complement": comp,
                "g_eff": g, "tier": cluster_tier(g)}
        if method.name == "current":
            flag["current_rule_fragile"] = sub["schools"] < method.emm.nominal_min_schools
        flags.append(flag)
    boot = None
    if with_bootstrap and spec.bootstrap is not None:
        # the bootstrap mirrors the design: stratify only when schools were allocated to strata
        school_attrs = ([a for a, at in world.attributes.items() if at.level == "school"]
                        if world.schools.allocation == "stratified" else [])
        descs = [f.description for f in res.findings[: spec.bootstrap.top_k]]
        boot = cached("bootstrap", b_inputs, lambda: school_bootstrap(
            df, model, method.analysis, filt, school_attrs, descs, method.emm.baseline, spec.bootstrap,
            seed_sequence(spec.seed, *seed_key, 7)).to_dict())
    # three truths per model path (system_design §3) and the truth contrast of each finding
    from .estimand import total_effect_truth
    t3_full = tp.full()
    truths = {}
    for c, regs in model.equations().items():
        for r in regs:
            key = f"{r} -> {c}"
            arrow = next((a for a in world.arrows if a.parent == r and a.child == c), None)
            t1 = float(truth.group_average(cw, cw.B[:, cw.index[c], cw.index[r]])) if r in cw.index else None
            truths[key] = {"t1_direct_avg": t1 if arrow is not None else 0.0,
                           "t2_total_avg": total_effect_truth(cw, r, c),
                           "t3": t3_full[c][r], "units_t1_t2": "raw World units",
                           "units_t3": "standardized once (common metric)" if method.analysis.standardize ==
                           "global_once" else "standardized within each group"}
    label = {"none": "pooled slope (within and between schools blended)",
             "centered": "within-school slope (comparing students in the same school)",
             "within_std": "pooled within-school correlation structure (not a structural slope)"}
    finding_truth = []
    for f in ([] if light else res.findings[:10]):
        try:
            finding_truth.append(tp.contrast(f.description, method.emm.baseline))
        except Exception as e:  # pragma: no cover
            finding_truth.append({"source": "error", "error": repr(e)})
    validation = _validation(spec, df, prep, res, seed_key)
    return {
        "validation": validation,
        "truths": truths, "estimand_label": label[method.analysis.normalisation],
        "generating": None if light else unfold_world(cw),  # the truth this data came from, in primitives
        "fitted_unfolded": None if light else _fitted_unfolded(model, prep, res),
        "finding_truth": finding_truth,
        "manifest": manifest, "n": int(prep.n), "n_schools": int(prep.n_groups),
        "fits": fits, "estimation": est, "emm": emm_dict, "grade": gr, "grade_no_cover": gr_nc,
        "grade_no_prune": gr_np, "flags": flags,
        "bootstrap": boot,
        "truth_source": "analytic" if tp.engine is not None else "oracle", "spec": to_plain(spec),
    }


def _fitted_unfolded(model, prep, res) -> dict[str, Any]:
    """The fitted equations unfolded to the model's own primitives: full sample, and subgroup / baseline per finding."""
    def by_side(paths: list, key: str) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for p in paths:
            out.setdefault(p.child, {})[p.reg] = float(getattr(p, key))
        return out

    full = {c: {r: float(b) for r, b in zip(f.regs, f.coef, strict=True)} for c, f in prep.full.items()}
    return {"full": unfold_fitted(model.latents, full),
            "findings": {str(f.rank): {"subgroup": unfold_fitted(model.latents, by_side(f.evaluation.paths, "b_g")),
                                       "complement": unfold_fitted(model.latents, by_side(f.evaluation.paths, "b_b"))}
                         for f in res.findings[:10]}}


def rebuild_emm(emm_dict: dict, df: pd.DataFrame, prep, emm_cfg, search, seed: int):
    """Rebuild Finding objects from a cached EMM result without re-running the search."""
    from .emm import Condition, EMMResult, Finding, PathQuality
    frame = df.loc[prep.frame].reset_index(drop=True)
    pq = PathQuality(prep, emm_cfg, search, rng_seed=seed)

    def build(items):
        out = []
        for d in items:
            desc = Description(tuple(Condition(**c) for c in d["conditions"]))
            m = desc.mask(frame)
            out.append(Finding(desc, d["quality"], pq.evaluate(m), m, d["rank"]))
        return out

    nc, np_ = emm_dict.get("no_cover"), emm_dict.get("no_prune")
    return EMMResult(build(emm_dict["findings"]), build(emm_dict["raw"]), emm_dict["evaluations"],
                     emm_dict["descriptors"], emm_dict.get("coverage", {}), None if nc is None else build(nc),
                     None if np_ is None else build(np_))


def run_spec_from(world, model=None, preset_name: str = "recommended", **kw) -> RunSpec:
    from .spec import ResearcherModel
    model = model or ResearcherModel.from_world(world)
    return RunSpec(world=world, model=model, method=preset(preset_name), **kw)  # type: ignore[arg-type]


def describe(d: Description) -> str:
    return str(d)


def finite(x: float) -> bool:
    return x is not None and math.isfinite(x)


__all__ = ["execute_run", "run_spec_from", "truth"]


def _validation(spec, df, prep, res, seed_key) -> dict | None:
    from .validation import holdout, permutation_null
    out: dict[str, Any] = {}
    if spec.permutation_reps:
        top = max((f.quality for f in res.raw), default=0.0)
        out["permutation"] = permutation_null(df, prep, spec.method.emm, spec.search, spec.permutation_reps,
                                              seed_sequence(spec.seed, *seed_key, 11), top).to_dict()
    if spec.holdout:
        strata = ([a for a, at in spec.world.attributes.items() if at.level == "school"]
                  if spec.world is not None else [])
        out["holdout"] = holdout(df, spec.model, spec.method.analysis, spec.method.emm, spec.search, strata,
                                 seed_sequence(spec.seed, *seed_key, 13))
    return out or None


def execute_real(df: pd.DataFrame, frame_info: dict, model, method, search, *, bootstrap=None, seed: int = 0,
                 permutation_reps: int = 0, holdout_split: bool = False) -> dict[str, Any]:
    """Run the identical analysis on a real Dataset (no World, so no truths and no Scorecard)."""
    from .validation import holdout, permutation_null
    prep = prepare(df, model, method.analysis, search.filter)
    res = run_emm(df, prep, method.emm, search, seed=seed)
    flags = []
    for f in res.findings:
        sub = effective_schools(prep.codes, f.mask, prep.n_groups)
        comp = effective_schools(prep.codes, ~f.mask, prep.n_groups)
        g = min(sub["g_eff"], sub["g_eff_balanced"], comp["g_eff"], comp["g_eff_balanced"])
        flags.append({"description": str(f.description), "subgroup": sub, "complement": comp, "g_eff": g,
                      "tier": cluster_tier(g)})
    out: dict[str, Any] = {
        "origin": "real", "frame": frame_info, "n": int(prep.n), "n_schools": int(prep.n_groups),
        "fits": {c: f.as_dict() for c, f in prep.full.items()}, "emm": res.to_dict(), "flags": flags,
        "se_method": method.analysis.se, "weights": method.analysis.weights,
        "pv_pooling": method.analysis.pv_mode if hasattr(prep, "parts") else None,
        "label": "real data — no answer key",
    }
    if bootstrap is not None:
        descs = [f.description for f in res.findings[: bootstrap.top_k]]
        out["bootstrap"] = school_bootstrap(df, model, method.analysis, search.filter, [], descs,
                                            method.emm.baseline, bootstrap, seed_sequence(seed, 7)).to_dict()
    if permutation_reps:
        top = max((f.quality for f in res.raw), default=0.0)
        out["permutation"] = permutation_null(df, prep, method.emm, search, permutation_reps,
                                              seed_sequence(seed, 11), top).to_dict()
    if holdout_split:
        out["holdout"] = holdout(df, model, method.analysis, method.emm, search, [], seed_sequence(seed, 13))
    return json.loads(dumps_json(out))
