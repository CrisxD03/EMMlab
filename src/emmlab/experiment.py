"""Experiments: replicated Runs over factor levels and method presets (ADEMP, research note R2).

- Common random numbers: replication k uses the stream (master_seed, k) in every condition, preset
  and World variant, so comparisons are paired.
- Every condition runs the matching null World (twists removed). Its top qualities give an
  out-of-sample calibrated threshold (calibrate on even reps, evaluate FWER on odd reps).
  Conditions that differ only in ``world.twists.*`` have the same null World, so they share one set of null runs.
- Each planted run is graded three times from one search: the reported top-k, the top-k by quality without
  cover-based selection (``*_no_cover``) and the top-k without steps (b)+(c) (``*_no_prune``), so the cost of
  each filter is a paired comparison.
- "Recovered" = a hit (overlap >= tau, right sign) whose twisted path also deviates under the run's own rule.
- Every reported rate carries its Monte Carlo standard error (MCSE).
"""

from __future__ import annotations

import contextlib
import copy
import itertools
import math
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import optimize

from .spec import ExperimentSpec, RunSpec, World, canonical_json, preset, spec_hash, to_plain


@dataclass(frozen=True)
class Unit:
    cond: int
    preset: str
    variant: str  # "planted" | "null"
    rep: int

    @property
    def key(self) -> str:
        return f"{self.cond}|{self.preset}|{self.variant}|{self.rep}"


def set_path(d: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur: Any = d
    for p in parts[:-1]:
        cur = cur[int(p)] if isinstance(cur, list) else cur[p]
    last = parts[-1]
    if isinstance(cur, list):
        cur[int(last)] = value
    else:
        cur[last] = value


def conditions(spec: ExperimentSpec) -> list[dict[str, Any]]:
    """Full factorial over factor levels (a single factor = a one-factor sweep)."""
    if not spec.factors:
        return [{}]
    out = []
    for combo in itertools.product(*[f.levels for f in spec.factors]):
        cond: dict[str, Any] = {}
        for f, level in zip(spec.factors, combo, strict=True):
            for path in (f.path, *f.linked):
                cond[path] = level
        out.append(cond)
    return out


def run_spec_for(spec: ExperimentSpec, cond: dict[str, Any], preset_name: str, variant: str) -> RunSpec:
    base = to_plain(spec.base)
    for path, v in cond.items():
        set_path(base, path, copy.deepcopy(v))
    rs = RunSpec.model_validate(base)
    method = preset(preset_name)  # type: ignore[arg-type]
    # keep any non-preset overrides from the base method (e.g. top_k, paths)
    if spec.base.method.name == "custom":
        method = rs.method
    else:
        for sec in ("analysis", "emm"):
            base_sec = to_plain(getattr(preset(spec.base.method.name), sec))
            cur_sec = to_plain(getattr(rs.method, sec))
            for k, v in cur_sec.items():
                if base_sec.get(k) != v:
                    setattr(getattr(method, sec), k, v)
    if variant == "null":
        if spec.null_world is not None:
            nw = to_plain(spec.null_world)
            for path, v in cond.items():
                if path.startswith("world."):
                    with contextlib.suppress(KeyError, IndexError, TypeError):
                        set_path({"world": nw}, path, copy.deepcopy(v))
            world = World.model_validate(nw)
        else:
            world = rs.world.without_twists()
    else:
        world = rs.world
    return rs.model_copy(update={"method": method, "world": world, "seed": spec.master_seed, "bootstrap": None})


def null_conditions(spec: ExperimentSpec) -> list[int]:
    """For each condition, the condition whose null runs it uses: the first one equal to it outside ``world.twists``.

    The null World drops every twist, so changing only twists cannot change it (and the seeds are shared).
    """
    conds = conditions(spec)

    def outside_twists(c: dict[str, Any]) -> str:
        return canonical_json({k: v for k, v in c.items() if not k.startswith("world.twists")})

    first: dict[str, int] = {}
    return [first.setdefault(outside_twists(c), i) for i, c in enumerate(conds)]


def units(spec: ExperimentSpec, n_sim: int | None = None, start: int = 0) -> list[Unit]:
    n = n_sim or spec.n_sim
    reps = range(start, n)
    out = [Unit(c, p, "planted", r) for c in range(len(conditions(spec))) for p in spec.presets for r in reps]
    if spec.include_null_world:
        out += [Unit(c, p, "null", r) for c in sorted(set(null_conditions(spec))) for p in spec.presets for r in reps]
    return out


def _run_unit_star(args: tuple[str, Unit]) -> dict[str, Any]:
    return run_unit(*args)


def run_unit(spec_json: str, unit: Unit) -> dict[str, Any]:
    """Execute one replication and return a compact summary (runs in a worker process)."""
    from .run import execute_run
    spec = ExperimentSpec.model_validate_json(spec_json)
    cond = conditions(spec)[unit.cond]
    rs = run_spec_for(spec, cond, unit.preset, unit.variant)
    r = execute_run(rs, None, (unit.rep,), with_bootstrap=False, light=True)
    findings = r["emm"]["findings"]
    decoys = _decoy_attrs(rs.world)
    return {
        "unit": unit.key, "cond": unit.cond, "preset": unit.preset, "variant": unit.variant, "rep": unit.rep,
        "n": r["n"], "n_schools": r["n_schools"], "dataset_seed": [rs.seed, unit.rep],
        "dataset_world": spec_hash(rs.world),
        "top_quality": max((f["quality"] for f in findings), default=0.0),
        "findings": [{"description": f["description"], "quality": f["quality"], "rank": f["rank"],
                      "attrs": sorted({c["attr"] for c in f["conditions"]}), "n": f["n"],
                      "uses_decoy": bool({c["attr"] for c in f["conditions"]} & decoys)} for f in findings],
        "hits": r["grade"]["hits"], "misses": r["grade"]["misses"], "redundant": r["grade"]["redundant"],
        "candidates": r["grade"].get("candidates", []),
        **_variant("no_cover", r.get("grade_no_cover"), r["emm"].get("no_cover")),
        **_variant("no_prune", r.get("grade_no_prune"), r["emm"].get("no_prune")),
        "false_positives": r["grade"]["false_positives"], "estimation": r["estimation"],
        "tiers": [fl["tier"] for fl in r["flags"]],
    }


def _variant(name: str, gr: dict | None, findings: list[dict] | None) -> dict[str, Any]:
    """Grade of an alternative top-k list from the same search (``hits_<name>`` …); empty when absent."""
    if gr is None or findings is None:
        return {}
    return {f"hits_{name}": gr["hits"], f"n_findings_{name}": len(findings),
            f"candidates_{name}": gr.get("candidates", []),
            f"false_positives_{name}": [{"class": f["class"], "quality": f["quality"]} for f in gr["false_positives"]]}


def _decoy_attrs(world: World) -> set[str]:
    return {n for n, a in world.attributes.items() if a.role == "decoy"} | \
           {n for n, v in world.nodes.items() if v.kind == "attribute" and v.role == "decoy"}


# ---------------------------------------------------------------------------------------------
# Execution with progress, cancel and resume
# ---------------------------------------------------------------------------------------------


@dataclass
class Progress:
    done: int
    total: int


def execute(spec: ExperimentSpec, *, workers: int | None = None, existing: dict[str, dict] | None = None,
            on_result: Callable[[dict], None] | None = None, cancel: threading.Event | None = None,
            n_sim: int | None = None) -> dict[str, dict]:
    """Run all missing units; ``existing`` results (by unit key) are kept (resume)."""
    results = dict(existing or {})
    todo = [u for u in units(spec, n_sim) if u.key not in results]
    spec_json = spec.model_dump_json(by_alias=True)
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    if workers == 1:
        for u in todo:
            if cancel is not None and cancel.is_set():
                break
            res = run_unit(spec_json, u)
            results[u.key] = res
            if on_result:
                on_result(res)
        return results
    from .parallel import get_pool
    pool = get_pool(workers)
    try:
        for res in pool.imap_unordered(_run_unit_star, [(spec_json, u) for u in todo], chunksize=1):
            results[res["unit"]] = res
            if on_result:
                on_result(res)
            if cancel is not None and cancel.is_set():
                pool.terminate()
                break
        else:
            pool.close()
    finally:
        pool.terminate()
        pool.join()
    return results


def run_adaptive(spec: ExperimentSpec, *, workers: int | None = None, batch: int = 50,
                 existing: dict[str, dict] | None = None, on_result=None, cancel=None) -> dict[str, dict]:
    """Add replications in batches until every headline MCSE <= target_mcse (or max_sim)."""
    results = dict(existing or {})
    target = spec.target_mcse
    max_sim = spec.max_sim or 5000
    n = max(spec.n_sim, batch)
    while True:
        results = execute(spec, workers=workers, existing=results, on_result=on_result, cancel=cancel, n_sim=n)
        if cancel is not None and cancel.is_set():
            return results
        agg = aggregate(spec, results)
        worst = max((v["mcse"] for c in agg["conditions"] for p in c["presets"].values()
                     for v in p["headline"].values() if v["mcse"] is not None), default=0.0)
        if target is None or worst <= target or n >= max_sim:
            return results
        n = min(n + batch, max_sim)


# ---------------------------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------------------------


def rate(x: list[float] | np.ndarray) -> dict[str, float | None]:
    a = np.asarray(x, float)
    if len(a) == 0:
        return {"value": None, "mcse": None, "n": 0}
    p = float(a.mean())
    return {"value": p, "mcse": float(math.sqrt(max(p * (1 - p), 0) / len(a))) if len(a) else None, "n": len(a)}


def mean_mcse(x: list[float]) -> dict[str, float | None]:
    a = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if len(a) == 0:
        return {"value": None, "mcse": None, "n": 0}
    return {"value": float(a.mean()), "mcse": float(a.std(ddof=1) / math.sqrt(len(a))) if len(a) > 1 else None,
            "n": len(a)}


def _threshold(null: list[dict]) -> tuple[float | None, dict]:
    """Calibrated threshold (95th percentile of null top qualities) and its leave-one-out FWER.

    Each null replication is tested against the threshold computed from all *other* null
    replications, so the FWER estimate is out-of-sample yet uses every replication.
    """
    q = np.sort(np.array([r["top_quality"] for r in null], float))
    n = len(q)
    if n < 10:
        return None, rate([])
    exceed = []
    for i in range(n):
        others = np.delete(q, i)
        exceed.append(float(q[i] > np.quantile(others, 0.95)))
    return float(np.quantile(q, 0.95)), rate(exceed)


def aggregate(spec: ExperimentSpec, results: dict[str, dict]) -> dict[str, Any]:
    conds = conditions(spec)
    out_conds = []
    twists = [t.label or f"twist{i}" for i, t in enumerate(spec.base.world.twists)]
    null_of = null_conditions(spec)
    for ci, cond in enumerate(conds):
        presets_out = {}
        for p in spec.presets:
            planted = sorted((r for r in results.values() if r["cond"] == ci and r["preset"] == p
                              and r["variant"] == "planted"), key=lambda r: r["rep"])
            null = [r for r in results.values()
                    if r["cond"] == null_of[ci] and r["preset"] == p and r["variant"] == "null"]
            q_star, fwer = _threshold(null)
            per_twist = {}
            for t in twists:
                hit, hit_sig, ranks, est, true, n_g = [], [], [], [], [], []
                for r in planted:
                    h = next((h for h in r["hits"] if h["twist"] == t), None)
                    hit.append(float(h is not None))
                    hit_sig.append(float(h is not None and (q_star is None or h["quality"] >= q_star)))
                    if h is not None:
                        ranks.append(h["rank"])
                        est.append(h["est_diff"])
                        true.append(h["true_diff"])
                    rec = h or next((m for m in r["misses"] if m["twist"] == t), None)
                    if rec is not None and rec.get("n_g") is not None:
                        n_g.append(rec["n_g"])
                hr = rate(hit)
                per_twist[t] = {
                    "hit_rate": hr, "hit_rate_significant": rate(hit_sig),
                    **_variant_rates(planted, t, "", "hits"),
                    **_variant_rates(planted, t, "_no_cover", "hits_no_cover"),
                    **_variant_rates(planted, t, "_no_prune", "hits_no_prune"),
                    "mean_rank": mean_mcse(ranks), "est_diff": mean_mcse(est),
                    "true_diff": float(np.mean(true)) if true else None,
                    "winners_curse": (float(np.mean(est) - np.mean(true)) if est else None),
                    "n_g": float(np.mean(n_g)) if n_g else None,
                    "unstable": hr["value"] is not None and 0.2 <= hr["value"] <= 0.8,
                }
            fdp: dict[str, list[float]] = {}
            decoy_rate, fdp_sig = [], []
            for r in planted:
                nf = max(len(r["findings"]), 1)
                classes = [f["class"] for f in r["false_positives"]]
                for c in ("null", "decoy", "phantom_downstream", "variance_artifact", "level_shift_leak", "proxy"):
                    fdp.setdefault(c, []).append(classes.count(c) / nf)
                fdp.setdefault("all", []).append(len(classes) / nf)
                sig = [f for f in r["false_positives"] if q_star is None or f["quality"] >= q_star]
                fdp_sig.append(len(sig) / nf)
                decoy_rate.append(sum(f["uses_decoy"] for f in r["findings"]) / nf)
            fdp_nc, fdp_np = ([len(r[f"false_positives_{v}"]) / max(r[f"n_findings_{v}"], 1)
                               for r in planted if f"false_positives_{v}" in r] for v in ("no_cover", "no_prune"))
            estimation = _estimation(planted)
            headline = {f"hit_rate:{t}": v["hit_rate"] for t, v in per_twist.items()}
            headline["fwer_null"] = fwer
            presets_out[p] = {
                "n_planted": len(planted), "n_null": len(null), "threshold": q_star, "fwer_null": fwer,
                "twists": per_twist, "fdp": {k: mean_mcse(v) for k, v in fdp.items()},
                "fdp_significant": mean_mcse(fdp_sig), "fdp_no_cover": mean_mcse(fdp_nc),
                "fdp_no_prune": mean_mcse(fdp_np),
                "decoy_rate": mean_mcse(decoy_rate),
                "estimation": estimation, "headline": headline,
                "null_any_finding": rate([float(len(r["findings"]) > 0) for r in null]),
                "planted_any_significant": rate([float(q_star is not None and r["top_quality"] > q_star)
                                                 for r in planted]),
                "top1_attrs": _top1_attrs(planted), "top1_attrs_null": _top1_attrs(null),
            }
        out_conds.append({"index": ci, "factors": cond, "presets": presets_out})
    return {"name": spec.name, "aim": spec.aim, "n_sim": spec.n_sim, "master_seed": spec.master_seed,
            "presets": spec.presets, "factors": [to_plain(f) for f in spec.factors], "conditions": out_conds,
            "adempt": ademp_card(spec), "sweep": sweep_summary(spec, out_conds)}


def _variant_rates(planted: list[dict], twist: str, suffix: str, key: str) -> dict[str, dict]:
    """Hit and recovery rates (and ranks) of one twist on one graded list; runs without that list are skipped."""
    rows = [next((h for h in r[key] if h["twist"] == twist), None) for r in planted if key in r]
    out = {f"recovery_rate{suffix}": rate([float(h is not None and h.get("recovered_rank") is not None)
                                           for h in rows]),
           f"recovered_rank{suffix}": mean_mcse([h["recovered_rank"] for h in rows
                                                 if h is not None and h.get("recovered_rank") is not None])}
    if suffix:
        out[f"hit_rate{suffix}"] = rate([float(h is not None) for h in rows])
        out[f"mean_rank{suffix}"] = mean_mcse([h["rank"] for h in rows if h is not None])
    return out


LISTS = (("reported", ""), ("no_prune", "_no_prune"), ("no_cover", "_no_cover"))


def experiment_rows(spec: ExperimentSpec, results: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    """Flat rows for analysis: one per planted run x graded list x twist, and one per null run.

    Factor values are columns named by their dotted path; null rows leave out ``world.twists.*`` (shared there).
    """
    from .grade import FP_CLASSES
    conds = conditions(spec)
    twists = [t.label or f"twist{i}" for i, t in enumerate(spec.base.world.twists)]
    runs: list[dict] = []
    nulls: list[dict] = []
    for r in sorted(results.values(), key=lambda r: (r["cond"], r["preset"], r["variant"], r["rep"])):
        base = {"cond": r["cond"], "preset": r["preset"], "rep": r["rep"], "n": r["n"], "n_schools_fit": r["n_schools"],
                **conds[r["cond"]]}
        if r["variant"] == "null":
            fps = r["false_positives"]
            nulls.append({**{k: v for k, v in base.items() if not k.startswith("world.twists")},
                          "top_quality": r["top_quality"], "n_findings": len(r["findings"]),
                          "top_finding": r["findings"][0]["description"] if r["findings"] else None,
                          **{f"fp_{c}": sum(f["class"] == c for f in fps) for c in FP_CLASSES}})
            continue
        for name, suffix in LISTS:
            hits = r.get(f"hits{suffix}")
            if hits is None:
                continue
            fps = r[f"false_positives{suffix}"] if suffix else r["false_positives"]
            n_found = r[f"n_findings{suffix}"] if suffix else len(r["findings"])
            for t in twists:
                h = next((x for x in hits if x["twist"] == t), None)
                miss = next((m for m in r["misses"] if m["twist"] == t), None) if not suffix else None
                # the model path the twist is graded on (its own arrow, or where it lands when the model lacks it)
                path = next((x["path"] for x in (*r["hits"], *r["misses"]) if x["twist"] == t), None)
                runs.append({**base, "list": name, "twist": t, "path": path, "hit": h is not None,
                             "recovered": h is not None and h.get("recovered_rank") is not None,
                             "rank": h["rank"] if h else None,
                             "recovered_rank": h.get("recovered_rank") if h else None,
                             "recovered_quality": h.get("recovered_quality") if h else None,
                             "jaccard": h["jaccard"] if h else None,
                             "est_diff": h["est_diff"] if h else None,
                             "true_diff": h["true_diff"] if h else (miss["true_diff"] if miss else None),
                             "found_as": h["description"] if h else None,
                             "n_findings": n_found, "false_findings": len(fps),
                             **{f"fp_{c}": sum(f["class"] == c for f in fps) for c in FP_CLASSES}})
    return runs, nulls


def candidate_rows(spec: ExperimentSpec, results: dict[str, dict]) -> list[dict]:
    """One row per planted run x graded list x (twist, finding) pair with overlap >= min(tau, proxy_min_jaccard).

    Each row has the overlap, sign, effect, rank, score and whether the twist's path deviates, so recovery can be
    recomputed for another tau or rule without rerunning: at tau, a twist is recovered when some row has
    ``jaccard >= tau``, ``sign_ok`` and ``path_deviates``; ``recovered_rank`` is the lowest such rank (as in
    ``experiment_rows``).
    """
    conds = conditions(spec)
    out: list[dict] = []
    for r in sorted(results.values(), key=lambda r: (r["cond"], r["preset"], r["variant"], r["rep"])):
        if r["variant"] == "null":
            continue
        base = {"cond": r["cond"], "preset": r["preset"], "rep": r["rep"], **conds[r["cond"]]}
        for name, suffix in LISTS:
            cands = r.get(f"candidates{suffix}")
            if cands is None:
                continue
            # the finding's score is ``score`` here: ``quality`` is the quality-measure factor's column
            out.extend({**base, "list": name, **{("score" if k == "quality" else k): v for k, v in c.items()}}
                       for c in cands)
    return out


def _top1_attrs(rows: list[dict]) -> dict[str, dict]:
    """How often each attribute appears in the rank-1 finding (with MCSE)."""
    counts: dict[str, list[float]] = {}
    attrs = sorted({a for r in rows for f in r["findings"] for a in f["attrs"]})
    for a in attrs:
        counts[a] = [float(bool(r["findings"]) and a in r["findings"][0]["attrs"]) for r in rows]
    return {a: rate(v) for a, v in counts.items()}


def _estimation(planted: list[dict]) -> list[dict]:
    paths: dict[str, list[dict]] = {}
    for r in planted:
        for row in r["estimation"]:
            paths.setdefault(row["path"], []).append(row)
    out = []
    for path, rows in paths.items():
        est = np.array([x["estimate"] for x in rows])
        t3 = float(np.mean([x["t3"] for x in rows]))
        err = est - t3
        n = len(rows)
        emp_se = float(est.std(ddof=1)) if n > 1 else None
        bias = float(err.mean())
        model_se = float(np.mean([x["se"] for x in rows]))
        cov = rate([float(x["covered"]) for x in rows])
        cov_n = rate([float(x["covered_naive"]) for x in rows])
        rel_bias = bias / t3 if abs(t3) > 0.05 else None
        se_bias = (model_se - emp_se) / emp_se if emp_se else None
        out.append({
            "path": path, "t3": t3, "bias": bias, "bias_mcse": emp_se / math.sqrt(n) if emp_se else None,
            "rel_bias": rel_bias, "emp_se": emp_se, "rmse": float(np.sqrt((err**2).mean())), "model_se": model_se,
            "se_rel_error": se_bias, "coverage": cov, "coverage_naive": cov_n,
            "flags": {"rel_bias_ok": rel_bias is None or abs(rel_bias) <= 0.10,
                      "se_ok": se_bias is None or abs(se_bias) <= 0.05,
                      "coverage_ok": cov["value"] is not None and 0.91 <= cov["value"] <= 0.98},
        })
    return out


def ademp_card(spec: ExperimentSpec) -> dict:
    return {"aim": spec.aim, "data_generating_mechanism": {"world": spec.base.world.name,
                                                           "world_hash": spec_hash(spec.base.world),
                                                           "factors": [to_plain(f) for f in spec.factors]},
            "estimands": [t.label or t.arrow for t in spec.base.world.twists] + ["full-sample path coefficients"],
            "methods": spec.presets, "performance_measures": ["hit rate", "rank", "winner's curse", "FDP by class",
                                                              "null-World FWER", "bias", "EmpSE", "coverage"],
            "n_sim": spec.n_sim, "target_mcse": spec.target_mcse}


def logistic_n(xs: np.ndarray, ys: np.ndarray, ns: np.ndarray) -> dict[str, float | None]:
    """Fit P(hit) = 1 / (1 + exp(-(a + b log n))) and return n50, n80."""
    xs, ys, ns = np.asarray(xs, float), np.asarray(ys, float), np.asarray(ns, float)
    ok = np.isfinite(xs) & (xs > 0)
    xs, ys, ns = xs[ok], ys[ok], ns[ok]
    if len(xs) < 2 or len(set(xs)) < 2:
        return {"n50": None, "n80": None}
    lx = np.log(xs)

    def nll(theta):
        a, b = theta
        p = 1 / (1 + np.exp(-(a + b * lx)))
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -np.sum(ns * (ys * np.log(p) + (1 - ys) * np.log(1 - p)))

    res = optimize.minimize(nll, x0=[-5.0, 1.0], method="Nelder-Mead", options={"maxiter": 4000})
    a, b = res.x
    if b <= 0:
        return {"n50": None, "n80": None, "a": float(a), "b": float(b)}
    return {"n50": float(math.exp(-a / b)), "n80": float(math.exp((math.log(4) - a) / b)), "a": float(a),
            "b": float(b)}


def sweep_summary(spec: ExperimentSpec, conds: list[dict]) -> dict:
    """For each preset and twist: hit rate vs subgroup size with logistic n50/n80.

    Two curves: "hit" (twist found in the canonical top-k) and "significant" (found AND its quality
    exceeds the null-World calibrated threshold, i.e. what a researcher would report).
    """
    out: dict[str, Any] = {}
    for p in spec.presets:
        for t in [tw.label or f"twist{i}" for i, tw in enumerate(spec.base.world.twists)]:
            curves: dict[str, Any] = {}
            for kind, field in (("hit", "hit_rate"), ("significant", "hit_rate_significant")):
                pts = []
                for c in conds:
                    tw = c["presets"].get(p, {}).get("twists", {}).get(t)
                    if tw and tw["n_g"] is not None and tw[field]["value"] is not None:
                        pts.append({"n_g": tw["n_g"], "hit_rate": tw[field]["value"], "mcse": tw[field]["mcse"],
                                    "n": tw[field]["n"], "factors": c["factors"]})
                fit = (logistic_n(np.array([q["n_g"] for q in pts]), np.array([q["hit_rate"] for q in pts]),
                                  np.array([q["n"] for q in pts])) if len(pts) >= 2 else {"n50": None, "n80": None})
                curves[kind] = {"points": pts, **fit}
            out[f"{p}:{t}"] = {**curves["hit"], "significant": curves["significant"]}
    return out
