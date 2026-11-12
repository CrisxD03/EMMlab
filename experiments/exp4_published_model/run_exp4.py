"""Experiment 4: the supervisor's loop with the paper's own recipe. Resumable, parallel.

    source experiments/exp4_published_model/env.sh
    exp4py experiments/exp4_published_model/run_exp4.py --workers 15
    exp4py experiments/exp4_published_model/run_exp4.py --reps 2 --out /tmp/exp4_smoke --workers 4   # smoke

Worlds:
- ``no_subgroups``: Figure 2 as published (not ``null``: pandas reads that word back as missing). Every fake
  Greece gets a full PLS fit (question A: do the 17 numbers come back?) and the original search (question B: what
  does EMM report when no subgroup exists?).
- ``planted_<name>``: Figure 2 with one real finding of Step 0 planted on feedback -> achievement (question C). The
  planted slopes keep the paper's overall slope and add the finding's real inside-vs-rest gap:
  ``inside = -0.135 + (b_inside_real - b_all_real)``, ``outside = -0.135 + (b_outside_real - b_all_real)``.
  Every planted run also records PLS on the planted subgroup and on the rest, found or not.

Outputs in ``results/exp4_published_model/``: ``coefs.csv`` (no_subgroups: 17 estimates and SEs per run), ``runs.csv`` (one row per
run), ``findings.csv`` (every reported subgroup), ``planted.json`` (the planted Worlds). A finished (world, rep) is
skipped on rerun; replications are scheduled round-robin over the Worlds.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import search as S  # noqa: E402
from frame import load_real_frame, prepared  # noqa: E402
from pls_estimates import inner_estimates  # noqa: E402
from grading import findings_frame, grade, ranked_qualities, real_finding_hits  # noqa: E402
from recipe import FIGURE2, Planted, Recipe  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "exp4_published_model"
MASTER_SEED = 20261101
PATH = "feedback -> Achievement"
PLAN = {"no_subgroups": 1000, "planted_fb_medium": 200, "planted_size_medium": 200, "planted_both_medium": 200}
REAL = {  # Step 0 (step0_real.py): the real search on the 4,564 students found these three
    "planted_fb_medium": "school_fb_tertile=='medium'",
    "planted_size_medium": "school_size_tertile=='medium'",
    "planted_both_medium": "school_fb_tertile=='medium' AND school_size_tertile=='medium'",
}
WORLD_CODE = {w: i for i, w in enumerate(PLAN)}
RUN_COLUMNS = (["world", "rep", "seconds", "planted", "n_findings", "top_quality", "top_description", "n_unrelated",
                "n_overlap", "recovered", "top_is_planted", "n_planted_listed", "match_rank", "match_jaccard",
                "match_quality", "match_mirror", "match_estimate_sg", "match_estimate_rest", "planted_inside_est",
                "planted_outside_est", "planted_disjoint"]
               + [f"q{i}" for i in range(1, 4)] + [f"{p}_{k}" for k in REAL for p in ("listed", "rank")])

_G: dict = {}


def build_worlds(frame: pd.DataFrame, cfg) -> dict[str, Recipe]:
    """The null recipe and one planted recipe per real finding, with the planted slopes."""
    b_all = inner_estimates(frame, cfg).loc[PATH, "estimate"]
    base = FIGURE2[("feedback", "Achievement")]
    worlds = {"no_subgroups": Recipe(FIGURE2)}
    for name, desc in REAL.items():
        m = S.cover_of(desc, frame)
        b_in = inner_estimates(frame, cfg, m).loc[PATH, "estimate"]
        b_out = inner_estimates(frame, cfg, ~m).loc[PATH, "estimate"]
        worlds[name] = Recipe(FIGURE2, Planted(desc, m, base + (b_in - b_all), base + (b_out - b_all)))
    return worlds


def _init() -> None:
    real, _ = load_real_frame()
    frame, cfg = prepared(real)
    _G["template"] = frame
    _G["worlds"] = build_worlds(frame, cfg)
    _G["real_masks"] = {k: S.cover_of(d, frame) for k, d in REAL.items()}


def run_unit(unit: tuple[str, int]) -> tuple[dict, list[dict], list[dict]]:
    world, rep = unit
    rec = _G["worlds"][world]
    frame, cfg = prepared(rec.simulate(_G["template"], np.random.SeedSequence([MASTER_SEED, WORLD_CODE[world], rep])))
    t0 = time.perf_counter()
    found = S.run_search(frame, cfg, S.SearchSpec("path"))
    planted = rec.planted.mask if rec.planted else None
    row = {"world": world, "rep": rep, "seconds": round(time.perf_counter() - t0, 1),
           "planted": rec.planted.description if rec.planted else ""}
    inside = rec.planted.inside if rec.planted else 0.0
    outside = rec.planted.outside if rec.planted else 0.0
    row.update(grade(found, planted, direction=float(np.sign(inside - outside)), quality="path"))
    row.update(ranked_qualities(found, 3))
    coefs = []
    if planted is None:
        row.update(real_finding_hits(found, _G["real_masks"]))
        fit = inner_estimates(frame, cfg)
        coefs = [{"rep": rep, "path": p, "estimate": float(r.estimate), "se": float(r.se)} for p, r in fit.iterrows()]
    else:
        a = inner_estimates(frame, cfg, planted).loc[PATH]
        b = inner_estimates(frame, cfg, ~planted).loc[PATH]
        row.update(planted_inside_est=float(a.estimate), planted_outside_est=float(b.estimate),
                   planted_disjoint=bool(abs(a.estimate - b.estimate) > 1.96 * (a.se + b.se)))
    f = findings_frame(found, planted)
    f.insert(0, "rep", rep)
    f.insert(0, "world", world)
    return row, f.to_dict("records"), coefs


def units(cap: int | None, done: set) -> list[tuple[str, int]]:
    out = []
    for rep in range(max(PLAN.values())):
        for w, n in PLAN.items():
            if rep < (min(n, cap) if cap else n) and (w, rep) not in done:
                out.append((w, rep))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    _init()
    meta = {"master_seed": MASTER_SEED, "plan": PLAN, "error_sd": _G["worlds"]["no_subgroups"].error_sd,
            "planted": {k: {"description": w.planted.description, "size": int(w.planted.mask.sum()),
                            "inside": w.planted.inside, "outside": w.planted.outside}
                        for k, w in _G["worlds"].items() if w.planted}}
    (out / "planted.json").write_text(json.dumps(meta, indent=1))
    runs_csv, finds_csv, coefs_csv = out / "runs.csv", out / "findings.csv", out / "coefs.csv"
    done = set()
    if runs_csv.exists():
        prev = pd.read_csv(runs_csv)
        done = set(zip(prev.world, prev.rep, strict=True))
    todo = units(args.reps, done)
    print(f"{len(todo)} runs to do ({len(done)} done), {args.workers} workers", flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers, initializer=_init) as pool:
        for i, (row, finds, coefs) in enumerate(pool.imap_unordered(run_unit, todo), start=1):
            pd.DataFrame([row]).reindex(columns=RUN_COLUMNS).to_csv(runs_csv, mode="a", header=not runs_csv.exists(),
                                                                    index=False)
            if finds:
                pd.DataFrame(finds).to_csv(finds_csv, mode="a", header=not finds_csv.exists(), index=False)
            if coefs:
                pd.DataFrame(coefs).to_csv(coefs_csv, mode="a", header=not coefs_csv.exists(), index=False)
            if i % 50 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"{i}/{len(todo)}, {el / 60:.1f} min, eta {el / i * (len(todo) - i) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
