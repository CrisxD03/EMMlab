"""Run the experiments in this folder in order, resumably, and write CSVs, a summary and plots.

    uv run python experiments/run_experiments.py                    # every experiment, in order
    uv run python experiments/run_experiments.py exp1a_current       # one experiment
    uv run python experiments/run_experiments.py exp1a_current --reps 2 --out /tmp/x --results /tmp/x/results  # smoke

Each experiment file names a World file, a method preset, the overlap bar (tau), the replications and the factors.
Jobs go into ``<out>/jobs.sqlite`` (``lab_data`` by default, the app's own list, so they also show in the
Experiments tab); progress is saved after every run, so stopping and rerunning resumes. Results per experiment go to
``experiments/results/<name>/``:

- ``runs.csv``: one row per planted run x graded list (reported / no_prune / no_cover) x twist;
- ``findings.csv``: one row per subgroup in each planted run's reported top 10, with what it is (a twist or a
  false-finding class);
- ``candidates.csv.gz``: one row per planted run x graded list x (twist, subgroup) pair that overlaps the twist by
  at least 0.3, with overlap, sign, effect, rank, score and whether the path deviates: recovery under another tau
  or rule without rerunning (gzipped; ``pd.read_csv`` reads it as is);
- ``null_runs.csv``: one row per null run (Experiment 0 for exp1a and exp1b);
- ``summary.csv`` / ``null_summary.csv``: rates with standard errors per cell;
- ``recovery_<quality>.png`` (or ``models_<quality>.png`` when the experiment varies the model),
  ``null_threshold.png``, ``report.html``, ``aggregate.json``, ``spec.json``.

An experiment file can list ``models``: named changes to the World's own arrows (``drop``, ``add``,
``drop_latents``), which become a factor on the Researcher's Model; the search then uses the World's attributes
only, so a latent a model leaves out (not measured) does not become a descriptor. It can also list ``worlds``:
named sets of changes to the World (dotted paths such as ``nodes.grade.scale``), a factor on ``world``.
"""

from __future__ import annotations

import argparse
import math
import threading
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from emmlab.experiment import aggregate, candidate_rows, execute, experiment_rows, set_path, units  # noqa: E402
from emmlab.jobs import JobDB  # noqa: E402
from emmlab.report import experiment_report  # noqa: E402
from emmlab.spec import (  # noqa: E402
    ExperimentSpec,
    Factor,
    GradingConfig,
    ResearcherModel,
    RunSpec,
    World,
    _plainify,
    load_world,
    load_yaml,
    preset,
    to_plain,
)
from emmlab.store import dumps_json  # noqa: E402

HERE = Path(__file__).resolve().parent
ORDER = ["exp1a_current", "exp1b_measures", "exp2_models", "exp3a_skew", "exp3b_scales"]
RENAME = {"world.schools.n_schools": "schools", "world.twists.0.delta": "delta", "method.emm.quality": "quality"}
LIST_STYLE = {"reported": "-", "no_prune": "--"}
COND_COLS = ["world", "model", "quality", "schools", "delta"]
VARIANT_COLS = ("world", "model")  # factors whose levels are whole specs, shown by name


def variant_model(true: ResearcherModel, name: str, change: dict) -> ResearcherModel:
    """The World's own model with arrows dropped or added, or latents left out (with their arrows)."""
    gone = set(change.get("drop_latents", []))
    drop = {" -> ".join(x.strip() for x in a.split("->")) for a in change.get("drop", [])}
    add = [" -> ".join(x.strip() for x in a.split("->")) for a in change.get("add", [])]
    missing = drop - set(true.arrows)
    if missing or gone - set(true.latents) or set(add) & set(true.arrows):
        raise ValueError(f"model {name}: cannot drop {sorted(missing)} / {sorted(gone)} or add an existing arrow")
    arrows = [a for a in true.arrows if a not in drop and not set(map(str.strip, a.split("->"))) & gone] + add
    return ResearcherModel(name=name, latents=[v for v in true.latents if v not in gone], arrows=arrows)


def variant_world(world: World, name: str, patch: dict) -> World:
    """The World with dotted paths set (e.g. ``nodes.grade.scale``), named after the variant."""
    d = to_plain(world)
    for path, value in patch.items():
        set_path(d, path, value)
    d["name"] = name
    return World.model_validate(d)


def build_spec(path: Path, reps: int | None = None, suffix: str = "") -> ExperimentSpec:
    cfg = _plainify(load_yaml(path))
    world = load_world(HERE / cfg["world"])
    true = ResearcherModel.from_world(world)
    base = RunSpec(world=world, model=true, method=preset(cfg["preset"]),
                   grading=GradingConfig(tau=cfg.get("tau", 0.7)))
    factors = [Factor(**f) for f in cfg["factors"]]
    if cfg.get("models"):
        models = [variant_model(true, name, change or {}) for name, change in cfg["models"].items()]
        factors.insert(0, Factor(path="model", levels=[to_plain(m) for m in models]))
        # search the measured attributes only: a latent a model leaves out was not measured, so it is no descriptor
        base.search.attributes = list(world.attributes)
    if cfg.get("worlds"):
        worlds = [variant_world(world, name, patch or {}) for name, patch in cfg["worlds"].items()]
        factors.insert(0, Factor(path="world", levels=[to_plain(w) for w in worlds]))
    return ExperimentSpec(name=cfg["name"] + suffix, aim=cfg.get("aim", ""), base=base, presets=[cfg["preset"]],
                          n_sim=reps or cfg["n_sim"], factors=factors)


def run_job(spec: ExperimentSpec, db: JobDB, workers: int | None, fresh: bool = False) -> dict[str, dict]:
    """Create or resume the job for this exact spec and run what is missing (``fresh``: always a new job)."""
    jid = None if fresh else next((j["id"] for j in db.list() if j["name"] == spec.name and
                                   db.spec(j["id"]).model_dump() == spec.model_dump()), None)
    total = len(units(spec))
    if jid is None:
        jid = db.create(spec, total)
    db.set_status(jid, "running")
    done = db.results(jid)
    print(f"{spec.name}: job {jid}, {len(done)}/{total} runs already done", flush=True)
    t0, counter = time.perf_counter(), {"n": len(done)}
    cancel = threading.Event()

    def on_result(res: dict) -> None:
        db.add_result(jid, res)
        counter["n"] += 1
        if counter["n"] % 200 == 0 or counter["n"] == total:
            print(f"  {counter['n']}/{total} runs, {time.perf_counter() - t0:.0f} s", flush=True)

    try:
        results = execute(spec, workers=workers, existing=done, on_result=on_result, cancel=cancel)
    except KeyboardInterrupt:
        cancel.set()
        db.set_status(jid, "cancelled")
        raise
    db.set_status(jid, "done")
    return results


def _se(p: float, n: int) -> float:
    return math.sqrt(max(p * (1 - p), 0.0) / n) if n else float("nan")


def frame(spec: ExperimentSpec, rows: list[dict]) -> pd.DataFrame:
    """Rows as a table with short factor names (schools, delta, quality, model)."""
    default_q = spec.base.method.emm.quality if spec.base.method.name == "custom" else \
        preset(spec.presets[0]).emm.quality  # type: ignore[arg-type]
    df = pd.DataFrame(rows).rename(columns=RENAME)
    df = df.drop(columns=[c for c in df.columns if c.startswith("world.twists.")], errors="ignore")
    if "quality" not in df.columns:
        df["quality"] = default_q
    for col in VARIANT_COLS:
        if col in df.columns:
            df[col] = df[col].map(lambda m: m["name"] if isinstance(m, dict) else m)
    return df


def tables(spec: ExperimentSpec, results: dict[str, dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs, nulls = experiment_rows(spec, results)
    return frame(spec, runs), frame(spec, nulls)


def findings_table(spec: ExperimentSpec, results: dict[str, dict], runs: pd.DataFrame) -> pd.DataFrame:
    """Every subgroup in each planted run's reported top 10: the twist it recovers or its false-finding class."""
    keys = runs.drop_duplicates("cond").set_index("cond")[[c for c in COND_COLS if c in runs.columns]]
    rows = []
    for r in results.values():
        if r["variant"] != "planted":
            continue
        what = {f["description"]: f["class"] for f in r["false_positives"]}
        what |= {f["description"]: "redundant" for f in r["redundant"]}
        what |= {h["description"]: f"twist {h['twist']}" for h in r["hits"]}
        for f in r["findings"]:
            rows.append({"cond": r["cond"], "rep": r["rep"], **keys.loc[r["cond"]].to_dict(), "rank": f["rank"],
                         "description": f["description"], "quality_value": f["quality"], "n": f["n"],
                         "is": what.get(f["description"], "unknown")})
    return pd.DataFrame(rows).sort_values(["cond", "rep", "rank"])


def summarize(runs: pd.DataFrame, nulls: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    nkeys = [c for c in (*VARIANT_COLS, "quality", "schools") if c in nulls.columns]
    null_rows = []
    for key, g in nulls.groupby(nkeys):
        top = g["top_finding"].fillna("(none)").value_counts(normalize=True).head(3)
        null_rows.append({**dict(zip(nkeys, key, strict=True)), "reps": len(g),
                          "threshold_95": g["top_quality"].quantile(0.95),
                          "median_top_quality": g["top_quality"].median(),
                          "share_with_findings": float((g["n_findings"] > 0).mean()),
                          "scale_artifacts_per_run": float(g["fp_scale_artifact"].mean()),
                          "most_common_top_finding": "; ".join(f"{k} ({v:.0%})" for k, v in top.items())})
    null_sum = pd.DataFrame(null_rows)
    thr = {tuple(r[k] for k in nkeys): r["threshold_95"] for _, r in null_sum.iterrows()}
    keys = [*nkeys, "delta", "list", "twist"]
    rows = []
    for key, g in runs.groupby(keys):
        n = len(g)
        rec, hit = float(g["recovered"].mean()), float(g["hit"].mean())
        th = thr.get(key[: len(nkeys)])
        sig = g["recovered"] & (g["recovered_quality"].fillna(-math.inf) >= (th if th is not None else -math.inf))
        found = g[g["hit"]]
        share = (g["false_findings"] / g["n_findings"].where(g["n_findings"] > 0)).fillna(0)
        path = next(iter(g["path"].dropna().mode()), None)
        rows.append({**dict(zip(keys, key, strict=True)), "path": path, "reps": n,
                     "recovery_rate": rec, "recovery_se": _se(rec, n), "hit_rate": hit,
                     "recovered_above_null_threshold": float(sig.mean()),
                     "mean_recovered_rank": g["recovered_rank"].mean(),
                     "mean_found_diff": found["est_diff"].mean(), "true_diff": g["true_diff"].mean(),
                     "false_finding_share": float(share.mean()),
                     "level_shift_leaks_per_run": float(g["fp_level_shift_leak"].mean()),
                     "proxies_per_run": float(g["fp_proxy"].mean()),
                     "chance_findings_per_run": float(g["fp_null"].mean()),
                     "scale_artifacts_per_run": float(g["fp_scale_artifact"].mean())})
    return pd.DataFrame(rows), null_sum


def plot_recovery(summary: pd.DataFrame, out: Path) -> list[Path]:
    paths = []
    twists = sorted(summary["twist"].unique())
    for q, gq in summary.groupby("quality"):
        fig, axes = plt.subplots(1, len(twists), figsize=(5.5 * len(twists), 4.2), sharey=True, squeeze=False)
        deltas = sorted(gq["delta"].unique())
        colors = plt.cm.viridis([i / max(len(deltas) - 1, 1) for i in range(len(deltas))])
        for ax, t in zip(axes[0], twists, strict=True):
            for color, d in zip(colors, deltas, strict=True):
                for lst, style in LIST_STYLE.items():
                    g = gq[(gq["twist"] == t) & (gq["delta"] == d) & (gq["list"] == lst)].sort_values("schools")
                    if g.empty:
                        continue
                    ax.errorbar(g["schools"], g["recovery_rate"], yerr=1.96 * g["recovery_se"], color=color,
                                linestyle=style, marker="o", ms=3, capsize=2,
                                label=f"Δ = {d}" if lst == "reported" else None)
            ax.set_xscale("log")
            ax.minorticks_off()
            ax.set_xticks(sorted(gq["schools"].unique()), [str(s) for s in sorted(gq["schools"].unique())])
            ax.set_ylim(-0.03, 1.03)
            ax.set_title(t, fontsize=10)
            ax.set_xlabel("schools (20 students each)")
            ax.grid(alpha=0.3)
        axes[0][0].set_ylabel("recovery rate (95% CI)")
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(deltas) + 1, fontsize=8, frameon=False,
                   title="solid: reported top 10 · dashed: without steps (b)+(c)", title_fontsize=8)
        fig.suptitle(f"Recovery of the planted twists, quality measure: {q}", fontsize=11)
        fig.tight_layout(rect=(0, 0.1, 1, 1))
        p = out / f"recovery_{q}.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(p)
    return paths


def plot_models(summary: pd.DataFrame, out: Path, col: str = "model") -> list[Path]:
    """Reported recovery per model (or World variant): one panel per twist, a line per (schools, delta) point."""
    paths = []
    rep = summary[summary["list"] == "reported"]
    models = list(dict.fromkeys(rep[col]))
    twists = sorted(rep["twist"].unique())
    points = sorted({(s, d) for s, d in zip(rep["schools"], rep["delta"], strict=True)})
    colors = plt.cm.viridis([i / max(len(points) - 1, 1) for i in range(len(points))])
    for q, gq in rep.groupby("quality"):
        fig, axes = plt.subplots(1, len(twists), figsize=(6 * len(twists), 4.6), sharey=True, squeeze=False)
        for ax, t in zip(axes[0], twists, strict=True):
            for color, (s, d) in zip(colors, points, strict=True):
                g = gq[(gq["twist"] == t) & (gq["schools"] == s) & (gq["delta"] == d)].set_index(col).loc[models]
                ax.plot(range(len(models)), g["recovery_rate"], color=color, marker="o", ms=3,
                        label=f"{s} schools, Δ = {d}")
            ax.set_xticks(range(len(models)), [m.split("_")[0] if col == "model" else m for m in models])
            ax.set_ylim(-0.03, 1.03)
            ax.set_title(t, fontsize=10)
            ax.grid(alpha=0.3)
        axes[0][0].set_ylabel("recovery rate, reported top 10")
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7, frameon=False)
        what = "Researcher's Model" if col == "model" else "World variant"
        fig.suptitle(f"Recovery by {what}, quality measure: {q}\n" + " · ".join(models), fontsize=9)
        fig.tight_layout(rect=(0, 0.14, 1, 1))
        p = out / f"{col}s_{q}.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(p)
    return paths


def plot_null(null_sum: pd.DataFrame, out: Path) -> Path:
    qualities = sorted(null_sum["quality"].unique())
    col = next((c for c in VARIANT_COLS if c in null_sum.columns), None)
    by_model = col is not None
    panels = len(qualities) if by_model else 1
    fig, axes = plt.subplots(1, panels, figsize=(5.5 * panels, 4), squeeze=False)
    groups = [(ax, q, null_sum[null_sum["quality"] == q]) for ax, q in zip(axes[0], qualities, strict=True)] \
        if by_model else [(axes[0][0], None, null_sum)]
    for ax, title, data in groups:
        for name, g in data.groupby(col if col else "quality"):
            g = g.sort_values("schools")
            ax.plot(g["schools"], g["threshold_95"], marker="o", label=f"{name}: 95th percentile")
            if not by_model:
                ax.plot(g["schools"], g["median_top_quality"], marker=".", linestyle=":", label=f"{name}: median")
        ax.set_xscale("log")
        ax.minorticks_off()
        ticks = sorted(data["schools"].unique())
        ax.set_xticks(ticks, [str(s) for s in ticks])
        ax.set_xlabel("schools (20 students each)")
        ax.set_ylabel("top quality with no twist planted")
        ax.set_title(f"Null World, {title}" if title else "Null World: how high scores get by chance", fontsize=11)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    p = out / "null_threshold.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def write_outputs(spec: ExperimentSpec, results: dict[str, dict], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    runs, nulls = tables(spec, results)
    summary, null_sum = summarize(runs, nulls)
    runs.to_csv(out / "runs.csv", index=False)
    findings_table(spec, results, runs).to_csv(out / "findings.csv", index=False)
    # large: gzip, without a timestamp so the same results give the same bytes
    frame(spec, candidate_rows(spec, results)).to_csv(out / "candidates.csv.gz", index=False,
                                                      compression={"method": "gzip", "mtime": 0})
    nulls.to_csv(out / "null_runs.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False, float_format="%.4g")
    null_sum.to_csv(out / "null_summary.csv", index=False, float_format="%.4g")
    col = next((c for c in VARIANT_COLS if c in summary.columns), None)
    if col:
        plot_models(summary, out, col)
    else:
        plot_recovery(summary, out)
    plot_null(null_sum, out)
    agg = aggregate(spec, results)
    (out / "aggregate.json").write_text(dumps_json(agg))
    (out / "spec.json").write_text(dumps_json(to_plain(spec)))
    (out / "report.html").write_text(experiment_report(agg, spec.base.world))
    print(f"  wrote {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", default=ORDER)
    ap.add_argument("--reps", type=int, default=None, help="override the replications (smoke runs)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("lab_data"), help="where the job database lives")
    ap.add_argument("--results", type=Path, default=HERE / "results")
    ap.add_argument("--fresh", action="store_true", help="start a new job even if one with this spec exists")
    args = ap.parse_args()
    db = JobDB(args.out / "jobs.sqlite")
    for name in args.names:
        spec = build_spec(HERE / f"{name}.yaml", args.reps, "_smoke" if args.reps else "")
        t0 = time.perf_counter()
        results = run_job(spec, db, args.workers, args.fresh)
        print(f"{spec.name}: {len(results)} runs in {time.perf_counter() - t0:.0f} s", flush=True)
        write_outputs(spec, results, args.results / spec.name)


if __name__ == "__main__":
    main()
