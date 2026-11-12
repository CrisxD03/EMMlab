"""Command line: validate, truth, run, experiment, report, reproduce, presets, serve."""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import typer

from .spec import (
    ExperimentSpec,
    ResearcherModel,
    RunSpec,
    World,
    load_spec,
    load_world,
    preset,
    spec_hash,
    to_plain,
)
from .store import Store, dumps_json

app = typer.Typer(add_completion=False, help="emmlab: ground-truth laboratory for EMM on path models")
PRESETS = Path(__file__).parent / "presets"


def _world(path: str) -> World:
    p = Path(path)
    if not p.exists() and (PRESETS / f"{path}.yaml").exists():
        p = PRESETS / f"{path}.yaml"
    return load_world(p)


def _run_spec(path: str, method: str, seed: int) -> RunSpec:
    p = Path(path)
    raw = None
    if p.exists():
        from .spec import _plainify, load_yaml
        raw = _plainify(load_yaml(p))
    if raw is not None and "world" in raw:
        return RunSpec.model_validate(raw)
    w = _world(path)
    return RunSpec(world=w, model=ResearcherModel.from_world(w), method=preset(method), seed=seed)  # type: ignore


@app.command()
def presets() -> None:
    """List the preset Worlds."""
    for p in sorted(PRESETS.glob("*.yaml")):
        w = load_world(p)
        typer.echo(f"{p.stem:20s} {w.description[:90]}")


@app.command()
def validate(world: str) -> None:
    """Validate a World spec (path or preset name)."""
    from .world import InfeasibleTargetError, WorldValidationError, compile_world, validate_world
    w = _world(world)
    issues = validate_world(w)
    if issues:
        for i in issues:
            typer.echo(f"✗ {i}")
        raise typer.Exit(1)
    try:
        cw = compile_world(w)
    except (WorldValidationError, InfeasibleTargetError) as e:
        for i in e.issues:
            typer.echo(f"✗ {i}")
        raise typer.Exit(1) from e
    for msg in cw.warnings:
        typer.echo(f"! {msg}")
    typer.echo(f"✓ {w.name}: {len(cw.visible)} variables, {len(cw.assigns)} cells, hash {spec_hash(w)}")


@app.command()
def truth(world: str, as_json: bool = typer.Option(False, "--json")) -> None:
    """Print the World's derived True Quantities."""
    from . import truth as tr
    from .report import world_equations
    from .world import compile_world
    w = _world(world)
    d = tr.derived(compile_world(w))
    if as_json:
        typer.echo(json.dumps(d, indent=1))
        return
    for e in world_equations(w):
        typer.echo(e)
    typer.echo("")
    for n, v in d["nodes"].items():
        typer.echo(f"{n:14s} mean {v['mean']:+.3f}  sd {v['sd']:.3f}  R² {v['r2']:.3f}  ICC {v['icc']:.3f}")
    typer.echo("")
    for k, v in d["total_effects"].items():
        typer.echo(f"total {k:30s} {v:+.4f}")


@app.command()
def run(world: str, method: str = "recommended", seed: int = 0, out: Path = Path("lab_data/runs"),
        bootstrap: int = typer.Option(0, help="school bootstrap replicates (0 = off)"),
        permutation: int = typer.Option(0, help="search-level permutation null replicates (0 = off)"),
        holdout: bool = typer.Option(False, help="discovery/confirmation split with Benjamini-Hochberg")) -> None:
    """Run one seed: generate, analyse, EMM, grade, and write a report."""
    from .report import run_report
    from .run import execute_run
    from .spec import BootstrapConfig
    spec = _run_spec(world, method, seed)
    if bootstrap:
        spec.bootstrap = BootstrapConfig(reps=bootstrap)
    spec.permutation_reps = permutation
    spec.holdout = holdout
    store = Store(out.parent / "artifacts")
    r = execute_run(spec, store)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{spec.world.name}_{spec.method.name}_s{spec.seed}"
    (out / f"{stem}.json").write_text(dumps_json(r))
    (out / f"{stem}.html").write_text(run_report(r))
    for f in r["emm"]["findings"][:5]:
        typer.echo(f"#{f['rank']} {f['description']:40s} q={f['quality']:.3f} n={f['n']} best={f['best_path']}")
    g = r["grade"]
    typer.echo(f"hits {len(g['hits'])} misses {len(g['misses'])} false positives {len(g['false_positives'])} "
               f"({r['manifest']['label']})")
    typer.echo(f"report: {out / (stem + '.html')}")


@app.command()
def real(data_spec: Path, model_spec: Path, method: str = "recommended", out: Path = Path("lab_data/real"),
         weights: bool = True, brr: bool = True, permutation: int = 0, holdout: bool = False) -> None:
    """Analyse a real PISA-format file (no answer key): weights, plausible values, BRR, validation."""
    from .loader import RealDataSpec, load_real
    from .run import execute_real
    from .spec import SearchSpace
    ds = load_spec(data_spec, RealDataSpec)
    model = load_spec(model_spec, ResearcherModel)
    m = preset(method)  # type: ignore[arg-type]
    if weights:
        m.analysis.weights = "final"
    if brr:
        m.analysis.se = "brr"
    df, frame = load_real(ds)
    r = execute_real(df, frame, model, m, SearchSpace(), permutation_reps=permutation, holdout_split=holdout)
    out.mkdir(parents=True, exist_ok=True)
    (out / "real_result.json").write_text(dumps_json(r))
    typer.echo(f"{frame['rows_after_listwise']} students in {frame['schools']} schools ({r['label']})")
    for f in r["emm"]["findings"][:5]:
        typer.echo(f"#{f['rank']} {f['description']:40s} q={f['quality']:.3f} n={f['n']} best={f['best_path']}")
    typer.echo(f"results: {out / 'real_result.json'}")


@app.command()
def experiment(spec_path: Path, out: Path = Path("lab_data/experiments"), workers: int = 0,
               adaptive: bool = False) -> None:
    """Run (or resume) an Experiment from a YAML spec; writes results + report."""
    from .experiment import aggregate, execute, run_adaptive, units
    from .jobs import JobDB
    from .report import experiment_report
    spec = load_spec(spec_path, ExperimentSpec)
    db = JobDB(out / "jobs.sqlite")
    existing = [j for j in db.list() if j["name"] == spec.name]
    jid = next((j["id"] for j in existing if db.spec(j["id"]).model_dump() == spec.model_dump()), None)
    total = len(units(spec))
    if jid is None:
        jid = db.create(spec, total)
    db.set_status(jid, "running")
    done = db.results(jid)
    typer.echo(f"job {jid}: {len(done)}/{total} units already done")
    cancel = threading.Event()

    def on_result(res: dict) -> None:
        db.add_result(jid, res)

    try:
        fn = run_adaptive if adaptive else execute
        results = fn(spec, workers=workers or None, existing=done, on_result=on_result, cancel=cancel)
    except KeyboardInterrupt:
        cancel.set()
        db.set_status(jid, "cancelled")
        typer.echo("cancelled; rerun the same command to resume")
        raise typer.Exit(130) from None
    agg = aggregate(spec, results)
    db.set_status(jid, "done")
    d = out / spec.name
    d.mkdir(parents=True, exist_ok=True)
    (d / "aggregate.json").write_text(dumps_json(agg))
    (d / "spec.json").write_text(dumps_json(to_plain(spec)))
    (d / "report.html").write_text(experiment_report(agg))
    typer.echo(f"report: {d / 'report.html'}")


@app.command()
def reproduce(manifest: Path) -> None:
    """Re-execute a stored Run or Experiment and check byte-identical results."""
    from .experiment import aggregate, execute
    from .run import execute_run
    d = manifest.parent
    if (d / "aggregate.json").exists():
        spec = ExperimentSpec.model_validate_json((d / "spec.json").read_text())
        agg = aggregate(spec, execute(spec))
        same = dumps_json(agg) == (d / "aggregate.json").read_text()
    else:
        stored = json.loads((d / "run.json").read_text())
        spec = RunSpec.model_validate(stored["spec"])
        seed_key = tuple(stored["manifest"]["seed_path"][1:])
        r = execute_run(spec, None, seed_key)
        same = dumps_json(_strip(r)) == dumps_json(_strip(stored))
    typer.echo("identical" if same else "DIFFERENT")
    if not same:
        raise typer.Exit(1)


def _strip(r: dict) -> dict:
    return {k: v for k, v in r.items() if k not in ("run_key",)}


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8765, data: Path = Path("lab_data")) -> None:
    """Serve the web app (API + built frontend)."""
    import os

    import uvicorn
    os.environ["EMMLAB_DATA"] = str(data.resolve())
    uvicorn.run("emmlab.server.app:app", host=host, port=port, log_level="info")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
