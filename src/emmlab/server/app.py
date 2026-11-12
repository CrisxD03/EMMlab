"""FastAPI server: live truth for the canvas, runs, experiment jobs (SSE progress), reports."""

from __future__ import annotations

import asyncio
import json
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError

from .. import truth as tr
from ..emm import quality_catalog
from ..estimand import EstimandEngine
from ..experiment import aggregate, execute, run_adaptive, units
from ..jobs import JobDB
from ..report import experiment_report, run_report, world_equations
from ..run import execute_run
from ..spec import (
    AnalysisConfig,
    ExperimentSpec,
    ResearcherModel,
    RunSpec,
    SearchSpace,
    World,
    _plainify,
    dump_yaml,
    load_world,
    preset,
    spec_hash,
    to_plain,
)
from ..stages import residualization_cases
from ..store import Store
from ..unfold import unfold_world
from ..world import InfeasibleTargetError, WorldValidationError, compile_world, validate_world
from ..yamldoc import update_document

PRESETS = Path(__file__).resolve().parents[1] / "presets"
STATIC = Path(__file__).parent / "static"
DATA = Path(os.environ.get("EMMLAB_DATA", Path.cwd() / "lab_data"))

app = FastAPI(title="emmlab", version="0.1.0")
store = Store(DATA / "artifacts")
db = JobDB(DATA / "jobs.sqlite")
_cancel: dict[str, threading.Event] = {}
_runs: dict[str, dict] = {}


def _issues_from(e: Exception) -> list[dict]:
    if isinstance(e, (WorldValidationError, InfeasibleTargetError)):
        return [dict(i) for i in e.issues]
    if isinstance(e, ValidationError):
        return [{"element": ".".join(str(x) for x in err["loc"]), "message": err["msg"]} for err in e.errors()]
    return [{"element": "world", "message": str(e)}]


def _parse_world(data: dict) -> World:
    try:
        return World.model_validate(data)
    except ValidationError as e:
        raise HTTPException(422, {"issues": _issues_from(e)}) from e


# ----------------------------------------------------------------------------- presets & glossary


@app.get("/api/presets")
def presets() -> list[dict]:
    out = []
    for p in sorted(PRESETS.glob("*.yaml")):
        w = load_world(p)
        out.append({"name": p.stem, "description": w.description, "world": to_plain(w), "yaml": p.read_text(),
                    "twists": len(w.twists), "null_variant": bool(w.twists)})
    return out


@app.get("/api/glossary")
def glossary() -> dict:
    return json.loads((Path(__file__).resolve().parents[1] / "glossary.json").read_text())


@app.get("/api/qualities")
def qualities() -> list[dict]:
    """Every quality function in emm.QUALITY_FUNCTIONS, for the Search & method dropdown and the Results tab."""
    return quality_catalog()


@app.get("/api/schema")
def schema() -> dict:
    return {"world": World.model_json_schema(by_alias=True), "run": RunSpec.model_json_schema(by_alias=True),
            "experiment": ExperimentSpec.model_json_schema(by_alias=True)}


# ----------------------------------------------------------------------------- World: truth, yaml


class WorldBody(BaseModel):
    world: dict[str, Any]


@app.post("/api/world/truth")
def world_truth(body: WorldBody) -> dict:
    """Validate + compile + derived quantities. Never raises for World problems: returns issues."""
    try:
        w = World.model_validate(body.world)
    except ValidationError as e:
        return {"ok": False, "issues": _issues_from(e)}
    issues = validate_world(w)
    if issues:
        return {"ok": False, "issues": [dict(i) for i in issues]}
    try:
        cw = compile_world(w)
    except (WorldValidationError, InfeasibleTargetError) as e:
        return {"ok": False, "issues": _issues_from(e)}
    d = tr.derived(cw)
    return {"ok": True, "issues": [], "derived": d, "equations": world_equations(w, cw), "hash": spec_hash(w),
            "unfolded": unfold_world(cw), "warnings": cw.warnings}


class YamlBody(BaseModel):
    world: dict[str, Any]
    text: str = ""


@app.post("/api/world/yaml")
def world_yaml(body: YamlBody) -> dict:
    w = _parse_world(body.world)
    plain = to_plain(w)
    text = update_document(body.text, plain) if body.text.strip() else dump_yaml(plain)
    return {"text": text, "hash": spec_hash(w)}


class TextBody(BaseModel):
    text: str


@app.post("/api/world/parse")
def world_parse(body: TextBody) -> dict:
    import io

    from ..spec import _yaml
    try:
        data = _plainify(_yaml().load(io.StringIO(body.text)))
        w = World.model_validate(data)
    except ValidationError as e:
        return {"ok": False, "issues": _issues_from(e)}
    except Exception as e:  # YAML syntax
        return {"ok": False, "issues": [{"element": "yaml", "message": str(e).splitlines()[0]}]}
    return {"ok": True, "world": to_plain(w), "hash": spec_hash(w)}


@app.post("/api/world/null")
def world_null(body: WorldBody) -> dict:
    return {"world": to_plain(_parse_world(body.world).without_twists())}


# ----------------------------------------------------------------------------- Researcher's Model


@app.post("/api/model/from_world")
def model_from_world(body: WorldBody) -> dict:
    return to_plain(ResearcherModel.from_world(_parse_world(body.world)))


class ModelBody(BaseModel):
    world: dict[str, Any]
    model: dict[str, Any]
    analysis: dict[str, Any] = {}


@app.post("/api/model/estimands")
def model_estimands(body: ModelBody) -> dict:
    """Full-population T3 per equation, plus triangle/chain labels and R² before/after residualization."""
    w = _parse_world(body.world)
    m = ResearcherModel.model_validate(body.model)
    an = AnalysisConfig.model_validate(body.analysis)
    cw = compile_world(w)
    unknown = [x for x in m.latents if x not in w.nodes or not w.nodes[x].observed]
    if unknown:
        raise HTTPException(422, {"issues": [{"element": "model", "message": f"not observed latents: {unknown}"}]})
    base = an.model_copy(update={"residualize_on": []})
    try:
        before = EstimandEngine(cw, m, base).full
        after = EstimandEngine(cw, m, an).full
        source = "analytic"
    except tr.NotAnalytic:
        return {"source": "oracle-required", "equations": {}}
    cases = {}
    for x in an.residualize_on:
        for c, kind in residualization_cases(m, x).items():
            cases[c] = kind
    eqs = {}
    for c, e in after.items():
        b = before[c]
        eqs[c] = {"regs": e.regs, "coef": e.coef.tolist(), "coef_before": b.coef.tolist(),
                  "r2": e.r2, "r2_before": b.r2, "case": cases.get(c)}
    return {"source": source, "equations": eqs}


# ----------------------------------------------------------------------------- Search space helpers


@app.post("/api/search/descriptors")
def search_descriptors(body: ModelBody) -> dict:
    """Descriptor distributions and default cut points (from one generated dataset)."""
    from ..emm import numeric_cuts
    from ..generate import generate
    w = _parse_world(body.world)
    cw = compile_world(w)
    df = generate(cw, 0)
    ss = SearchSpace.model_validate(body.analysis.get("search", {})) if body.analysis.get("search") else SearchSpace()
    m = ResearcherModel.model_validate(body.model)
    out = {}
    for c in df.columns:
        if c in m.latents or c == "school_id" or c not in {**w.attributes, **w.nodes}:
            continue
        if c in w.attributes:
            vc = df[c].astype(str).value_counts(normalize=True)
            out[c] = {"type": "categorical", "level": w.attributes[c].level, "role": w.attributes[c].role,
                      "values": {k: float(v) for k, v in vc.items()}}
        else:
            b = ss.binning.get(c, ss.default_binning)
            x = df[c].to_numpy(float)
            hist, edges = np.histogram(x, bins=30)
            out[c] = {"type": "numeric", "level": w.nodes[c].level, "role": w.nodes[c].role,
                      "hist": hist.tolist(), "edges": edges.tolist(),
                      "cuts": [v for _, v in numeric_cuts(x, df["school_id"].to_numpy(), b.n_bins, b.weighting)]}
    return out


# ----------------------------------------------------------------------------- Runs


class RunBody(BaseModel):
    spec: dict[str, Any]


@app.post("/api/run")
def run(body: RunBody) -> dict:
    try:
        spec = RunSpec.model_validate(body.spec)
        r = execute_run(spec, store)
    except (WorldValidationError, InfeasibleTargetError, ValidationError) as e:
        raise HTTPException(422, {"issues": _issues_from(e)}) from e
    _runs[r["run_key"]] = r
    return r


@app.get("/api/run/{key}/report", response_class=HTMLResponse)
def run_report_html(key: str) -> str:
    r = _runs.get(key)
    if r is None:
        p = store.path(key) / "run.json"
        if not p.exists():
            raise HTTPException(404)
        r = json.loads(p.read_text())
    return run_report(r)


@app.get("/api/run/{key}/scatter")
def run_scatter(key: str, path: str, rank: int = 0, limit: int = 1500) -> dict:
    """Sample of (regressor, outcome, in_subgroup) points for a model path, after the analysis stages."""
    from ..emm import Condition, Description
    from ..run import dataset_for
    from ..stages import prepare
    r = _runs.get(key)
    if r is None:
        p = store.path(key) / "run.json"
        if not p.exists():
            raise HTTPException(404)
        r = json.loads(p.read_text())
    spec = RunSpec.model_validate(r["spec"])
    df, _ = dataset_for(spec, store, tuple(r["manifest"]["seed_path"][1:]))
    prep = prepare(df, spec.model, spec.method.analysis, spec.search.filter)
    reg, child = [x.strip() for x in path.split("->")]
    if child not in prep.equations or reg not in prep.equations[child]:
        raise HTTPException(422, "path not in the Researcher's Model")
    x = prep.X[child][:, prep.equations[child].index(reg)]
    y = prep.y[child]
    frame = df.loc[prep.frame].reset_index(drop=True)
    grp = np.zeros(len(y), dtype=bool)
    desc = None
    if rank > 0 and rank <= len(r["emm"]["findings"]):
        f = r["emm"]["findings"][rank - 1]
        d = Description(tuple(Condition(**c) for c in f["conditions"]))
        grp = d.mask(frame)
        desc = f["description"]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(y), size=min(limit, len(y)), replace=False)
    return {"path": path, "subgroup": desc, "x": x[idx].round(4).tolist(), "y": y[idx].round(4).tolist(),
            "in_group": grp[idx].tolist()}


@app.get("/api/run/{key}/contrast")
def run_contrast(key: str, rank: int) -> dict:
    """Contrastive analysis of finding ``rank`` of a stored run: the finding against its flipped slices."""
    from ..contrast import contrast_finding
    from ..emm import Condition, Description, PathQuality
    from ..run import dataset_for
    from ..stages import prepare
    r = _runs.get(key)
    if r is None:
        p = store.path(key) / "run.json"
        if not p.exists():
            raise HTTPException(404, "unknown run")
        r = json.loads(p.read_text())
    findings = r["emm"]["findings"]
    if not 1 <= rank <= len(findings):
        raise HTTPException(404, "no such finding")
    f = findings[rank - 1]
    spec = RunSpec.model_validate(r["spec"])
    df, _ = dataset_for(spec, store, tuple(r["manifest"]["seed_path"][1:]))
    prep = prepare(df, spec.model, spec.method.analysis, spec.search.filter)
    frame = df.loc[prep.frame].reset_index(drop=True)
    pq = PathQuality(prep, spec.method.emm, spec.search, rng_seed=spec.seed)
    desc = Description(tuple(Condition(**c) for c in f["conditions"]))
    return {**contrast_finding(frame, prep, desc, pq), "rank": rank}


@app.get("/api/preset_method/{name}")
def method_preset(name: str) -> dict:
    if name not in ("current", "recommended"):
        raise HTTPException(404)
    return to_plain(preset(name))  # type: ignore[arg-type]


# ----------------------------------------------------------------------------- Experiments


class ExperimentBody(BaseModel):
    spec: dict[str, Any]
    workers: int | None = None
    adaptive: bool = False


def _start(jid: str, spec: ExperimentSpec, workers: int | None, adaptive: bool) -> None:
    ev = threading.Event()
    _cancel[jid] = ev

    def target() -> None:
        db.set_status(jid, "running")
        try:
            fn = run_adaptive if adaptive else execute
            fn(spec, workers=workers, existing=db.results(jid), on_result=lambda r: db.add_result(jid, r),
               cancel=ev)
            db.set_status(jid, "cancelled" if ev.is_set() else "done")
        except Exception as e:  # pragma: no cover - surfaced to the UI
            db.set_status(jid, "error", repr(e))

    threading.Thread(target=target, daemon=True).start()


@app.post("/api/experiments")
def create_experiment(body: ExperimentBody) -> dict:
    try:
        spec = ExperimentSpec.model_validate(body.spec)
    except ValidationError as e:
        raise HTTPException(422, {"issues": _issues_from(e)}) from e
    jid = db.create(spec, len(units(spec)))
    _start(jid, spec, body.workers, body.adaptive)
    return {"id": jid, "total": len(units(spec))}


@app.get("/api/experiments")
def list_experiments() -> list[dict]:
    return db.list()


@app.get("/api/experiments/{jid}")
def get_experiment(jid: str) -> dict:
    j = db.get(jid)
    if j is None:
        raise HTTPException(404)
    spec = ExperimentSpec.model_validate(j["spec"])
    res = db.results(jid)
    j["aggregate"] = aggregate(spec, res) if res else None
    return j


@app.post("/api/experiments/{jid}/cancel")
def cancel_experiment(jid: str) -> dict:
    ev = _cancel.get(jid)
    if ev is not None:
        ev.set()
    return {"ok": True}


@app.post("/api/experiments/{jid}/resume")
def resume_experiment(jid: str, workers: int | None = None) -> dict:
    j = db.get(jid)
    if j is None:
        raise HTTPException(404)
    if j["status"] == "running" and jid in _cancel and not _cancel[jid].is_set():
        return {"ok": True, "status": "running"}
    _start(jid, ExperimentSpec.model_validate(j["spec"]), workers, False)
    return {"ok": True, "status": "running"}


@app.get("/api/experiments/{jid}/events")
async def experiment_events(jid: str) -> StreamingResponse:
    async def gen():
        last = None
        while True:
            j = db.get(jid)
            if j is None:
                yield "event: error\ndata: {}\n\n"
                return
            state = (j["status"], j["done"], j["total"])
            if state != last:
                yield f"data: {json.dumps({'status': j['status'], 'done': j['done'], 'total': j['total']})}\n\n"
                last = state
            if j["status"] in ("done", "cancelled", "error"):
                return
            await asyncio.sleep(0.5)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/experiments/{jid}/report", response_class=HTMLResponse)
def experiment_report_html(jid: str) -> str:
    j = get_experiment(jid)
    if not j["aggregate"]:
        raise HTTPException(409, "no results yet")
    return experiment_report(j["aggregate"])


# ----------------------------------------------------------------------------- frontend


if STATIC.exists():
    app.mount("/assets", StaticFiles(directory=STATIC / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        f = STATIC / path
        if path and f.is_file():
            return FileResponse(f)
        return FileResponse(STATIC / "index.html")
