"""Build a single-file offline preview of the web app (for visual checks where no server can run).

The real backend computes fixtures (presets, truths, models, runs, scatter, reports) once; a fetch shim
serves them from inside the page. All JavaScript and CSS are inlined, so the file opens from file://.
Edits do not recompute truths (the shim answers by World name) — use it to look, not to test logic.

    uv run python scripts/offline_preview.py            # writes $TMPDIR/emmlab-preview/index.html
"""

from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("PREVIEW_DIR", Path(tempfile.gettempdir()) / "emmlab-preview"))

SHIM = """<script>
window.__FX__ = %s;
(function(){
  const FX = window.__FX__;
  const nameOf = (b) => (b && b.world && b.world.name) || (b && b.spec && b.spec.world && b.spec.world.name) || 'minimal';
  const resp = (o, s=200) => new Response(typeof o === 'string' ? o : JSON.stringify(o),
      {status: s, headers: {'content-type': typeof o === 'string' ? 'text/html' : 'application/json'}});
  window.fetch = async (input, init) => {
    const url = new URL(typeof input === 'string' ? input : input.url, 'http://x');
    const path = url.pathname; const method = ((init && init.method) || 'GET').toUpperCase();
    const body = init && init.body ? JSON.parse(init.body) : null;
    const key = method + ' ' + path;
    if (FX[key]) return resp(FX[key]);
    if (key === 'POST /api/world/truth') return resp(FX.truth_by_name[nameOf(body)] || FX.truth_by_name.minimal);
    if (key === 'POST /api/model/from_world') return resp(FX.model_by_name[nameOf(body)] || FX.model_by_name.minimal);
    if (key === 'POST /api/run') return resp(FX.run_by_name[nameOf(body)] || FX.run_by_name.minimal);
    if (path.startsWith('/api/run/') && path.endsWith('/scatter')) return resp(FX.scatter_by_key[path.split('/')[3]]);
    if (path.startsWith('/api/run/') && path.endsWith('/report')) return resp(FX.report_by_key[path.split('/')[3]]);
    if (key === 'POST /api/model/estimands') return resp(FX.estimands_by_name[nameOf(body)] || {source: 'analytic', equations: {}});
    if (key === 'POST /api/search/descriptors') return resp(FX.descriptors_by_name[nameOf(body)] || {});
    if (key === 'POST /api/world/yaml') return resp({text: FX.yaml_by_name[nameOf(body)] || '', hash: ''});
    if (key === 'GET /api/experiments') return resp([]);
    return resp({detail: 'offline preview: not available'}, 503);
  };
})();
</script>"""


def fixtures() -> dict:
    warnings.simplefilter("ignore")
    os.environ.setdefault("EMMLAB_DATA", str(OUT / "data"))
    from fastapi.testclient import TestClient

    from emmlab.server.app import app
    c = TestClient(app)
    fx: dict = {"GET /api/presets": c.get("/api/presets").json(), "GET /api/glossary": c.get("/api/glossary").json()}
    for n in ("current", "recommended"):
        fx[f"GET /api/preset_method/{n}"] = c.get(f"/api/preset_method/{n}").json()
    fx.update(truth_by_name={}, model_by_name={}, run_by_name={}, scatter_by_key={}, report_by_key={},
              estimands_by_name={}, descriptors_by_name={}, yaml_by_name={})
    for p in fx["GET /api/presets"]:
        name, w = p["name"], p["world"]
        fx["truth_by_name"][name] = c.post("/api/world/truth", json={"world": w}).json()
        m = c.post("/api/model/from_world", json={"world": w}).json()
        fx["model_by_name"][name] = m
        fx["yaml_by_name"][name] = p["yaml"]
        fx["estimands_by_name"][name] = c.post("/api/model/estimands", json={"world": w, "model": m,
                                                                             "analysis": {}}).json()
        fx["descriptors_by_name"][name] = c.post("/api/search/descriptors", json={"world": w, "model": m}).json()
        spec = {"world": w, "model": m, "method": fx["GET /api/preset_method/recommended"], "seed": 0}
        run = c.post("/api/run", json={"spec": spec}).json()
        fx["run_by_name"][name] = run
        rk = run["run_key"]
        fx["report_by_key"][rk] = c.get(f"/api/run/{rk}/report").text
        if run["emm"]["findings"]:
            f0 = run["emm"]["findings"][0]
            fx["scatter_by_key"][rk] = c.get(f"/api/run/{rk}/scatter", params={"path": f0["best_path"],
                                                                                 "rank": 1}).json()
    return fx


def main() -> int:
    build = OUT / "build"
    cfg = ROOT / "web" / "vite.preview.config.ts"
    cfg.write_text(
        'import { mergeConfig } from "vite";\nimport base from "./vite.config";\n'
        "export default mergeConfig(base, { base: './', build: { outDir: process.env.PREVIEW_OUT, emptyOutDir: true,"
        " assetsInlineLimit: 100000000, cssCodeSplit: false,"
        " rollupOptions: { output: { inlineDynamicImports: true } } } });\n")
    try:
        subprocess.run(["npx", "vite", "build", "--config", "vite.preview.config.ts"], cwd=ROOT / "web", check=True,
                       env={**os.environ, "PREVIEW_OUT": str(build)}, stdout=subprocess.DEVNULL)
    finally:
        cfg.unlink(missing_ok=True)
    fx = json.dumps(fixtures()).replace("</", "<\\/")
    html = (build / "index.html").read_text()
    js = Path(glob.glob(str(build / "assets" / "*.js"))[0]).read_text().replace("</script", "<\\/script")
    css = "".join(Path(p).read_text() for p in glob.glob(str(build / "assets" / "*.css")))
    html = re.sub(r'<script type="module"[^>]*src="[^"]+"></script>', "", html)
    html = re.sub(r'<link rel="stylesheet"[^>]*>', "", html)
    html = html.replace("<head>", "<head>" + SHIM % fx + f"<style>{css}</style>", 1)
    html = html.replace("</body>", f'<script type="module">{js}</script></body>')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "index.html").write_text(html)
    print(f"file://{(OUT / 'index.html').resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
