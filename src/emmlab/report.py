"""Self-contained HTML reports (inline CSS + SVG, no external assets) for Runs and Experiments."""

from __future__ import annotations

import html
import math
from typing import Any

from jinja2 import Environment, select_autoescape

from . import truth
from .spec import World
from .world import compile_world

CSS = """
:root{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#6b6a65;--line:#e3e0d8;--accent:#3559a8;--good:#2f7d4f;--bad:#b0412e;
--warn:#a8741a;--card:#ffffff}
@media (prefers-color-scheme: dark){:root{--bg:#161615;--fg:#ecebe6;--muted:#a3a19a;--line:#33322e;
--accent:#8fa8e8;--good:#6fc28f;--bad:#e8826f;--warn:#e0b25a;--card:#1f1e1c}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,system-ui,Segoe UI,Roboto,sans-serif;
margin:0;padding:24px 16px}
main{max-width:1100px;margin:0 auto}h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:28px 0 8px}
.muted{color:var(--muted)}.badge{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;
border:1px solid var(--line)}.badge.warn{color:var(--warn);border-color:var(--warn)}
.badge.bad{color:var(--bad);border-color:var(--bad)}.badge.good{color:var(--good);border-color:var(--good)}
table{border-collapse:collapse;width:100%;font-size:13px;background:var(--card)}
th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
th{font-weight:600;color:var(--muted)}td.num{font-variant-numeric:tabular-nums;text-align:right}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 16px;margin:12px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}
svg text{fill:var(--fg);font-size:11px}code{font-size:12px}.wrap{overflow-x:auto}
"""


def fmt(x: Any, d: int = 3) -> str:
    if x is None:
        return "–"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int, float)):
        if isinstance(x, float) and not math.isfinite(x):
            return "–"
        return f"{x:.{d}f}" if isinstance(x, float) else str(x)
    return html.escape(str(x))


def pm(r: dict | None, d: int = 3) -> str:
    """value ± MCSE (every reported rate carries its Monte Carlo error)."""
    if not r or r.get("value") is None:
        return "–"
    m = r.get("mcse")
    return f"{r['value']:.{d}f} ± {m:.{d}f}" if m is not None else f"{r['value']:.{d}f}"


def sweep_svg(points: list[dict], fit: dict, analytic: list[tuple[float, float]] | None = None,
              title: str = "") -> str:
    W, H, L, B, T, R = 520, 260, 48, 36, 18, 12
    xs = [p["n_g"] for p in points] or [1]
    lo, hi = math.log(max(min(xs) * 0.8, 1)), math.log(max(xs) * 1.25)

    def X(v: float) -> float:
        return L + (math.log(max(v, 1e-9)) - lo) / (hi - lo or 1) * (W - L - R)

    def Y(v: float) -> float:
        return T + (1 - v) * (H - T - B)

    parts = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="{html.escape(title)}">']
    for v in (0, 0.5, 0.8, 1):
        parts.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(v)}" y2="{Y(v)}" stroke="var(--line)"/>'
                     f'<text x="{L - 6}" y="{Y(v) + 4}" text-anchor="end">{v:.1f}</text>')
    if analytic:
        d = " ".join(f"{'M' if i == 0 else 'L'}{X(n):.1f},{Y(p):.1f}" for i, (n, p) in enumerate(analytic))
        parts.append(f'<path d="{d}" fill="none" stroke="var(--muted)" stroke-dasharray="4 3"/>')
    if fit.get("b") and fit.get("a") is not None:
        a, b = fit["a"], fit["b"]
        pts = []
        for i in range(60):
            n = math.exp(lo + (hi - lo) * i / 59)
            pts.append((n, 1 / (1 + math.exp(-(a + b * math.log(n))))))
        d = " ".join(f"{'M' if i == 0 else 'L'}{X(n):.1f},{Y(p):.1f}" for i, (n, p) in enumerate(pts))
        parts.append(f'<path d="{d}" fill="none" stroke="var(--accent)" stroke-width="1.5"/>')
    for p in points:
        x, y, m = X(p["n_g"]), Y(p["hit_rate"]), (p.get("mcse") or 0) * 1.96
        parts.append(f'<line x1="{x}" x2="{x}" y1="{Y(min(1, p["hit_rate"] + m))}" '
                     f'y2="{Y(max(0, p["hit_rate"] - m))}" stroke="var(--accent)"/>'
                     f'<circle cx="{x}" cy="{y}" r="3.5" fill="var(--accent)"/>')
    for n in sorted({round(p["n_g"]) for p in points}):
        parts.append(f'<text x="{X(n)}" y="{H - B + 16}" text-anchor="middle">{n}</text>')
    parts.append(f'<text x="{(W + L) / 2}" y="{H - 4}" text-anchor="middle">students in subgroup (log scale)</text>')
    parts.append("</svg>")
    return "".join(parts)


def bars_svg(items: list[tuple[str, float, float | None]], title: str = "") -> str:
    W, row = 520, 22
    H = row * len(items) + 10
    mx = max([v for _, v, _ in items] + [1e-9])
    parts = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="{html.escape(title)}">']
    for i, (label, v, m) in enumerate(items):
        y = 5 + i * row
        w = (v / mx) * (W - 220)
        parts.append(f'<text x="0" y="{y + 14}">{html.escape(label)}</text>'
                     f'<rect x="150" y="{y + 3}" width="{w:.1f}" height="14" rx="3" fill="var(--accent)"/>'
                     f'<text x="{156 + w:.1f}" y="{y + 14}">{v:.3f}{"" if m is None else f" ± {m:.3f}"}</text>')
    parts.append("</svg>")
    return "".join(parts)


ENV = Environment(autoescape=select_autoescape(["html"]))
ENV.filters["fmt"] = fmt
ENV.filters["pm"] = pm

RUN_T = ENV.from_string("""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Run report</title><style>{{css}}</style></head>
<body><main>
<h1>Run: {{ spec.world.name }} · {{ man.method_preset }}</h1>
<p class="muted">seed path {{ man.seed_path }} · {{ r.n }} students in {{ r.n_schools }} schools ·
truth: {{ r.truth_source }} · <span class="badge warn">{{ man.label }}</span></p>
<h2>World equations</h2><div class="card"><code>{% for e in equations %}{{ e }}<br>{% endfor %}</code></div>
<h2>Full-sample paths vs step estimand (T3)</h2><div class="wrap"><table>
<tr><th>path</th><th>estimate</th><th>SE (cluster)</th><th>SE (naive)</th><th>T3</th><th>error</th></tr>
{% for row in r.estimation %}<tr><td>{{row.path}}</td><td class="num">{{row.estimate|fmt}}</td>
<td class="num">{{row.se|fmt}}</td><td class="num">{{row.se_naive|fmt}}</td><td class="num">{{row.t3|fmt}}</td>
<td class="num">{{row.error|fmt}}</td></tr>{% endfor %}</table></div>
<h2>Findings (canonical)</h2><div class="wrap"><table>
<tr><th>#</th><th>subgroup</th><th>quality</th><th>n</th><th>schools</th><th>clusters</th><th>best path</th>
<th>subgroup b</th><th>baseline b</th><th>Δb</th></tr>
{% for f in findings %}<tr><td>{{f.rank}}</td><td>{{f.description}}</td><td class="num">{{f.quality|fmt}}</td>
<td class="num">{{f.n}}</td><td class="num">{{f.n_schools}}</td><td><span class="badge {{ tiers[loop.index0] }}">
{{ tiers[loop.index0] }}</span></td><td>{{f.best_path}}</td><td class="num">{{f.best.b_g|fmt}}</td>
<td class="num">{{f.best.b_b|fmt}}</td><td class="num">{{f.best.diff|fmt}}</td></tr>{% endfor %}</table></div>
<h2>Scorecard (this draw)</h2><div class="grid">
<div class="card"><b>Hits</b>{% for h in g.hits %}<div>{{h.twist}}: rank {{h.rank}}, Δb {{h.est_diff|fmt}}
(true {{h.true_diff|fmt}}), Jaccard {{h.jaccard|fmt(2)}}</div>{% else %}<div class="muted">none</div>{% endfor %}</div>
<div class="card"><b>Misses</b>{% for m in g.misses %}<div>{{m.twist}} (true Δb {{m.true_diff|fmt}})</div>
{% else %}<div class="muted">none</div>{% endfor %}</div>
<div class="card"><b>False positives by class</b>{% for k, v in g.fp_by_class.items() %}<div>{{k}}: {{v}}</div>
{% endfor %}</div></div>
{% if boot %}<h2>School bootstrap ({{boot.reps}} replicates)</h2><div class="wrap"><table>
<tr><th>path</th><th>SE</th><th>interval</th></tr>{% for p, v in boot.full.items() %}<tr><td>{{p}}</td>
<td class="num">{{v.se|fmt}}</td><td class="num">[{{v.ci[0]|fmt}}, {{v.ci[1]|fmt}}]</td></tr>{% endfor %}
</table></div>{% endif %}
<p class="muted">Generated by emmlab · manifest: world {{man.world}}, model {{man.model}}, method {{man.method}}</p>
</main></body></html>""")

EXP_T = ENV.from_string("""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Experiment report</title>
<style>{{css}}</style></head><body><main>
<h1>Experiment: {{ a.name }}</h1><p class="muted">{{ a.aim }}</p>
<div class="card"><b>ADEMP</b><br>Data-generating mechanism: World <code>{{a.adempt.data_generating_mechanism.world}}</code>
· factors {{ a.factors|length }} · methods {{ a.presets|join(", ") }} · n_sim {{ a.n_sim }} per condition ·
master seed {{ a.master_seed }}</div>
{% for c in a.conditions %}<h2>Condition {{ c.index }} {% for k, v in c.factors.items() %}· {{k}} = {{v}}{% endfor %}</h2>
<div class="wrap"><table><tr><th>preset</th><th>twist</th><th>hit rate</th><th>hit rate (sig.)</th>
<th>mean rank</th><th>recovered</th><th>recovered without (b, c)</th><th>recovered without (d)</th><th>winner's curse</th>
<th>n in subgroup</th><th>stability</th></tr>
{% for p, d in c.presets.items() %}{% for t, v in d.twists.items() %}<tr><td>{{p}}</td><td>{{t}}</td>
<td class="num">{{v.hit_rate|pm}}</td><td class="num">{{v.hit_rate_significant|pm}}</td>
<td class="num">{{v.mean_rank|pm(2)}}</td><td class="num">{{v.recovery_rate|pm}}</td>
<td class="num">{{v.recovery_rate_no_prune|pm}}</td><td class="num">{{v.recovery_rate_no_cover|pm}}</td>
<td class="num">{{v.winners_curse|fmt}}</td><td class="num">{{v.n_g|fmt(0)}}</td>
<td>{% if v.unstable %}<span class="badge warn">unstable</span>{% else %}<span class="badge good">stable</span>{% endif %}</td>
</tr>{% endfor %}{% endfor %}</table></div>
<div class="grid">{% for p, d in c.presets.items() %}<div class="card"><b>{{p}}</b>
<div>null-World FWER: {{ d.fwer_null|pm }}</div><div>false-discovery share: {{ d.fdp.all|pm }}</div>
<div>significant false discoveries: {{ d.fdp_significant|pm }}</div>
<div>false-discovery share without (b, c): {{ d.fdp_no_prune|pm }}</div>
<div>false-discovery share without (d): {{ d.fdp_no_cover|pm }}</div><div>decoy rate: {{ d.decoy_rate|pm }}</div>
<div>calibrated threshold: {{ d.threshold|fmt }}</div>
{{ fdp_svgs[c.index][p]|safe }}</div>{% endfor %}</div>
<details><summary>Estimation block</summary>{% for p, d in c.presets.items() %}<div class="wrap"><table>
<tr><th>{{p}} path</th><th>T3</th><th>bias</th><th>rel. bias</th><th>EmpSE</th><th>RMSE</th><th>coverage</th>
<th>naive coverage</th><th>flags</th></tr>{% for e in d.estimation %}<tr><td>{{e.path}}</td><td class="num">{{e.t3|fmt}}</td>
<td class="num">{{e.bias|fmt}} ± {{e.bias_mcse|fmt}}</td><td class="num">{{e.rel_bias|fmt}}</td>
<td class="num">{{e.emp_se|fmt}}</td><td class="num">{{e.rmse|fmt}}</td><td class="num">{{e.coverage|pm}}</td>
<td class="num">{{e.coverage_naive|pm}}</td><td>{% for k, ok in e.flags.items() %}<span class="badge {{ 'good' if ok else 'bad' }}">{{k}}</span> {% endfor %}</td></tr>
{% endfor %}</table></div>{% endfor %}</details>{% endfor %}
{% if sweeps %}<h2>Sample-size sweep</h2>{% for k, s in sweeps.items() %}<div class="card"><b>{{k}}</b> ·
n₅₀ = {{ s.n50|fmt(0) }} · n₈₀ = {{ s.n80|fmt(0) }}{{ s.svg|safe }}</div>{% endfor %}{% endif %}
</main></body></html>""")


def world_equations(world: World, cw=None) -> list[str]:
    cw = cw if cw is not None else compile_world(world)
    out = []
    for v in cw.visible:
        node = world.nodes[v]
        if node.root is not None:
            out.append(f"{v} ~ Normal({node.root.mean:g}, {node.root.sd:g})")
            continue
        terms = [f"{cw.slopes.get(f'{a.parent} -> {v}', 0):+.3g}·{a.parent}" for a in world.arrows if a.child == v]
        terms += [f"{s.delta:+.3g}·[{' & '.join(f'{k}={x}' for k, x in s.when.items())}]"
                  for s in world.shifts if s.target == v]
        terms += [f"{t.delta:+.3g}·[{' & '.join(f'{k}={x}' for k, x in t.when.items())}]·{t.parent}"
                  for t in world.twists if t.child == v]
        sd = cw.solved.get(v, {}).get("noise_sd", 0.0)
        out.append(f"{v} = {node.recipe.intercept if node.recipe else 0:g} {' '.join(terms)} + ε(sd {sd:.3g})")
    return out


def run_report(result: dict) -> str:
    spec = World.model_validate(result["spec"]["world"])
    findings = []
    for f in result["emm"]["findings"]:
        best = next((p for p in f["paths"] if p["path"] == f["best_path"]), f["paths"][0] if f["paths"] else {})
        findings.append({**f, "best": best})
    return RUN_T.render(css=CSS, r=result, man=result["manifest"], spec=result["spec"], findings=findings,
                        tiers=[fl["tier"] for fl in result["flags"]], g=result["grade"],
                        boot=result.get("bootstrap"), equations=world_equations(spec))


def experiment_report(agg: dict, base_world: World | None = None) -> str:
    fdp_svgs: dict[int, dict[str, str]] = {}
    for c in agg["conditions"]:
        fdp_svgs[c["index"]] = {}
        for p, d in c["presets"].items():
            items = [(k, v["value"] or 0.0, v["mcse"]) for k, v in d["fdp"].items() if k != "all"]
            fdp_svgs[c["index"]][p] = bars_svg(items, "false-discovery share by class")
    sweeps = {}
    for k, s in agg.get("sweep", {}).items():
        if len(s["points"]) >= 2:
            analytic = None
            sweeps[k] = {**s, "svg": sweep_svg(s["points"], s, analytic, k)}
    return EXP_T.render(css=CSS, a=agg, fdp_svgs=fdp_svgs, sweeps=sweeps)


def power_curve_points(p: float, delta: float, sigma2: float, alpha: float, ns: list[float]) -> list[tuple[float, float]]:
    return [(n, float(truth.power_slope_difference(n, p, delta, sigma2, alpha))) for n in ns]
