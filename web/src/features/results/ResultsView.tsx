import { useEffect, useState } from "react";
import { api, downloadReport } from "../../api/client";
import type { Finding, RunResult } from "../../api/types";
import { Equation, formatG, subgroupEquation } from "../../components/Equation";
import { Empty, Term, Tip } from "../../components/ui";
import { useTruth } from "../../state/truth";
import { useUI } from "../../state/ui";
import { fmt, parseArrowKey } from "../../state/worldOps";
import { repeat200, runOnce } from "../commands";
import { IssuesPanel } from "../world/IssuesPanel";
import { layered, type Pt, type Route, roundedPath } from "../world/layout";
import { ContrastCard } from "./ContrastCard";
import { ContrastPlotCard } from "./ContrastPlot";
import { EquationsCard } from "./EquationsCard";
import { deviatingRule, intervalSeLabel, isDeviating, useQualityInfo } from "./quality";
import { TierBadge, TierLegend } from "./TierBadge";

const FP_CLASSES = ["null", "decoy", "phantom_downstream", "variance_artifact", "level_shift_leak", "proxy"];

// ----------------------------------------------------------------------------- coefficient hover

function CoefTip({
  r,
  path,
  rank,
  value,
  kind,
}: {
  r: RunResult;
  path: string;
  rank: number;
  value: number;
  kind: string;
}) {
  const t = r.truths[path];
  const ft = rank > 0 ? r.finding_truth[rank - 1] : undefined;
  const m = r.manifest;
  const overrides = Object.keys(m.overrides ?? {});
  return (
    <Tip
      content={
        <div data-testid="coef-tip">
          <div style={{ fontWeight: 600 }}>
            {kind} {path}: {fmt(value)}
          </div>
          <dl>
            <dt>estimand</dt>
            <dd>{r.estimand_label}</dd>
            <dt>T1 structural</dt>
            <dd>
              {fmt(t?.t1_direct_avg)} <span style={{ opacity: 0.7 }}>direct, avg ({t?.units_t1_t2})</span>
            </dd>
            <dt>T2 causal target</dt>
            <dd>
              {fmt(t?.t2_total_avg)} <span style={{ opacity: 0.7 }}>total, avg</span>
            </dd>
            <dt>T3 step estimand</dt>
            <dd>
              {fmt(t?.t3)} <span style={{ opacity: 0.7 }}>({t?.units_t3})</span>
            </dd>
            {ft?.common ? (
              <>
                <dt>true Δ (T3)</dt>
                <dd>
                  {fmt(ft.common[path])}{" "}
                  <span style={{ opacity: 0.7 }}>subgroup − baseline, {ft.source}</span>
                </dd>
              </>
            ) : null}
            <dt>provenance</dt>
            <dd>
              World {m.world} · model {m.model} · seed {JSON.stringify(m.seed_path)} · preset{" "}
              {m.method_preset}
              {overrides.length ? ` (+${overrides.join(", ")})` : ""} · run {r.run_key}
            </dd>
          </dl>
        </div>
      }
    >
      <button
        type="button"
        className="derived"
        style={{ border: 0, borderBottom: "1px dotted var(--derived-fg)" }}
        data-coef={path}
        aria-label={`${kind} ${path}: ${fmt(value)}. Estimand ${r.estimand_label}`}
      >
        {fmt(value)}
      </button>
    </Tip>
  );
}

// ----------------------------------------------------------------------------- model diagram per subgroup

const NW = 132;
const NH = 40;

/** Synchronous first paint (longest-path layers); replaced by the ELK layout as soon as it is ready. */
function simpleLayers(latents: string[], arrows: string[]): Record<string, Pt> {
  const layer = new Map(latents.map((l) => [l, 0]));
  const edges = arrows.map((k) => parseArrowKey(k));
  for (let i = 0; i < latents.length; i++)
    for (const [p, c] of edges)
      if ((layer.get(p) ?? 0) + 1 > (layer.get(c) ?? 0)) layer.set(c, (layer.get(p) ?? 0) + 1);
  const rows = new Map<number, number>();
  const pos: Record<string, Pt> = {};
  for (const l of latents) {
    const L = layer.get(l) ?? 0;
    const r = rows.get(L) ?? 0;
    rows.set(L, r + 1);
    pos[l] = { x: 16 + L * 330, y: 16 + r * 120 };
  }
  return pos;
}

function SubgroupDiagram({ r, finding }: { r: RunResult; finding: Finding | null }) {
  const model = r.spec.model;
  const qinfo = useQualityInfo(r.spec.method.emm.quality);
  const mode = finding ? "subgroup" : "full";
  const [lay, setLay] = useState<{
    key: string;
    positions: Record<string, Pt>;
    routes: Record<string, Route>;
  } | null>(null);
  const layoutKey = `${JSON.stringify(model)}|${mode}`;
  useEffect(() => {
    let cancel = false;
    const labelW = mode === "subgroup" ? 212 : 76;
    layered(
      model.latents.map((id) => ({ id, width: NW, height: NH })),
      model.arrows.map((k) => {
        const [from, to] = parseArrowKey(k);
        return { id: k, from, to, label: { width: labelW, height: 26 } };
      }),
      { layerGap: 50, nodeGap: 34 },
    )
      .then((res) => {
        if (!cancel) setLay({ key: layoutKey, positions: res.positions, routes: res.routes });
      })
      .catch(() => {});
    return () => {
      cancel = true;
    };
  }, [layoutKey, model, mode]);
  const ready = lay && lay.key === layoutKey ? lay : null;
  const pos = ready?.positions ?? simpleLayers(model.latents, model.arrows);
  const routes = ready?.routes ?? {};
  const rows = model.arrows.map((k) => {
    const fp = finding?.paths.find((p) => p.path === k);
    const est = r.estimation.find((e) => e.path === k);
    const [p, c] = parseArrowKey(k);
    const a = pos[p];
    const b = pos[c];
    const route = routes[k];
    const points: Pt[] | null = route
      ? route.points
      : a && b
        ? [
            { x: a.x + NW, y: a.y + NH / 2 },
            { x: b.x, y: b.y + NH / 2 },
          ]
        : null;
    const label = route?.label
      ? { x: route.label.x + route.label.width / 2, y: route.label.y + route.label.height / 2 }
      : points
        ? {
            x: (points[0].x + points[points.length - 1].x) / 2,
            y: (points[0].y + points[points.length - 1].y) / 2 - 14,
          }
        : null;
    return { key: k, fp, est, points, label };
  });
  const maxAbs = Math.max(1e-9, ...rows.map((x) => Math.abs(x.fp?.diff ?? 0)));
  const xs = [
    ...Object.values(pos).map((p) => p.x + NW),
    ...rows.flatMap((x) => (x.label ? [x.label.x + 110] : [])),
  ];
  const ys = [
    ...Object.values(pos).map((p) => p.y + NH),
    ...rows.flatMap((x) => (x.label ? [x.label.y + 16] : [])),
    ...rows.flatMap((x) => x.points?.map((p) => p.y) ?? []),
  ];
  const width = Math.max(...xs) + 16;
  const height = Math.max(...ys) + 16;
  return (
    <div
      className="relative overflow-x-auto"
      data-testid="subgroup-diagram"
      data-layout={ready ? "elk" : "simple"}
    >
      <div className="relative" style={{ width, height }}>
        <svg width={width} height={height} className="absolute inset-0" aria-hidden>
          <defs>
            <marker
              id="arr"
              viewBox="0 0 10 10"
              refX="9"
              refY="5"
              markerWidth="8"
              markerHeight="8"
              markerUnits="userSpaceOnUse"
              orient="auto-start-reverse"
            >
              <path d="M0,0 L10,5 L0,10 z" fill="var(--muted)" />
            </marker>
            <marker
              id="arr-hot"
              viewBox="0 0 10 10"
              refX="9"
              refY="5"
              markerWidth="10"
              markerHeight="10"
              markerUnits="userSpaceOnUse"
              orient="auto-start-reverse"
            >
              <path d="M0,0 L10,5 L0,10 z" fill="var(--twist)" />
            </marker>
          </defs>
          {rows.map(({ key, fp, points }) => {
            if (!points) return null;
            const s = finding ? Math.abs(fp?.diff ?? 0) / maxAbs : 0;
            const hot = !!finding && !!fp && isDeviating(fp, qinfo);
            return (
              <path
                key={key}
                d={roundedPath(points)}
                fill="none"
                stroke={hot ? "var(--twist)" : "var(--line-strong)"}
                strokeWidth={1.4 + 3 * s}
                markerEnd={hot ? "url(#arr-hot)" : "url(#arr)"}
              />
            );
          })}
        </svg>
        {model.latents.map((l) => {
          const p = pos[l];
          if (!p) return null;
          return (
            <div
              key={l}
              className="wnode absolute grid place-items-center"
              style={{ left: p.x, top: p.y, width: NW, minWidth: NW, height: NH, padding: 4 }}
            >
              <span className="title">{l}</span>
            </div>
          );
        })}
        {rows.map(({ key, fp, est, label }) => {
          if (!label) return null;
          const hot = !!finding && !!fp && isDeviating(fp, qinfo);
          return (
            <div
              key={key}
              className={`edge-label ${hot ? "badge-twist" : ""}`}
              style={{ left: label.x, top: label.y, transform: "translate(-50%,-50%)", cursor: "default" }}
              data-testid={`diagram-edge-${key}`}
              data-changed={hot ? "true" : "false"}
              data-delta={fp ? fp.diff.toFixed(4) : ""}
            >
              {finding && fp ? (
                <>
                  <CoefTip r={r} path={key} rank={finding.rank} value={fp.b_g} kind="subgroup b" />
                  <span className="text-faint">vs</span>
                  <CoefTip
                    r={r}
                    path={key}
                    rank={finding.rank}
                    value={fp.b_b}
                    kind={`${r.spec.method.emm.baseline} b`}
                  />
                  <span className="num text-[10.5px]">Δ {fmt(fp.diff, 2)}</span>
                </>
              ) : est ? (
                <CoefTip r={r} path={key} rank={0} value={est.estimate} kind="full-sample b" />
              ) : (
                "–"
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

/** The numbers behind the purple arrows: both estimates with their intervals and the deviation verdict. */
function PathIntervals({ r, finding }: { r: RunResult; finding: Finding }) {
  const { quality, interval_level: level, baseline } = r.spec.method.emm;
  const qinfo = useQualityInfo(quality);
  const ci = (b: number, [lo, hi]: [number, number]) => `${fmt(b)} [${fmt(lo)}, ${fmt(hi)}]`;
  return (
    <div className="mt-3 overflow-x-auto" data-testid="subgroup-paths">
      <table className="data">
        <thead>
          <tr>
            <th>path</th>
            <th className="r">subgroup b [{Math.round(level * 100)}% interval]</th>
            <th className="r">
              {baseline} b [{Math.round(level * 100)}% interval]
            </th>
            <th className="r">Δb</th>
            <th className="r">z</th>
            <th>intervals overlap</th>
            <th>deviating</th>
          </tr>
        </thead>
        <tbody>
          {finding.paths.map((p) => {
            const dev = isDeviating(p, qinfo);
            return (
              <tr key={p.path} data-deviating={dev ? "true" : "false"}>
                <td>{p.path}</td>
                <td className="r num">{ci(p.b_g, p.ci_g)}</td>
                <td className="r num">{ci(p.b_b, p.ci_b)}</td>
                <td className="r num">{fmt(p.diff)}</td>
                <td className="r num">{fmt(p.z, 1)}</td>
                <td>{p.disjoint ? "no" : "yes"}</td>
                <td>{dev ? <span className="badge badge-twist">deviating</span> : "–"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="mt-1 mb-0 text-[11.5px] text-faint">
        Deviating = {deviatingRule(qinfo, level)}. Wald intervals with{" "}
        {intervalSeLabel(r.spec.method.emm, r.spec.method.analysis)}.
      </p>
    </div>
  );
}

// ----------------------------------------------------------------------------- scorecard

function Scorecard({ r }: { r: RunResult }) {
  const g = r.grade;
  return (
    <section aria-label="Scorecard" data-testid="scorecard" className="mb-4">
      <div className="mb-2 flex items-center gap-2">
        <h2 className="m-0 font-semibold text-[15px]">
          <Term term="Scorecard" />
        </h2>
        <span className="badge badge-warn" data-testid="one-draw">
          {r.manifest.label}
        </span>
      </div>
      <div className="grid grid-cols-1 items-start gap-3 lg:grid-cols-3">
        <div className="card p-3" data-testid="hits">
          <div className="mb-1 font-semibold text-good">Hits · {g.hits.length}</div>
          {g.hits.length ? (
            g.hits.map((h, i) => (
              <div key={i} className="text-[12.5px]">
                <b>{h.twist}</b> found as <code>{h.description}</code> (rank {h.rank}) · Δb {fmt(h.est_diff)}{" "}
                vs true {fmt(h.true_diff)} · Jaccard {fmt(h.jaccard, 2)}
                {h.significant ? <span className="badge badge-good ml-1">above null threshold</span> : null}
              </div>
            ))
          ) : (
            <Empty>None.</Empty>
          )}
        </div>
        <div className="card p-3" data-testid="misses">
          <div className="mb-1 font-semibold text-bad">Misses · {g.misses.length}</div>
          {g.misses.length ? (
            g.misses.map((m, i) => (
              <div key={i} className="text-[12.5px]">
                <b>{m.twist}</b> (true Δb {fmt(m.true_diff)}
                {m.n_g ? `, ${m.n_g} students` : ""})
              </div>
            ))
          ) : (
            <Empty>None.</Empty>
          )}
        </div>
        <div className="card p-3" data-testid="false-positives">
          <div className="mb-1 font-semibold">False positives by class · {g.false_positives.length}</div>
          <table className="data">
            <tbody>
              {FP_CLASSES.map((c) => (
                <tr key={c} data-fp-class={c}>
                  <td>{c.replace(/_/g, " ")}</td>
                  <td className="r num">{g.fp_by_class[c] ?? 0}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {g.false_positives.length ? (
            <ul className="m-0 mt-1.5 list-none p-0 text-[12px]">
              {g.false_positives.map((f, i) => (
                <li key={i}>
                  <span className="badge badge-bad mr-1">{f.class}</span>
                  <code>{f.description}</code> · rank {f.rank}
                </li>
              ))}
            </ul>
          ) : null}
        </div>
      </div>
    </section>
  );
}

// ----------------------------------------------------------------------------- view

/** The World's own coefficient on a model path, e.g. "0.25 + 0.15·[income=low] (average 0.292)". */
function plantedText(r: RunResult, path: string): string {
  const [from, to] = parseArrowKey(path);
  const v = r.generating?.equations.direct[to]?.vars.find((x) => x.name === from);
  const avg = r.truths[path]?.t1_direct_avg;
  if (!v) return avg === undefined || avg === null ? "–" : `0 (not an arrow of the World)`;
  const expr = v.poly
    .map((t, i) => {
      const term = [formatG(Math.abs(t.coef), 3), ...t.when.map(([a, x]) => `[${a}=${x}]`)].join("·");
      return i === 0 ? (t.coef < 0 ? `−${term}` : term) : `${t.coef < 0 ? "−" : "+"} ${term}`;
    })
    .join(" ");
  return v.poly.length > 1 && avg !== undefined && avg !== null ? `${expr} (average ${fmt(avg, 3)})` : expr;
}

/** Did the search reach quality 0 within the list it kept? If not, positive subgroups may have been cut off. */
function SearchCoverageLine({ r }: { r: RunResult }) {
  const c = r.emm.coverage;
  if (!c) return null;
  return (
    <p className="mt-1 mb-0 text-[12px] text-muted" data-testid="search-coverage">
      Scored {c.scored} candidate subgroups: {c.positive} above 0, {c.zero} at 0
      {c.too_small ? `, ${c.too_small} too small to score` : ""}.{" "}
      {c.truncated ? (
        <span className="badge badge-warn" data-testid="coverage-truncated">
          the {c.kept_limit} kept by the search all scored above 0: more may exist, increase Top k
        </span>
      ) : (
        <span>
          The search kept {c.kept_positive} subgroups above 0, with room for {c.kept_limit}, so none was cut
          off.
        </span>
      )}
    </p>
  );
}

export function ResultsView() {
  const r = useUI((s) => s.run);
  const running = useUI((s) => s.running);
  const runIssues = useUI((s) => s.runIssues);
  const rank = useUI((s) => s.selectedFinding);
  const set = useUI((s) => s.set);
  // The run is stale when the World's spec hash (server-computed, same function as the run manifest) differs.
  const truth = useTruth().data;
  const [busy, setBusy] = useState(false);
  const qinfo = useQualityInfo(r?.spec.method.emm.quality ?? "");
  if (!r) {
    return (
      <div className="relative grid h-full place-items-center p-6">
        <div className="max-w-md text-center">
          <h1 className="font-semibold text-[16px]">No run yet</h1>
          <p className="text-muted">
            Press <b>Run</b> to generate one <Term term="Dataset" /> from the World with one{" "}
            <Term term="Seed" />, fit the Researcher's Model, search for subgroups and grade them against the
            truth.
          </p>
          <button type="button" className="btn btn-primary" onClick={runOnce} disabled={running}>
            {running ? "Running…" : "Run"}
          </button>
        </div>
        <IssuesPanel issues={runIssues} title="Run problems" />
      </div>
    );
  }
  const findings = r.emm.findings;
  const finding = rank > 0 ? (findings.find((f) => f.rank === rank) ?? null) : null;
  const stale = !!truth && (!truth.ok || (!!truth.hash && truth.hash !== r.manifest.world));
  const children = [...new Set(r.spec.model.arrows.map((k) => parseArrowKey(k)[1]))];
  return (
    <div className="scroll-y relative h-full" data-testid="results">
      <div className="mx-auto max-w-[1280px] p-4">
        <header className="mb-3 flex items-start justify-between gap-4">
          <div className="min-w-0">
            <h1 className="m-0 font-semibold text-[16px]">
              <Term term="Analysis Run" />: {r.spec.world.name} · {r.manifest.method_preset}
            </h1>
            <div className="mt-0.5 flex flex-wrap items-center gap-x-2 gap-y-1 text-[12px] text-muted">
              <span>
                seed {JSON.stringify(r.manifest.seed_path)} · {r.n} students in {r.n_schools} schools · truth{" "}
                {r.truth_source} · <Term term="Provenance">run {r.run_key}</Term>
              </span>
              {stale ? (
                <span className="badge badge-warn" data-testid="stale-badge">
                  World changed since this run
                </span>
              ) : null}
            </div>
          </div>
          <div className="flex shrink-0 items-center gap-2">
            <a
              className="btn btn-sm"
              href={api.runReportUrl(r.run_key)}
              target="_blank"
              rel="noreferrer"
              data-testid="open-run-report"
            >
              Open report
            </a>
            <button
              type="button"
              className="btn btn-sm"
              onClick={() =>
                downloadReport(api.runReportUrl(r.run_key), `run-${r.spec.world.name}-${r.run_key}.html`)
              }
            >
              Download report (.html)
            </button>
            <button
              type="button"
              className="btn btn-sm btn-primary"
              disabled={busy}
              onClick={async () => {
                setBusy(true);
                try {
                  await repeat200();
                } finally {
                  setBusy(false);
                }
              }}
            >
              Repeat ×200
            </button>
          </div>
        </header>
        <Scorecard r={r} />
        <div className="flex flex-col gap-4">
          <section className="card min-w-0 p-3" aria-label="Findings">
            <h2 className="mt-0 mb-2 font-semibold text-[14px]">
              Findings · <Term term="Subgroup">subgroups</Term> ranked by{" "}
              <Term term="Quality Function">quality</Term>
            </h2>
            <p className="mt-0 mb-2 text-[12px] text-muted" data-testid="quality-rule">
              Scored by <code>{r.spec.method.emm.quality}</code>: {qinfo?.description ?? ""}. A path{" "}
              <b>deviates</b> when {deviatingRule(qinfo, r.spec.method.emm.interval_level)}; deviating paths
              are purple in the diagram below.
            </p>
            <table className="data" data-testid="findings">
              <thead>
                <tr>
                  <th>#</th>
                  <th>subgroup</th>
                  <th className="r">quality</th>
                  <th>scored by</th>
                  <th className="r">deviating paths</th>
                  <th className="r">n</th>
                  <th className="r">schools</th>
                  <th>clusters</th>
                  <th>best path</th>
                  <th className="r">Δb</th>
                </tr>
              </thead>
              <tbody>
                <tr aria-selected={rank === 0}>
                  <td />
                  <td colSpan={9}>
                    <button type="button" className="link" onClick={() => set({ selectedFinding: 0 })}>
                      Full sample
                    </button>
                  </td>
                </tr>
                {findings.map((f, i) => {
                  const best = f.paths.find((p) => p.path === f.best_path);
                  return (
                    <tr key={f.rank} aria-selected={rank === f.rank} data-testid={`finding-${f.rank}`}>
                      <td className="num">{f.rank}</td>
                      <td>
                        <button
                          type="button"
                          className="link"
                          onClick={() => set({ selectedFinding: f.rank })}
                        >
                          {f.description}
                        </button>
                      </td>
                      <td className="r num">{fmt(f.quality, 2)}</td>
                      <td className="text-[12px]">
                        <code>{r.spec.method.emm.quality}</code>
                      </td>
                      <td className="r num" data-testid={`deviating-${f.rank}`}>
                        {f.paths.filter((p) => isDeviating(p, qinfo)).length} / {f.paths.length}
                      </td>
                      <td className="r num">{f.n}</td>
                      <td className="r num">{f.n_schools}</td>
                      <td>{<TierBadge tier={r.flags[i]?.tier} />}</td>
                      <td className="text-[12px]">{f.best_path}</td>
                      <td className="r">
                        {best ? (
                          <CoefTip r={r} path={f.best_path} rank={f.rank} value={best.diff} kind="Δb" />
                        ) : (
                          "–"
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            {!findings.length ? <Empty>No subgroup passed the size limits.</Empty> : null}
            <TierLegend />
            <SearchCoverageLine r={r} />
          </section>
          {finding ? <ContrastCard r={r} finding={finding} /> : null}
          <section className="card min-w-0 p-3" aria-label="Model per subgroup">
            <h2 className="mt-0 mb-2 font-semibold text-[14px]">
              {finding ? (
                <>
                  Model in <code>{finding.description}</code> vs {r.spec.method.emm.baseline}
                </>
              ) : (
                "Model in the full sample"
              )}
            </h2>
            <SubgroupDiagram r={r} finding={finding} />
            {finding ? <PathIntervals r={r} finding={finding} /> : null}
            {finding ? (
              <div className="mt-3 flex flex-col gap-1" data-testid="subgroup-equations">
                {children.map((c) => {
                  const g = subgroupEquation(c, finding.paths, "b_g");
                  const b = subgroupEquation(c, finding.paths, "b_b");
                  return g && b ? (
                    <div key={c} className="grid grid-cols-[90px_1fr] items-center gap-x-2 text-[12px]">
                      <span className="text-muted">subgroup</span>
                      <Equation eq={g} testId={`subgroup-eq-${c}`} />
                      <span className="text-muted">{r.spec.method.emm.baseline}</span>
                      <Equation eq={b} />
                    </div>
                  ) : null;
                })}
              </div>
            ) : null}
          </section>
        </div>
        <div className="mt-4">
          <EquationsCard r={r} finding={finding} />
        </div>
        <ContrastPlotCard key={`${r.run_key}:${rank}`} r={r} finding={finding} />
        <section className="card mt-4 p-3" aria-label="Full-sample paths">
          <h2 className="mt-0 mb-2 font-semibold text-[14px]">
            Full-sample paths vs <Term term="T3 Step estimand">step estimand</Term> ·{" "}
            <span className="font-normal text-muted">{r.estimand_label}</span>
          </h2>
          <table className="data" data-testid="estimation">
            <thead>
              <tr>
                <th>path</th>
                <th>planted (World)</th>
                <th className="r">estimate</th>
                <th className="r">SE (cluster)</th>
                <th className="r">SE (naive)</th>
                <th className="r">T3</th>
                <th className="r">error</th>
                <th>covered</th>
              </tr>
            </thead>
            <tbody>
              {r.estimation.map((e) => (
                <tr key={e.path}>
                  <td>{e.path}</td>
                  <td className="text-[12px]" data-testid={`planted-${e.path}`}>
                    {plantedText(r, e.path)}
                  </td>
                  <td className="r">
                    <CoefTip r={r} path={e.path} rank={0} value={e.estimate} kind="estimate" />
                  </td>
                  <td className="r num">{fmt(e.se)}</td>
                  <td className="r num">{fmt(e.se_naive)}</td>
                  <td className="r">
                    <CoefTip r={r} path={e.path} rank={0} value={e.t3} kind="T3" />
                  </td>
                  <td className="r num">{fmt(e.error)}</td>
                  <td>
                    {e.covered ? (
                      <span className="badge badge-good">yes</span>
                    ) : (
                      <span className="badge badge-bad">no</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-2 mb-0 text-[11.5px] text-faint">
            <Term term="T1 Structural truth">T1</Term> · <Term term="T2 Causal target">T2</Term> ·{" "}
            <Term term="T3 Step estimand">T3</Term> · <Term term="Common metric" /> ·{" "}
            <Term term="Bootstrap" /> · <Term term="Report" />
          </p>
        </section>
      </div>
      <IssuesPanel issues={runIssues} title="Run problems" />
    </div>
  );
}
