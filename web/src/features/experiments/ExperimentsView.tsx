import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { api, downloadReport } from "../../api/client";
import type { Aggregate, Factor, Job, JobStatus, Rate } from "../../api/types";
import { Plot, PlotFigure } from "../../components/PlotFigure";
import { Checkbox, Empty, NumberField, Section, SelectField, Term, TextField } from "../../components/ui";
import { useDoc } from "../../state/doc";
import { useUI } from "../../state/ui";
import { fmt, parseNumber } from "../../state/worldOps";
import { ensureModel, experimentSpec } from "../commands";

export function pm(r: Rate | undefined | null, d = 3): string {
  if (!r || r.value === null || r.value === undefined) return "–";
  return r.mcse === null || r.mcse === undefined ? fmt(r.value, d) : `${fmt(r.value, d)} ± ${fmt(r.mcse, d)}`;
}

const TERMINAL: JobStatus[] = ["done", "cancelled", "error"];

const FACTOR_PATHS = [
  { value: "world.schools.n_schools", label: "number of schools" },
  { value: "world.schools.students_per_school.n", label: "students per school" },
  { value: "world.twists.0.delta", label: "first twist's Δ" },
  { value: "search.depth", label: "search depth" },
  { value: "search.min_n", label: "min subgroup size" },
  { value: "search.min_schools", label: "min schools" },
  { value: "method.emm.quality", label: "quality function" },
  { value: "method.emm.baseline", label: "comparison baseline" },
  { value: "seed", label: "seed" },
];

function parseLevels(text: string): unknown[] {
  return text
    .split(",")
    .map((x) => x.trim())
    .filter(Boolean)
    .map((x) => {
      const n = parseNumber(x);
      return n === null ? x : n;
    });
}

// ----------------------------------------------------------------------------- launcher

function Launcher() {
  const world = useDoc((s) => s.doc.world);
  const methodName = useDoc((s) => s.doc.method.name);
  const qc = useQueryClient();
  const [name, setName] = useState(`${world.name} experiment`);
  const [presets, setPresets] = useState<("current" | "recommended")[]>([
    methodName === "current" ? "current" : "recommended",
  ]);
  const [factors, setFactors] = useState<{ path: string; levels: string }[]>([]);
  const [nSim, setNSim] = useState(50);
  const [withNull, setWithNull] = useState(true);
  const [workers, setWorkers] = useState<number | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const togglePreset = (p: "current" | "recommended", on: boolean) =>
    setPresets((cur) => (on ? [...new Set([...cur, p])] : cur.filter((x) => x !== p)));
  const start = async () => {
    if (!presets.length) return setErr("choose at least one method preset");
    const fs: Factor[] = factors
      .filter((f) => f.path.trim())
      .map((f) => ({ path: f.path.trim(), levels: parseLevels(f.levels) }));
    if (fs.some((f) => !f.levels.length)) return setErr("every factor needs at least one level");
    setErr(null);
    setBusy(true);
    try {
      await ensureModel();
      const spec = experimentSpec({ name, presets, factors: fs, n_sim: nSim, include_null_world: withNull });
      const r = await api.createExperiment(spec, workers ?? undefined);
      useUI.getState().set({ experimentId: r.id });
      await qc.invalidateQueries({ queryKey: ["experiments"] });
    } catch (e) {
      setErr(String((e as Error).message));
    } finally {
      setBusy(false);
    }
  };
  return (
    <Section
      title={
        <>
          New <Term term="Experiment" />
        </>
      }
      testId="launcher"
    >
      <div className="flex flex-col gap-2">
        <TextField label="Experiment name" value={name} onCommit={setName} />
        <fieldset className="m-0 border-0 p-0">
          <legend className="label mb-1">
            <Term term="Method preset">Method presets</Term>
          </legend>
          <div className="flex gap-3">
            <Checkbox
              label="recommended"
              checked={presets.includes("recommended")}
              onChange={(v) => togglePreset("recommended", v)}
            />
            <Checkbox
              label="current"
              checked={presets.includes("current")}
              onChange={(v) => togglePreset("current", v)}
            />
          </div>
        </fieldset>
        <div className="flex flex-col gap-2">
          {factors.map((f, i) => (
            <div key={i} className="rounded-md border border-line p-2" data-testid={`factor-${i}`}>
              <div className="grid grid-cols-[1fr_auto] items-end gap-1.5">
                <SelectField
                  label={`Factor ${i + 1}`}
                  value={FACTOR_PATHS.some((p) => p.value === f.path) ? f.path : "custom"}
                  options={[...FACTOR_PATHS, { value: "custom", label: "custom path…" }]}
                  onChange={(v) =>
                    setFactors((fs) =>
                      fs.map((x, j) => (j === i ? { ...x, path: v === "custom" ? "" : v } : x)),
                    )
                  }
                />
                <button
                  type="button"
                  className="btn btn-ghost btn-sm"
                  aria-label={`Remove factor ${i + 1}`}
                  onClick={() => setFactors((fs) => fs.filter((_, j) => j !== i))}
                >
                  ×
                </button>
              </div>
              {!FACTOR_PATHS.some((p) => p.value === f.path) ? (
                <div className="mt-1.5">
                  <TextField
                    label="Dotted path into the Run spec"
                    value={f.path}
                    placeholder="world.nodes.selfeff.recipe.noise_sd"
                    onCommit={(v) => setFactors((fs) => fs.map((x, j) => (j === i ? { ...x, path: v } : x)))}
                  />
                </div>
              ) : null}
              <div className="mt-1.5">
                <TextField
                  label={`Levels of factor ${i + 1} (comma-separated)`}
                  value={f.levels}
                  placeholder="100, 200, 400"
                  onCommit={(v) => setFactors((fs) => fs.map((x, j) => (j === i ? { ...x, levels: v } : x)))}
                />
              </div>
            </div>
          ))}
          <button
            type="button"
            className="btn btn-sm self-start"
            onClick={() => setFactors((fs) => [...fs, { path: FACTOR_PATHS[0].value, levels: "" }])}
          >
            + {factors.length ? "Another factor" : "Factor (sweep)"}
          </button>
        </div>
        <div className="grid grid-cols-2 gap-2">
          <NumberField
            label="Replications (n_sim)"
            value={nSim}
            min={1}
            onCommit={(v) => setNSim(Math.round(v ?? 50))}
          />
          <NumberField
            label="Workers (blank = auto)"
            value={workers}
            allowEmpty
            min={1}
            onCommit={(v) => setWorkers(v === null ? null : Math.round(v))}
          />
        </div>
        <Checkbox label="Run the matching null World" checked={withNull} onChange={setWithNull}>
          Run the matching <Term term="Null World">null World</Term> (FWER)
        </Checkbox>
        {err ? (
          <div role="alert" className="text-[12px] text-bad">
            {err}
          </div>
        ) : null}
        <button type="button" className="btn btn-primary self-start" disabled={busy} onClick={start}>
          {busy ? "Starting…" : "Start experiment"}
        </button>
      </div>
    </Section>
  );
}

// ----------------------------------------------------------------------------- live job

function useJobProgress(id: string, onProgress: () => void, restart: number) {
  const [progress, setProgress] = useState<{ status: JobStatus; done: number; total: number } | null>(null);
  const [polling, setPolling] = useState(false);
  const cb = useRef(onProgress);
  cb.current = onProgress;
  // biome-ignore lint/correctness/useExhaustiveDependencies: restart re-opens the stream after resume
  useEffect(() => {
    setPolling(false);
    if (typeof EventSource === "undefined") {
      setPolling(true);
      return;
    }
    let errors = 0;
    const es = new EventSource(api.experimentEventsUrl(id));
    es.onmessage = (e) => {
      errors = 0;
      const p = JSON.parse(e.data);
      setProgress(p);
      cb.current();
      if (TERMINAL.includes(p.status)) es.close();
    };
    es.onerror = () => {
      errors += 1;
      if (errors > 3) {
        es.close();
        setPolling(true);
      }
    };
    return () => es.close();
  }, [id, restart]);
  return { progress, polling };
}

function SweepPlot({ agg }: { agg: Aggregate }) {
  const f0 = agg.factors[0];
  if (!f0 || agg.conditions.length < 2) return null;
  const rows = agg.conditions.flatMap((c) =>
    Object.entries(c.presets).flatMap(([p, d]) =>
      Object.entries(d.twists).map(([t, v]) => {
        const x = c.factors[f0.path];
        const h = v.hit_rate.value ?? Number.NaN;
        const m = v.hit_rate.mcse ?? 0;
        return { x, series: `${p} · ${t}`, h, lo: Math.max(0, h - 1.96 * m), hi: Math.min(1, h + 1.96 * m) };
      }),
    ),
  );
  if (!rows.length) return null;
  const numeric = rows.every((r) => typeof r.x === "number");
  return (
    <div data-testid="sweep-plot">
      <h3 className="mt-3 mb-1 font-semibold text-[13px]">Hit rate across {f0.path} (bands: ±1.96 MCSE)</h3>
      <PlotFigure
        label={`Hit rate against ${f0.path} with MCSE bands`}
        deps={[JSON.stringify(rows)]}
        make={() => ({
          width: 620,
          height: 240,
          marginLeft: 44,
          grid: true,
          x: numeric ? { label: f0.path, type: "point" } : { label: f0.path, type: "point" },
          y: { label: "hit rate", domain: [0, 1] },
          color: { legend: true },
          marks: [
            Plot.areaY(rows, { x: "x", y1: "lo", y2: "hi", fill: "series", fillOpacity: 0.18 }),
            Plot.line(rows, { x: "x", y: "h", stroke: "series", strokeWidth: 1.8 }),
            Plot.dot(rows, { x: "x", y: "h", fill: "series", r: 3 }),
            Plot.ruleY([0.8], { stroke: "var(--muted)", strokeDasharray: "3 3" }),
          ],
        })}
      />
    </div>
  );
}

function ResultsTable({ agg }: { agg: Aggregate }) {
  const first = agg.conditions[0];
  const fp = first ? Object.keys(first.presets)[0] : undefined;
  const ft = fp ? Object.keys(first.presets[fp].twists)[0] : undefined;
  const headline = fp && ft ? first.presets[fp].twists[ft].hit_rate : undefined;
  return (
    <div>
      {headline ? (
        <div className="mb-3 flex flex-wrap items-baseline gap-3">
          <div className="card px-4 py-2.5">
            <div className="text-[11.5px] text-muted">
              hit rate · {ft} · {fp}
            </div>
            <div className="num font-semibold text-[22px]" data-testid="headline-hit-rate">
              {pm(headline)}
            </div>
            <div className="text-[11px] text-faint">
              ± <Term term="MCSE" /> over {headline.n} replications
            </div>
          </div>
          <div className="card px-4 py-2.5">
            <div className="text-[11.5px] text-muted">null-World FWER · {fp}</div>
            <div className="num font-semibold text-[22px]" data-testid="headline-fwer">
              {pm(first.presets[fp!].fwer_null)}
            </div>
            <div className="text-[11px] text-faint">from {first.presets[fp!].n_null} null replications</div>
          </div>
        </div>
      ) : null}
      <div className="overflow-x-auto">
        <table className="data" data-testid="experiment-table">
          <thead>
            <tr>
              <th>condition</th>
              <th>preset</th>
              <th>twist</th>
              <th className="r">hit rate</th>
              <th className="r">hit rate (sig.)</th>
              <th className="r">mean rank</th>
              <th className="r">recovered</th>
              <th className="r">recovered without (b, c)</th>
              <th className="r">FWER (null)</th>
              <th className="r">false-discovery share</th>
              <th className="r">decoy rate</th>
              <th>stability</th>
            </tr>
          </thead>
          <tbody>
            {agg.conditions.flatMap((c) =>
              Object.entries(c.presets).flatMap(([p, d]) => {
                const tw = Object.entries(d.twists);
                const rows = tw.length ? tw : [["(no twist)", null] as const];
                return rows.map(([t, v]) => (
                  <tr key={`${c.index}-${p}-${t}`}>
                    <td className="text-[12px]">
                      {Object.entries(c.factors)
                        .map(([k, x]) => `${k.split(".").slice(-2).join(".")}=${x}`)
                        .join(", ") || "base"}
                    </td>
                    <td>{p}</td>
                    <td>{t}</td>
                    <td className="r num" data-testid="hit-rate">
                      {v ? pm(v.hit_rate) : "–"}
                    </td>
                    <td className="r num">{v ? pm(v.hit_rate_significant) : "–"}</td>
                    <td className="r num">{v ? pm(v.mean_rank, 2) : "–"}</td>
                    <td className="r num" data-testid="recovery-rate">
                      {v ? pm(v.recovery_rate) : "–"}
                    </td>
                    <td className="r num" data-testid="recovery-rate-no-prune">
                      {v ? pm(v.recovery_rate_no_prune) : "–"}
                    </td>
                    <td className="r num">{pm(d.fwer_null)}</td>
                    <td className="r num">{pm(d.fdp.all)}</td>
                    <td className="r num">{pm(d.decoy_rate)}</td>
                    <td>
                      {v?.unstable ? (
                        <span className="badge badge-warn">
                          <Term term="Unstable">unstable</Term>
                        </span>
                      ) : v ? (
                        <span className="badge badge-good">stable</span>
                      ) : null}
                    </td>
                  </tr>
                ));
              }),
            )}
          </tbody>
        </table>
      </div>
      <SweepPlot agg={agg} />
    </div>
  );
}

function JobDetail({ id }: { id: string }) {
  const qc = useQueryClient();
  const [restart, setRestart] = useState(0);
  const lastRefetch = useRef(0);
  const job = useQuery({ queryKey: ["experiment", id], queryFn: () => api.experiment(id) });
  const refresh = () => {
    const now = Date.now();
    if (now - lastRefetch.current < 900) return;
    lastRefetch.current = now;
    void qc.invalidateQueries({ queryKey: ["experiment", id] });
    void qc.invalidateQueries({ queryKey: ["experiments"] });
  };
  const { progress, polling } = useJobProgress(id, refresh, restart);
  const j: Job | undefined = job.data;
  const status = progress?.status ?? j?.status;
  const done = progress?.done ?? j?.done ?? 0;
  const total = progress?.total ?? j?.total ?? 0;
  const live = status === "running" || status === "queued";
  // Polling fallback (no EventSource), and a final refresh when the stream reports a terminal state.
  useEffect(() => {
    if (!polling || !live) return;
    const t = setInterval(refresh, 1000);
    return () => clearInterval(t);
  });
  // biome-ignore lint/correctness/useExhaustiveDependencies: refresh on terminal transitions only
  useEffect(() => {
    if (status && TERMINAL.includes(status)) {
      lastRefetch.current = 0;
      refresh();
    }
  }, [status]);
  if (!j) return <Empty>Loading experiment…</Empty>;
  const pct = total ? Math.round((done / total) * 100) : 0;
  return (
    <section className="min-w-0" aria-label={`Experiment ${j.name}`} data-testid="job-detail">
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <h2 className="m-0 font-semibold text-[15px]">{j.name}</h2>
        <span
          className={`badge ${status === "done" ? "badge-good" : status === "error" ? "badge-bad" : status === "cancelled" ? "badge-warn" : "badge-accent"}`}
          data-testid="job-status"
        >
          {status}
        </span>
        <span className="text-[12px] text-muted">
          {j.spec.n_sim} replications · {j.spec.presets.join(" + ")} ·{" "}
          {j.spec.factors.length ? `${j.spec.factors.length} factor(s)` : "no factors"}
          {polling ? " · polling" : " · live (SSE)"}
        </span>
        <span className="flex-1" />
        {live ? (
          <button
            type="button"
            className="btn btn-sm"
            onClick={async () => {
              await api.cancelExperiment(id);
              refresh();
            }}
          >
            Cancel
          </button>
        ) : null}
        {status === "cancelled" || status === "error" || (status === "done" && done < total) ? (
          <button
            type="button"
            className="btn btn-sm"
            onClick={async () => {
              await api.resumeExperiment(id);
              setRestart((r) => r + 1);
              lastRefetch.current = 0;
              refresh();
            }}
          >
            Resume
          </button>
        ) : null}
        {j.aggregate ? (
          <>
            <a
              className="btn btn-sm"
              href={api.experimentReportUrl(id)}
              target="_blank"
              rel="noreferrer"
              data-testid="open-experiment-report"
            >
              Open report
            </a>
            <button
              type="button"
              className="btn btn-sm"
              onClick={() => downloadReport(api.experimentReportUrl(id), `experiment-${j.name}-${id}.html`)}
            >
              Download report (.html)
            </button>
          </>
        ) : null}
      </div>
      <div className="mb-1 flex items-center gap-2 text-[12px] text-muted">
        <div
          className="progress flex-1"
          role="progressbar"
          aria-label="Experiment progress"
          aria-valuemin={0}
          aria-valuemax={total}
          aria-valuenow={done}
          data-testid="progress"
        >
          <div style={{ width: `${pct}%` }} />
        </div>
        <span className="num" data-testid="progress-text">
          {done} / {total} units
        </span>
      </div>
      {j.error ? <p className="text-bad text-[12px]">{j.error}</p> : null}
      <p className="mt-0 mb-3 text-[11.5px] text-faint">
        Each unit is one replication of one condition × preset × (planted | null World). Rates update as units
        finish.
      </p>
      {j.aggregate ? <ResultsTable agg={j.aggregate} /> : <Empty>Waiting for the first replications…</Empty>}
    </section>
  );
}

// ----------------------------------------------------------------------------- view

export function ExperimentsView() {
  const id = useUI((s) => s.experimentId);
  const set = useUI((s) => s.set);
  const list = useQuery({ queryKey: ["experiments"], queryFn: api.experiments });
  return (
    <div className="grid h-full min-h-0" style={{ gridTemplateColumns: "340px minmax(0,1fr)" }}>
      <aside className="scroll-y h-full border-line border-r bg-panel" aria-label="Experiments">
        <Launcher />
        <Section title="Experiments">
          {list.data?.length ? (
            <ul className="m-0 flex list-none flex-col gap-1 p-0" data-testid="job-list">
              {list.data.map((j) => (
                <li key={j.id}>
                  <button
                    type="button"
                    aria-current={id === j.id ? "true" : undefined}
                    className={`flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-[12.5px] ${id === j.id ? "bg-accent-soft" : "hover:bg-panel-2"}`}
                    onClick={() => set({ experimentId: j.id })}
                  >
                    <span className="min-w-0 flex-1 truncate">{j.name}</span>
                    <span className="num text-muted">
                      {j.done}/{j.total}
                    </span>
                    <span className="badge">{j.status}</span>
                  </button>
                </li>
              ))}
            </ul>
          ) : (
            <Empty>No experiments yet.</Empty>
          )}
        </Section>
      </aside>
      <div className="scroll-y h-full min-w-0 p-4">
        {id ? (
          <JobDetail key={id} id={id} />
        ) : (
          <div className="mx-auto mt-16 max-w-md text-center text-muted">
            <h1 className="font-semibold text-[16px] text-fg">
              <Term term="Experiment" />s
            </h1>
            An Experiment replicates the Run many times (and on the <Term term="Null World" />) so every rate
            comes with its <Term term="MCSE" />. Start one on the left, or use <b>Repeat ×200</b>.
          </div>
        )}
      </div>
    </div>
  );
}
