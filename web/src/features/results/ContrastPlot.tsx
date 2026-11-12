import { useQuery } from "@tanstack/react-query";
import { useRef } from "react";
import { api, downloadText } from "../../api/client";
import type { ContrastCompare, ContrastFit, ContrastResult, Finding, RunResult } from "../../api/types";
import { Empty } from "../../components/ui";
import { fmt } from "../../state/worldOps";
import { deviatingRule, useQualityInfo } from "./quality";

/** Theme colours resolved to literal values, so a downloaded SVG looks the same outside the app. */
function palette() {
  const css = getComputedStyle(document.documentElement);
  const v = (name: string, fallback: string) => css.getPropertyValue(name).trim() || fallback;
  return {
    bg: v("--panel", "#1f1f1f"),
    fg: v("--fg", "#e8e8e8"),
    muted: v("--muted", "#9a9a9a"),
    line: v("--line-strong", "#555555"),
    hot: v("--twist", "#c77dff"),
  };
}

const W = 1100;
const GRID_W = 560;
const CELL_W = 170;
const CELL_H = 84;

type Cell = { key: string; label: string; fit: ContrastFit | null | undefined; n: number; finding: boolean };

function cmpLabel(c: ContrastCompare | undefined) {
  return c && c.diff !== null ? `Δ ${fmt(c.diff, 3)} · z ${fmt(c.z, 1)}` : "–";
}

/** One path: the slices as a 2 × 2 grid with every comparison between them, and all slices on one axis. */
function PathPanel({
  c,
  path,
  r,
  testId,
}: {
  c: ContrastResult;
  path: string;
  r: RunResult;
  testId: string;
}) {
  const ref = useRef<SVGSVGElement>(null);
  const col = palette();
  const by = Object.fromEntries(c.slices.map((s) => [s.key, s]));
  const two = c.conditions.length === 2;
  const cells: Cell[] = (two ? ["finding", "flip:1", "flip:0", "cell:00"] : ["finding", "complement"]).map(
    (k) => ({
      key: k,
      label: by[k]?.label ?? k,
      fit: by[k]?.fits[path],
      n: by[k]?.n ?? 0,
      finding: k === "finding",
    }),
  );
  // grid positions: rows = first condition holds / not, columns = second condition holds / not
  const at: Record<string, [number, number]> = two
    ? { finding: [0, 0], "flip:1": [1, 0], "flip:0": [0, 1], "cell:00": [1, 1] }
    : { finding: [0, 0], complement: [1, 0] };
  const gx = (i: number) => 40 + i * (CELL_W + 150);
  const gy = (j: number) => 52 + j * (CELL_H + 70);
  const pair = (a: string, b: string) =>
    c.pairs.find((p) => (p.a === a && p.b === b) || (p.a === b && p.b === a))?.vs[path] ??
    (b === "complement" ? by.complement?.vs_finding?.[path] : undefined);
  const links: [string, string][] = two
    ? [
        ["finding", "flip:1"],
        ["flip:0", "cell:00"],
        ["finding", "flip:0"],
        ["flip:1", "cell:00"],
        ["finding", "cell:00"],
        ["flip:1", "flip:0"],
      ]
    : [["finding", "complement"]];
  const centre = (k: string) => [gx(at[k][0]) + CELL_W / 2, gy(at[k][1]) + CELL_H / 2] as const;
  const geom = (a: string, b: string) => {
    const v = pair(a, b);
    const [x1, y1] = centre(a);
    const [x2, y2] = centre(b);
    // the two diagonals cross in the middle: put their labels at different points along the line
    const t = a === "finding" && b === "cell:00" ? 0.38 : a === "flip:1" && b === "flip:0" ? 0.62 : 0.5;
    return { v, x1, y1, x2, y2, hot: !!v?.deviating, mx: x1 + t * (x2 - x1), my: y1 + t * (y2 - y1) };
  };

  // strip plot: every slice plus the full sample on one axis
  const full = r.estimation.find((e) => e.path === path);
  const rows = [
    ...c.slices
      .filter((s) => s.role !== "population")
      .map((s) => ({ label: s.label, fit: s.fits[path], hot: s.role === "finding" })),
    ...(full
      ? [
          {
            label: "full sample",
            fit: {
              b: full.estimate,
              se: full.se,
              ci: [full.estimate - 1.96 * full.se, full.estimate + 1.96 * full.se] as [number, number],
            },
            hot: false,
          },
        ]
      : []),
  ];
  const vals = rows.flatMap((x) => (x.fit ? x.fit.ci : []));
  const lo = Math.min(...vals);
  const hi = Math.max(...vals);
  const pad = (hi - lo) * 0.1 || 0.05;
  const sx0 = GRID_W + 210;
  const sx1 = W - 30;
  const sx = (v: number) => sx0 + ((v - (lo - pad)) / (hi - lo + 2 * pad)) * (sx1 - sx0);
  const rowH = 24;
  const gridH = two ? gy(1) + CELL_H + 20 : gy(0) + CELL_H + 40;
  const H = Math.max(gridH, 60 + rows.length * rowH) + (c.interaction[path] ? 40 : 10);
  const it = c.interaction[path];

  const download = () => {
    if (!ref.current) return;
    const xml = new XMLSerializer().serializeToString(ref.current);
    const name = `contrast-${c.description}-${path}`.replace(/[^a-z0-9=_.-]+/gi, "_");
    downloadText(xml, `${name}.svg`, "image/svg+xml");
  };

  return (
    <figure className="m-0 mb-4" data-testid={testId}>
      <div className="mb-1 flex items-center gap-2">
        <code className="text-[12.5px]">{path}</code>
        <span className="flex-1" />
        <button type="button" className="btn btn-sm" onClick={download}>
          Download SVG
        </button>
      </div>
      <div className="overflow-x-auto">
        <svg
          ref={ref}
          xmlns="http://www.w3.org/2000/svg"
          width={W}
          height={H}
          viewBox={`0 0 ${W} ${H}`}
          fontFamily="system-ui, sans-serif"
          role="img"
          aria-label={`Contrast of ${c.description} on ${path}`}
        >
          <rect width={W} height={H} fill={col.bg} />
          <text x={20} y={22} fill={col.fg} fontSize={13} fontWeight={600}>
            {c.description} · {path}
          </text>
          {links.map(([a, b]) => {
            const g = geom(a, b);
            return (
              <line
                key={`${a}-${b}`}
                data-pair={`${a}|${b}`}
                data-deviating={g.hot ? "true" : "false"}
                x1={g.x1}
                y1={g.y1}
                x2={g.x2}
                y2={g.y2}
                stroke={g.hot ? col.hot : col.line}
                strokeWidth={g.hot ? 2.5 : 1.2}
              />
            );
          })}
          {cells.map((cell) => {
            const [i, j] = at[cell.key];
            return (
              <g key={cell.key} data-cell={cell.key}>
                <rect
                  x={gx(i)}
                  y={gy(j)}
                  width={CELL_W}
                  height={CELL_H}
                  rx={8}
                  fill={col.bg}
                  stroke={cell.finding ? col.hot : col.line}
                  strokeWidth={cell.finding ? 2 : 1}
                />
                <text x={gx(i) + 8} y={gy(j) + 17} fontSize={10.5} fill={col.muted}>
                  {cell.label.length > 30 ? `${cell.label.slice(0, 29)}…` : cell.label}
                </text>
                <text
                  x={gx(i) + 8}
                  y={gy(j) + 42}
                  fontSize={15}
                  fontWeight={600}
                  fill={cell.finding ? col.hot : col.fg}
                >
                  {cell.fit ? fmt(cell.fit.b, 3) : "too small"}
                </text>
                <text x={gx(i) + 8} y={gy(j) + 60} fontSize={10.5} fill={col.muted}>
                  {cell.fit ? `[${fmt(cell.fit.ci[0], 3)}, ${fmt(cell.fit.ci[1], 3)}]` : ""}
                </text>
                <text x={gx(i) + 8} y={gy(j) + 76} fontSize={10.5} fill={col.muted}>
                  n {cell.n}
                </text>
              </g>
            );
          })}
          {links.map(([a, b]) => {
            const g = geom(a, b);
            return (
              <g key={`${a}-${b}`}>
                <rect
                  x={g.mx - 62}
                  y={g.my - 9}
                  width={124}
                  height={18}
                  rx={4}
                  fill={col.bg}
                  stroke={g.hot ? col.hot : col.line}
                />
                <text
                  x={g.mx}
                  y={g.my + 4}
                  textAnchor="middle"
                  fontSize={10.5}
                  fill={g.hot ? col.hot : col.muted}
                >
                  {cmpLabel(g.v)}
                </text>
              </g>
            );
          })}
          {/* strip plot */}
          <text x={sx0} y={40} fontSize={11} fill={col.muted}>
            slope of {path} with its interval
          </text>
          {rows.map((row, k) => {
            const y = 60 + k * rowH;
            return (
              <g key={row.label} data-strip={row.label}>
                <text
                  x={sx0 - 8}
                  y={y + 4}
                  textAnchor="end"
                  fontSize={10.5}
                  fill={row.hot ? col.hot : col.fg}
                >
                  {row.label.length > 34 ? `${row.label.slice(0, 33)}…` : row.label}
                </text>
                {row.fit ? (
                  <>
                    <line
                      x1={sx(row.fit.ci[0])}
                      x2={sx(row.fit.ci[1])}
                      y1={y}
                      y2={y}
                      stroke={row.hot ? col.hot : col.muted}
                      strokeWidth={2}
                    />
                    <circle cx={sx(row.fit.b)} cy={y} r={4} fill={row.hot ? col.hot : col.fg} />
                  </>
                ) : (
                  <text x={sx0} y={y + 4} fontSize={10.5} fill={col.muted}>
                    too small
                  </text>
                )}
              </g>
            );
          })}
          <line
            x1={sx0}
            x2={sx1}
            y1={60 + rows.length * rowH - 8}
            y2={60 + rows.length * rowH - 8}
            stroke={col.line}
          />
          <text x={sx0} y={60 + rows.length * rowH + 6} fontSize={10} fill={col.muted}>
            {fmt(lo - pad, 2)}
          </text>
          <text x={sx1} y={60 + rows.length * rowH + 6} fontSize={10} fill={col.muted} textAnchor="end">
            {fmt(hi + pad, 2)}
          </text>
          {it ? (
            <text
              x={20}
              y={H - 14}
              fontSize={11.5}
              fill={it.additive ? col.fg : col.hot}
              data-testid="interaction"
            >
              Interaction (A∧B − A∧¬B) − (¬A∧B − ¬A∧¬B) = {fmt(it.estimate, 3)} [{fmt(it.ci[0], 3)},{" "}
              {fmt(it.ci[1], 3)}], z {fmt(it.z, 1)}:{" "}
              {it.additive
                ? "the two conditions add up (no extra effect of the combination)"
                : "the combination has an effect of its own"}
            </text>
          ) : null}
        </svg>
      </div>
    </figure>
  );
}

/** The contrast of the selected finding, drawn for every deviating path (purple in the diagram). */
export function ContrastPlotCard({ r, finding }: { r: RunResult; finding: Finding | null }) {
  const q = useQuery({
    queryKey: ["contrast", r.run_key, finding?.rank ?? 0],
    queryFn: () => api.contrast(r.run_key, finding!.rank),
    enabled: !!finding,
  });
  const emm = r.spec.method.emm;
  const qinfo = useQualityInfo(emm.quality);
  const c = q.data;
  return (
    <section className="card mt-4 p-3" aria-label="Contrast picture" data-testid="contrast-plot">
      <h2 className="mt-0 mb-1 font-semibold text-[14px]">
        Contrast picture
        {finding ? (
          <>
            {" "}
            · <code>{finding.description}</code>
          </>
        ) : null}
      </h2>
      <p className="mt-0 mb-2 text-[12px] text-muted">
        For each deviating path: the slices of the finding as a grid (rows: first condition holds or not,
        columns: second condition), every comparison between them, purple where they differ (
        {deviatingRule(qinfo, emm.interval_level)}), and all slices on one axis with their intervals.
      </p>
      {!finding ? (
        <Empty>Select a finding above.</Empty>
      ) : q.isError ? (
        <p className="text-bad">{String((q.error as Error).message)}</p>
      ) : !c ? (
        <Empty>Computing the slices…</Empty>
      ) : !c.focal_paths.length ? (
        <Empty>No path deviates for this finding, so there is nothing to draw.</Empty>
      ) : c.conditions.length > 2 ? (
        <Empty>The grid is drawn for one or two conditions; for three, read the contrast table above.</Empty>
      ) : (
        c.focal_paths.map((p) => <PathPanel key={p} c={c} path={p} r={r} testId={`contrast-plot-${p}`} />)
      )}
    </section>
  );
}
