import { keepPreviousData, useQuery } from "@tanstack/react-query";
import {
  Background,
  BaseEdge,
  type Connection,
  Controls,
  type Edge,
  EdgeLabelRenderer,
  type EdgeProps,
  getBezierPath,
  Handle,
  MarkerType,
  type Node,
  type NodeProps,
  Position,
  ReactFlow,
  ReactFlowProvider,
  useReactFlow,
} from "@xyflow/react";
import { useEffect, useMemo, useState } from "react";
import { api } from "../../api/client";
import type { EstimandsResponse } from "../../api/types";
import { Equation, modelEquation } from "../../components/Equation";
import { DerivedValue, Empty, Section, SelectField, Term } from "../../components/ui";
import { useDoc } from "../../state/doc";
import { useTruth } from "../../state/truth";
import { useUI } from "../../state/ui";
import { fmt, modelCandidates, modelWouldCycle, parseArrowKey } from "../../state/worldOps";
import { layered } from "../world/layout";

const editModel = (label: string, fn: (d: ReturnType<typeof useDoc.getState>["doc"]) => void) =>
  useDoc.getState().apply(label, "other", (d) => fn(d as never));

export function toggleResidualize(name: string) {
  editModel(`residualize ${name}`, (d) => {
    const r = d.method.analysis.residualize_on;
    d.method.analysis.residualize_on = r.includes(name) ? r.filter((x) => x !== name) : [...r, name];
  });
}

export async function copyModelFromWorld() {
  const m = await api.modelFromWorld(useDoc.getState().doc.world);
  editModel("copy model from World", (d) => {
    d.model = m;
    d.method.analysis.residualize_on = d.method.analysis.residualize_on.filter((x) => m.latents.includes(x));
  });
}

type MNode = Node<{ label: string }, "model">;

function ModelNodeView({ id }: NodeProps<MNode>) {
  const res = useDoc((s) => s.doc.method.analysis.residualize_on.includes(id));
  return (
    <div
      className={`wnode ${res ? "is-residualized" : ""}`}
      data-testid={`model-node-${id}`}
      data-residualized={res ? "true" : "false"}
    >
      <Handle type="target" position={Position.Left} />
      <div className="title">{id}</div>
      <div className="typed text-[11px]">
        {res ? "residualized on (click to undo)" : "click to residualize on"}
      </div>
      <Handle type="source" position={Position.Right} />
    </div>
  );
}
const nodeTypes = { model: ModelNodeView };

/** Full-population T3 of the Researcher's Model; the canvas labels and the equation cards share one query. */
function useModelEstimands() {
  const { world, model, method } = useDoc((s) => s.doc);
  const { issues } = useTruth();
  return useQuery({
    queryKey: ["estimands", JSON.stringify([world, model, method.analysis])],
    queryFn: () => api.estimands(world, model, method.analysis),
    enabled: model.arrows.length > 0 && issues.length === 0,
    placeholderData: keepPreviousData,
  });
}

type EdgeCoef = { before: number; after: number };

/** Arrow key ("from -> to") to its expected coefficient without and with the chosen residualization. */
function edgeCoefs(data: EstimandsResponse | undefined): Record<string, EdgeCoef> | null {
  if (data?.source !== "analytic") return null;
  const out: Record<string, EdgeCoef> = {};
  for (const [child, e] of Object.entries(data.equations))
    e.regs.forEach((reg, i) => {
      out[`${reg} -> ${child}`] = { before: e.coef_before[i], after: e.coef[i] };
    });
  return out;
}

const CHANGED_TOL = 5e-4;

type CoefEdge = Edge<{ arrow: string; coef: EdgeCoef | null }, "coef">;

function CoefEdgeView({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  markerEnd,
  data,
}: EdgeProps<CoefEdge>) {
  const [path, lx, ly] = getBezierPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
  });
  const coef = data?.coef ?? null;
  const arrow = data?.arrow ?? id;
  const changed = !!coef && Math.abs(coef.after - coef.before) > CHANGED_TOL;
  return (
    <>
      <BaseEdge
        id={id}
        path={path}
        markerEnd={markerEnd}
        style={changed ? { stroke: "var(--warn)", strokeWidth: 2 } : undefined}
      />
      {coef ? (
        <EdgeLabelRenderer>
          <div
            className={`edge-label nodrag nopan ${changed ? "is-selected" : ""}`}
            style={{ transform: `translate(-50%, -50%) translate(${lx}px, ${ly}px)`, cursor: "default" }}
            data-testid={`model-edge-${arrow}`}
            data-changed={changed ? "true" : "false"}
          >
            {changed ? (
              <>
                <span className="num text-faint">{fmt(coef.before, 3)}</span>
                <span className="text-faint">→</span>
              </>
            ) : null}
            <DerivedValue
              value={coef.after}
              label={`Expected coefficient ${arrow}${changed ? ` (was ${fmt(coef.before, 3)} before residualization)` : ""}`}
              source="Step estimand (T3), whole population, standardized once (common metric)"
            />
          </div>
        </EdgeLabelRenderer>
      ) : null}
    </>
  );
}
const edgeTypes = { coef: CoefEdgeView };

function ModelCanvas() {
  const model = useDoc((s) => s.doc.model);
  const showCoefs = useUI((s) => s.modelCoefs);
  const estimands = useModelEstimands();
  const coefs = useMemo(() => edgeCoefs(estimands.data), [estimands.data]);
  const rf = useReactFlow();
  const [pos, setPos] = useState<Record<string, { x: number; y: number }>>({});
  const shape = JSON.stringify([model.latents, model.arrows]);
  // biome-ignore lint/correctness/useExhaustiveDependencies: re-layout when the model's shape changes
  useEffect(() => {
    let cancel = false;
    layered(
      model.latents.map((id) => ({ id, width: 200, height: 64 })),
      model.arrows.map((k) => parseArrowKey(k)).map(([from, to]) => ({ id: `${from} -> ${to}`, from, to })),
      { layerGap: 90, nodeGap: 50 },
    ).then((r) => {
      if (cancel) return;
      setPos(r.positions);
      requestAnimationFrame(() => rf.fitView({ padding: 0.25 }));
    });
    return () => {
      cancel = true;
    };
  }, [shape]);
  const [nodes, setNodes] = useState<MNode[]>([]);
  useEffect(() => {
    setNodes((prev) => {
      const old = new Map(prev.map((n) => [n.id, n]));
      return model.latents.map((id, i) => ({
        ...(old.get(id) ?? {}),
        id,
        type: "model",
        position: pos[id] ?? old.get(id)?.position ?? { x: i * 240, y: 0 },
        data: { label: id },
        ariaLabel: `Model latent ${id}: activate to toggle residualization`,
      }));
    });
  }, [model.latents, pos]);
  const edges: CoefEdge[] = model.arrows.map((k) => {
    const [s, t] = parseArrowKey(k);
    return {
      id: k,
      type: "coef",
      source: s,
      target: t,
      data: { arrow: k, coef: showCoefs ? (coefs?.[k] ?? null) : null },
      markerEnd: { type: MarkerType.ArrowClosed },
      ariaLabel: `Model arrow ${k}`,
    };
  });
  const onConnect = (c: Connection) => {
    const m = useDoc.getState().doc.model;
    if (modelWouldCycle(m, c.source, c.target) || m.arrows.includes(`${c.source} -> ${c.target}`)) return;
    editModel("model arrow", (d) => {
      d.model.arrows.push(`${c.source} -> ${c.target}`);
    });
  };
  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeTypes}
      edgeTypes={edgeTypes}
      onNodesChange={(ch) =>
        setNodes((ns) =>
          ns.map((n) => {
            const c = ch.find((x) => "id" in x && x.id === n.id);
            if (!c) return n;
            if (c.type === "position" && c.position) return { ...n, position: c.position };
            if (c.type === "dimensions" && c.dimensions) return { ...n, measured: c.dimensions };
            return n;
          }),
        )
      }
      onNodeClick={(_, n) => toggleResidualize(n.id)}
      onConnect={onConnect}
      isValidConnection={(c) => {
        const m = useDoc.getState().doc.model;
        return (
          c.source !== c.target &&
          !modelWouldCycle(m, c.source, c.target) &&
          !m.arrows.includes(`${c.source} -> ${c.target}`)
        );
      }}
      deleteKeyCode={null}
      fitView
      proOptions={{ hideAttribution: true }}
      aria-label="Researcher's Model canvas"
    >
      <Background gap={18} size={1.2} color="var(--canvas-dot)" />
      <Controls showInteractive={false} position="bottom-left" />
    </ReactFlow>
  );
}

function EstimandCards() {
  const model = useDoc((s) => s.doc.model);
  const q = useModelEstimands();
  if (!model.arrows.length) return <Empty>The model has no arrows yet.</Empty>;
  if (q.isError) return <p className="text-[12.5px] text-bad">{String((q.error as Error).message)}</p>;
  if (!q.data) return <Empty>Computing step estimands…</Empty>;
  if (q.data.source !== "analytic")
    return (
      <Empty>
        These settings have no closed form ({q.data.source}); the Run computes them by oracle simulation.
      </Empty>
    );
  return (
    <ul className="m-0 flex list-none flex-col gap-2 p-0" data-testid="model-equations">
      {Object.entries(q.data.equations).map(([child, e]) => (
        <li key={child} className="card p-2.5" data-testid={`eq-${child}`}>
          <div className="mb-1 flex items-center gap-2">
            <span className="font-semibold">{child}</span>
            {e.case ? (
              <span
                className={`badge ${e.case === "triangle" ? "badge-good" : "badge-warn"}`}
                data-testid={`case-${child}`}
              >
                {e.case}
              </span>
            ) : null}
            <span className="ml-auto text-[12px] text-muted" data-testid={`r2-${child}`}>
              R²{" "}
              <DerivedValue
                value={e.r2_before}
                label={`R² of ${child} before residualization (T3)`}
                source="Step estimand (T3), analytic"
              />{" "}
              →{" "}
              <DerivedValue
                value={e.r2}
                label={`R² of ${child} after residualization (T3)`}
                source="Step estimand (T3), analytic"
              />
            </span>
          </div>
          <Equation eq={modelEquation(child, e.regs, e.coef)} testId={`model-equation-${child}`} />
          <table className="data mt-1">
            <thead>
              <tr>
                <th>regressor</th>
                <th className="r">before</th>
                <th className="r">after</th>
              </tr>
            </thead>
            <tbody>
              {e.regs.map((r, i) => (
                <tr key={r}>
                  <td>{r}</td>
                  <td className="r">
                    <DerivedValue
                      value={e.coef_before[i]}
                      label={`T3 of ${r} → ${child} without residualization`}
                      source="Step estimand (T3), common metric"
                    />
                  </td>
                  <td className="r">
                    <DerivedValue
                      value={e.coef[i]}
                      label={`T3 of ${r} → ${child} with residualization`}
                      source="Step estimand (T3), common metric"
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {e.case === "chain" ? (
            <p className="m-0 mt-1 text-[11.5px] text-muted">
              Chain: the clicked node does not point at {child} directly, so information leaves the equation
              (R² can drop {fmt(e.r2_before - e.r2)}).
            </p>
          ) : e.case === "triangle" ? (
            <p className="m-0 mt-1 text-[11.5px] text-muted">
              Triangle: credit moves to the clicked node, R² is unchanged.
            </p>
          ) : null}
        </li>
      ))}
    </ul>
  );
}

function CoefCaption() {
  const q = useModelEstimands();
  const residualized = useDoc((s) => s.doc.method.analysis.residualize_on.length > 0);
  const analytic = q.data?.source === "analytic";
  return (
    <p
      className="pointer-events-none absolute right-3 bottom-3 left-14 m-0 max-w-[520px] rounded-md border border-line bg-panel px-2.5 py-1.5 text-[11.5px] text-muted"
      data-testid="coef-caption"
    >
      {q.isError ? (
        "Coefficients unavailable for these settings."
      ) : !q.data ? (
        "Computing coefficients…"
      ) : !analytic ? (
        "These settings have no closed form; the Run computes them by oracle simulation."
      ) : (
        <>
          Expected coefficients of this model on the whole population (standardized once, after the
          normalisation in Search &amp; method).{" "}
          {residualized
            ? "Highlighted arrows changed with the residualization: before → after."
            : "Click a node to residualize and see which arrows change."}{" "}
          The fit on generated data is in Results → Full sample.
        </>
      )}
    </p>
  );
}

export function ModelView() {
  const { world, model, method } = useDoc((s) => s.doc);
  const showCoefs = useUI((s) => s.modelCoefs);
  const setUI = useUI((s) => s.set);
  const candidates = modelCandidates(world);
  const [from, setFrom] = useState("");
  const [to, setTo] = useState("");
  const addProblem = useMemo(() => {
    if (!from || !to) return "choose both ends";
    if (from === to) return "a latent cannot cause itself";
    if (model.arrows.includes(`${from} -> ${to}`)) return "already in the model";
    if (modelWouldCycle(model, from, to)) return "would create a cycle";
    return null;
  }, [from, to, model]);
  return (
    <div className="grid h-full min-h-0" style={{ gridTemplateColumns: "minmax(0,1fr) 400px" }}>
      <section className="flex min-h-0 min-w-0 flex-col" aria-label="Researcher's Model">
        <div className="flex h-10 shrink-0 items-center gap-2 border-line border-b bg-panel px-3">
          <h1 className="m-0 font-semibold text-[14px]">
            <Term term="Researcher's Model" />
          </h1>
          <span className="text-[12px] text-muted">
            Click a node to <Term term="Residualization">residualize</Term> on it. Drag between nodes to add
            an arrow.
          </span>
          <span className="flex-1" />
          <label className="flex items-center gap-1.5 text-[12px]">
            <input
              type="checkbox"
              aria-label="Show SEM coefficients on the diagram"
              checked={showCoefs}
              onChange={(e) => setUI({ modelCoefs: e.target.checked })}
            />
            <Term term="T3 Step estimand">SEM coefficients</Term>
          </label>
          <button type="button" className="btn btn-sm btn-primary" onClick={copyModelFromWorld}>
            Copy from World
          </button>
        </div>
        <div className="relative min-h-0 flex-1" data-testid="model-canvas">
          <ReactFlowProvider>
            <ModelCanvas />
          </ReactFlowProvider>
          {showCoefs && model.arrows.length ? <CoefCaption /> : null}
          {!model.latents.length ? (
            <div className="pointer-events-none absolute inset-0 grid place-items-center text-muted">
              Empty model: use “Copy from World” to start from the World's latent part.
            </div>
          ) : null}
        </div>
      </section>
      <aside className="scroll-y h-full border-line border-l bg-panel" aria-label="Model inspector">
        <Section title="Latents in the model">
          {candidates.length ? (
            <table className="data">
              <thead>
                <tr>
                  <th>latent</th>
                  <th>in model</th>
                  <th>residualize on</th>
                </tr>
              </thead>
              <tbody>
                {candidates.map((c) => (
                  <tr key={c}>
                    <td>{c}</td>
                    <td>
                      <input
                        type="checkbox"
                        aria-label={`Include ${c} in the model`}
                        checked={model.latents.includes(c)}
                        onChange={(e) =>
                          editModel(`model latent ${c}`, (d) => {
                            if (e.target.checked) {
                              const order = candidates.filter((x) => x === c || d.model.latents.includes(x));
                              d.model.latents = order;
                            } else {
                              d.model.latents = d.model.latents.filter((x) => x !== c);
                              d.model.arrows = d.model.arrows.filter((k) => !parseArrowKey(k).includes(c));
                              d.method.analysis.residualize_on = d.method.analysis.residualize_on.filter(
                                (x) => x !== c,
                              );
                            }
                          })
                        }
                      />
                    </td>
                    <td>
                      <input
                        type="checkbox"
                        aria-label={`Residualize on ${c}`}
                        disabled={!model.latents.includes(c)}
                        checked={method.analysis.residualize_on.includes(c)}
                        onChange={() => toggleResidualize(c)}
                      />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <Empty>The World has no observed student-level latents.</Empty>
          )}
          <div className="mt-2">
            <SelectField
              label="Residualization method"
              value={method.analysis.residualize_method}
              options={[
                { value: "total_effect", label: "total effect (recommended)" },
                { value: "simple", label: "simple residual (current)" },
              ]}
              onChange={(v) =>
                editModel("residualize method", (d) => {
                  d.method.analysis.residualize_method = v;
                })
              }
            />
          </div>
        </Section>
        <Section title={<Term term="Arrow">Model arrows</Term>}>
          {model.arrows.length ? (
            <ul className="m-0 mb-2 flex list-none flex-col gap-1 p-0" data-testid="model-arrows">
              {model.arrows.map((k) => (
                <li key={k} className="flex items-center gap-2">
                  <span className="flex-1">{k.replace("->", "→")}</span>
                  <button
                    type="button"
                    className="btn btn-ghost btn-sm"
                    aria-label={`Remove model arrow ${k}`}
                    onClick={() =>
                      editModel("remove model arrow", (d) => {
                        d.model.arrows = d.model.arrows.filter((x) => x !== k);
                      })
                    }
                  >
                    ×
                  </button>
                </li>
              ))}
            </ul>
          ) : (
            <Empty>No arrows.</Empty>
          )}
          <div className="grid grid-cols-[1fr_1fr_auto] items-end gap-1.5">
            <SelectField
              label="Model arrow from"
              value={from}
              options={[{ value: "", label: "from…" }, ...model.latents.map((x) => ({ value: x, label: x }))]}
              onChange={setFrom}
            />
            <SelectField
              label="to"
              value={to}
              options={[{ value: "", label: "to…" }, ...model.latents.map((x) => ({ value: x, label: x }))]}
              onChange={setTo}
            />
            <button
              type="button"
              className="btn"
              disabled={!!addProblem}
              title={addProblem ?? undefined}
              onClick={() => {
                editModel("model arrow", (d) => void d.model.arrows.push(`${from} -> ${to}`));
                setFrom("");
                setTo("");
              }}
            >
              Add
            </button>
          </div>
        </Section>
        <Section
          title={
            <>
              Equations · <Term term="T3 Step estimand">T3</Term> with R² before → after
            </>
          }
        >
          <EstimandCards />
        </Section>
      </aside>
    </div>
  );
}
