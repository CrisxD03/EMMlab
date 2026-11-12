import {
  applyNodeChanges,
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
  type NodeChange,
  type NodeProps,
  Position,
  ReactFlow,
  useInternalNode,
  useReactFlow,
  useUpdateNodeInternals,
} from "@xyflow/react";
import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { DerivedValue } from "../../components/ui";
import { useDoc } from "../../state/doc";
import { useTruth } from "../../state/truth";
import { shiftEdgesShown, useUI } from "../../state/ui";
import { arrowKey, connectionProblem, defaultValue, fmtTyped, normKey } from "../../state/worldOps";
import { addArrow, addShift } from "./actions";
import { freeSlot, type Route, roundedPath, worldLayout } from "./layout";

type WNodeData = { kind: "latent" | "attribute" | "categorical" };
type WNode = Node<WNodeData, "world">;
type ArrowData = { key: string };
type ShiftData = { index: number };

// ----------------------------------------------------------------------------- nodes

const WorldNodeView = memo(function WorldNodeView({ id, data }: NodeProps<WNode>) {
  const world = useDoc((s) => s.doc.world);
  const selected = useUI(
    (s) => (s.selection.type === "node" || s.selection.type === "attribute") && s.selection.id === id,
  );
  const highlighted = useUI((s) => s.highlight.includes(id));
  const down = useUI((s) => s.layoutDir === "DOWN");
  // handles move with the layout direction: React Flow must re-measure them
  const updateInternals = useUpdateNodeInternals();
  // biome-ignore lint/correctness/useExhaustiveDependencies: re-measure whenever the direction flips
  useEffect(() => {
    updateInternals(id);
  }, [down, id, updateInternals]);
  const showShiftEdges = useUI((s) => shiftEdgesShown(s.shiftEdges, world.shifts.length));
  const shiftSelected = useUI((s) => (s.selection.type === "shift" ? s.selection.index : -1));
  const issues = useTruth().issues;
  const { derived } = useTruth();
  const invalid =
    highlighted || issues.some((i) => i.element === `nodes.${id}` || i.element.startsWith(`nodes.${id}.`));

  if (data.kind === "categorical") {
    const a = world.attributes[id];
    if (!a) return null;
    return (
      <div
        className={`wnode kind-categorical ${a.role === "decoy" ? "is-decoy" : ""} ${selected ? "is-selected" : ""} ${invalid ? "is-invalid" : ""}`}
        data-testid={`node-${id}`}
      >
        <div className="flex items-center gap-1.5">
          <span className="title">{id}</span>
          <span className="badge">{a.level ?? "student"}</span>
          {a.role === "decoy" ? <span className="badge badge-warn">decoy</span> : null}
        </div>
        <div className="mt-1 flex flex-wrap gap-1">
          {Object.entries(a.values).map(([v, p]) => (
            <span key={v} className="typed text-[11px]">
              {v} <span className="num">{fmtTyped(p)}</span>
            </span>
          ))}
        </div>
        <Handle type="source" id="out" position={down ? Position.Right : Position.Bottom} />
      </div>
    );
  }

  const n = world.nodes[id];
  if (!n) return null;
  const d = derived?.nodes[id];
  const shiftBadges = showShiftEdges
    ? []
    : world.shifts.map((s, i) => ({ s, i })).filter(({ s }) => s.target === id);
  const isAttr = (n.kind ?? "latent") === "attribute";
  const typed = n.root
    ? `N(${fmtTyped(n.root.mean ?? 0)}, ${fmtTyped(n.root.sd ?? 1)})`
    : `b₀ ${fmtTyped(n.recipe?.intercept ?? 0)}${n.recipe?.noise_sd !== undefined ? ` · σε ${fmtTyped(n.recipe.noise_sd)}` : ""}`;
  const t = n.target;
  const targetText = t
    ? [
        t.sd !== undefined && `sd→${fmtTyped(t.sd)}`,
        t.r2 !== undefined && `R²→${fmtTyped(t.r2)}`,
        t.icc !== undefined && `icc→${fmtTyped(t.icc)}`,
      ]
        .filter(Boolean)
        .join(" ")
    : "";
  return (
    <div
      className={`wnode ${isAttr ? "kind-attribute" : ""} ${n.role === "decoy" ? "is-decoy" : ""} ${selected ? "is-selected" : ""} ${invalid ? "is-invalid" : ""}`}
      data-testid={`node-${id}`}
    >
      <Handle type="target" id="in" position={down ? Position.Top : Position.Left} />
      <Handle
        type="target"
        id="shift"
        position={down ? Position.Left : Position.Top}
        className="handle-shift"
      />
      <div className="flex items-center gap-1.5">
        <span className="title" title={n.label ?? id}>
          {id}
        </span>
        {(n.level ?? "student") === "school" ? <span className="badge">school</span> : null}
        {isAttr ? <span className="badge">attribute</span> : null}
        {n.role === "decoy" ? <span className="badge badge-warn">decoy</span> : null}
        {n.observed === false ? <span className="badge">unobserved</span> : null}
      </div>
      <div className="typed num mt-0.5">
        {typed}
        {targetText ? <span className="ml-1">{targetText}</span> : null}
      </div>
      <div className="nodrag mt-1 flex flex-wrap items-center gap-x-1.5 gap-y-0.5 text-[11.5px] text-muted">
        <span>
          x̄{" "}
          <DerivedValue
            value={d?.mean}
            label={`Mean of ${id} (pooled)`}
            rippleKey={`node:${id}:mean`}
            digits={2}
          />
        </span>
        <span>
          sd{" "}
          <DerivedValue value={d?.sd} label={`SD of ${id} (pooled)`} rippleKey={`node:${id}:sd`} digits={2} />
        </span>
        {n.recipe ? (
          <span>
            R² <DerivedValue value={d?.r2} label={`R² of ${id}`} rippleKey={`node:${id}:r2`} digits={2} />
          </span>
        ) : null}
      </div>
      {shiftBadges.length ? (
        <div className="nodrag mt-1 flex flex-wrap gap-1">
          {shiftBadges.map(({ s, i }) => (
            <button
              type="button"
              key={i}
              className={`badge shift-badge ${shiftSelected === i ? "badge-accent" : ""}`}
              data-testid={`shift-badge-${i}`}
              aria-label={`Select level shift ${i + 1}: ${s.delta} when ${Object.entries(s.when)
                .map(([k, v]) => `${k}=${v}`)
                .join(", ")}`}
              onClick={(e) => {
                e.stopPropagation(); // the node itself must not take the selection
                useUI.getState().select({ type: "shift", index: i });
              }}
            >
              {s.delta >= 0 ? "+" : ""}
              {fmtTyped(s.delta)} {Object.values(s.when).join("&")}
            </button>
          ))}
        </div>
      ) : null}
      <Handle type="source" id="out" position={down ? Position.Bottom : Position.Right} />
    </div>
  );
});

// ----------------------------------------------------------------------------- edges

/** An ELK route is used while both end nodes are where the layout put them; otherwise a curve. */
function routeValid(r: Route, from?: { x: number; y: number }, to?: { x: number; y: number }) {
  const near = (a?: { x: number; y: number }, b?: { x: number; y: number }) =>
    !!a && !!b && Math.abs(a.x - b.x) < 1 && Math.abs(a.y - b.y) < 1;
  return near(from, r.basis.from) && near(to, r.basis.to);
}

function ArrowEdgeView(props: EdgeProps<Edge<ArrowData, "arrow">>) {
  const { sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, markerEnd, data, id } = props;
  const key = data?.key ?? "";
  const world = useDoc((s) => s.doc.world);
  const { derived } = useTruth();
  const selected = useUI((s) => s.selection.type === "arrow" && s.selection.id === key);
  const rawRipple = useUI((s) => s.ripple.kinds[`arrow:${key}:raw`]);
  const seq = useUI((s) => s.ripple.seq);
  const select = useUI((s) => s.select);
  const route = useUI((s) => s.routes[key]);
  const src = useInternalNode(props.source);
  const tgt = useInternalNode(props.target);
  let [path, lx, ly] = getBezierPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
  });
  if (route && routeValid(route, src?.internals.positionAbsolute, tgt?.internals.positionAbsolute)) {
    path = roundedPath(route.points);
    if (route.label) {
      lx = route.label.x + route.label.width / 2;
      ly = route.label.y + route.label.height / 2;
    } else {
      const m = route.points[Math.floor(route.points.length / 2)];
      lx = m.x;
      ly = m.y;
    }
  }
  const a = world.arrows.find((x) => arrowKey(x) === key);
  const twists = world.twists.map((t, i) => ({ t, i })).filter(({ t }) => normKey(t.arrow) === key);
  const da = derived?.arrows[key];
  return (
    <>
      <g
        key={rawRipple ? `r${seq}` : "e"}
        data-ripple={rawRipple}
        className={rawRipple === "slope" ? "edge-ripple-slope" : ""}
      >
        <BaseEdge id={id} path={path} markerEnd={markerEnd} />
      </g>
      <EdgeLabelRenderer>
        <div
          className={`edge-label nodrag nopan ${selected ? "is-selected" : ""}`}
          style={{ transform: `translate(-50%, -50%) translate(${lx}px, ${ly}px)` }}
          data-testid={`edge-${key}`}
          data-ripple={rawRipple}
        >
          <button
            type="button"
            className="num font-semibold"
            style={{ background: "none", border: 0, padding: 0, color: "inherit", cursor: "pointer" }}
            aria-label={`Select arrow ${key}`}
            onClick={() => select({ type: "arrow", id: key })}
          >
            {a?.slope !== undefined ? fmtTyped(a.slope) : `β ${fmtTyped(a?.std)}`}
          </button>
          <span className="text-faint">β</span>
          <DerivedValue
            value={da?.std}
            label={`Standardized coefficient ${key} (pooled, slope × SD parent ÷ SD child)`}
            rippleKey={`arrow:${key}:std`}
            digits={2}
          />
          {twists.map(({ t, i }) => (
            <button
              type="button"
              key={i}
              className="badge badge-twist"
              data-testid={`twist-badge-${key}`}
              aria-label={`Twist on ${key}: delta ${t.delta} when ${Object.entries(t.when)
                .map(([k, v]) => `${k}=${v}`)
                .join(", ")}`}
              onClick={() => select({ type: "twist", index: i })}
            >
              ⤳ {fmtTyped(t.delta)} |{" "}
              {Object.entries(t.when)
                .map(([k, v]) => `${k}=${v}`)
                .join(" & ")}
            </button>
          ))}
        </div>
      </EdgeLabelRenderer>
    </>
  );
}

function ShiftEdgeView(props: EdgeProps<Edge<ShiftData, "shift">>) {
  const { sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, markerEnd, data, id } = props;
  const index = data?.index ?? 0;
  const s = useDoc((st) => st.doc.world.shifts[index]);
  const selected = useUI((st) => st.selection.type === "shift" && st.selection.index === index);
  const select = useUI((st) => st.select);
  const [path] = getBezierPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
  });
  // label near the target so labels of one attribute's shifts fan out instead of stacking
  const lx = sourceX + 0.62 * (targetX - sourceX);
  const ly = sourceY + 0.62 * (targetY - sourceY);
  if (!s) return null;
  return (
    <>
      <BaseEdge
        id={id}
        path={path}
        markerEnd={markerEnd}
        style={{ strokeDasharray: "5 4", stroke: "var(--shift)" }}
      />
      <EdgeLabelRenderer>
        <button
          type="button"
          className={`edge-label kind-shift nodrag nopan ${selected ? "is-selected" : ""}`}
          style={{ transform: `translate(-50%, -50%) translate(${lx}px, ${ly}px)` }}
          data-testid={`shift-${index}`}
          aria-label={`Select level shift ${index + 1}`}
          onClick={() => select({ type: "shift", index })}
        >
          <span className="num">
            {s.delta >= 0 ? "+" : ""}
            {fmtTyped(s.delta)}
          </span>
          <span>
            {Object.entries(s.when)
              .map(([k, v]) => `${k}=${v}`)
              .join(" & ")}
          </span>
        </button>
      </EdgeLabelRenderer>
    </>
  );
}

const nodeTypes = { world: WorldNodeView };
const edgeTypes = { arrow: ArrowEdgeView, shift: ShiftEdgeView };

// ----------------------------------------------------------------------------- canvas

export function WorldCanvas() {
  const world = useDoc((s) => s.doc.world);
  const positions = useUI((s) => s.positions);
  const layoutSeq = useUI((s) => s.layoutSeq);
  const selection = useUI((s) => s.selection);
  const shiftEdges = useUI((s) => s.shiftEdges);
  const select = useUI((s) => s.select);
  const setUI = useUI((s) => s.set);
  const rf = useReactFlow();
  const [nodes, setNodes] = useState<WNode[]>([]);
  const [refused, setRefused] = useState<string | null>(null);

  // Positions for nodes created outside the canvas (YAML import) are filled in without moving others.
  useEffect(() => {
    const names = [...Object.keys(world.attributes), ...Object.keys(world.nodes)];
    const missing = names.filter((n) => !positions[n]);
    if (!missing.length) return;
    const next = { ...positions };
    for (const m of missing) {
      const left = m in world.attributes || world.nodes[m]?.kind === "attribute";
      const same = Object.entries(next)
        .filter(([k]) => (left ? k in world.attributes : k in world.nodes))
        .map(([, p]) => p);
      next[m] = freeSlot(Object.values(next), same, left ? "left" : "right");
    }
    setUI({ positions: next });
  }, [world, positions, setUI]);

  // Full auto-layout on request (preset load, "Tidy"): ELK with measured node sizes.
  const lastLayout = useRef(-1);
  const fitPending = useRef(false);
  // Fit once React Flow has the laid-out positions (after the nodes state below has been updated).
  useEffect(() => {
    if (!fitPending.current || !nodes.length) return;
    const pos = useUI.getState().positions;
    if (!nodes.every((n) => pos[n.id] && n.position.x === pos[n.id].x && n.position.y === pos[n.id].y))
      return;
    fitPending.current = false;
    setTimeout(() => rf.fitView({ padding: 0.1, maxZoom: 1.1, duration: 0 }), 30);
  }, [nodes, rf]);
  useEffect(() => {
    if (layoutSeq === lastLayout.current || layoutSeq === 0) return;
    lastLayout.current = layoutSeq;
    let cancelled = false;
    const run = async () => {
      const w0 = useDoc.getState().doc.world;
      const ids = [...Object.keys(w0.attributes), ...Object.keys(w0.nodes)];
      // wait (briefly) until React Flow has measured every node
      for (let i = 0; i < 30; i++) {
        const ns = rf.getNodes();
        if (ids.every((id) => ns.find((n) => n.id === id)?.measured?.width)) break;
        await new Promise((r) => setTimeout(r, 20));
      }
      if (cancelled) return;
      const sizes: Record<string, { width: number; height: number } | undefined> = {};
      for (const n of rf.getNodes())
        if (n.measured?.width && n.measured.height)
          sizes[n.id] = { width: n.measured.width, height: n.measured.height };
      const el = document.querySelector<HTMLElement>("[data-testid=world-canvas]");
      const viewport = el ? { width: el.clientWidth, height: el.clientHeight } : undefined;
      const ui = useUI.getState();
      const res = await worldLayout(
        useDoc.getState().doc.world,
        sizes,
        viewport,
        ui.keepLayoutDir ? ui.layoutDir : undefined,
      );
      if (cancelled) return;
      fitPending.current = true;
      setUI({ positions: res.positions, routes: res.routes, layoutDir: res.direction, keepLayoutDir: false });
    };
    void run();
    return () => {
      cancelled = true;
    };
  }, [layoutSeq, rf, setUI]);

  useEffect(() => {
    setNodes((prev) => {
      const byId = new Map(prev.map((n) => [n.id, n]));
      const out: WNode[] = [];
      const add = (id: string, kind: WNodeData["kind"]) => {
        const old = byId.get(id);
        const pos = positions[id] ?? old?.position ?? { x: 0, y: 0 };
        const sel = (selection.type === "node" || selection.type === "attribute") && selection.id === id;
        out.push({
          ...(old ?? {}),
          id,
          type: "world",
          position: pos,
          data: old && old.data.kind === kind ? old.data : { kind },
          selected: sel,
          ariaLabel: `${kind === "categorical" ? "Categorical attribute" : kind === "attribute" ? "Continuous attribute" : "Latent"} ${id}`,
        });
      };
      for (const id of Object.keys(world.attributes)) add(id, "categorical");
      for (const [id, n] of Object.entries(world.nodes))
        add(id, (n.kind ?? "latent") === "attribute" ? "attribute" : "latent");
      return out;
    });
  }, [world, positions, selection]);

  // Keep newly added nodes in view (without zooming in past 100%).
  const count = Object.keys(world.nodes).length + Object.keys(world.attributes).length;
  const lastCount = useRef(count);
  useEffect(() => {
    if (count > lastCount.current) {
      const t = setTimeout(() => rf.fitView({ padding: 0.2, maxZoom: 1, duration: 150 }), 30);
      lastCount.current = count;
      return () => clearTimeout(t);
    }
    lastCount.current = count;
  }, [count, rf]);

  const edges = useMemo<Edge[]>(() => {
    const out: Edge[] = [];
    for (const a of world.arrows) {
      if (!(a.from in world.nodes) || !(a.to in world.nodes)) continue;
      const key = arrowKey(a);
      out.push({
        id: `a:${key}`,
        type: "arrow",
        source: a.from,
        target: a.to,
        sourceHandle: "out",
        targetHandle: "in",
        data: { key },
        selected: selection.type === "arrow" && selection.id === key,
        markerEnd: { type: MarkerType.ArrowClosed, width: 16, height: 16 },
        ariaLabel: `Arrow ${key}`,
      });
    }
    if (shiftEdgesShown(shiftEdges, world.shifts.length))
      world.shifts.forEach((s, index) => {
        for (const attr of Object.keys(s.when)) {
          if (!(attr in world.attributes) || !(s.target in world.nodes)) continue;
          out.push({
            id: `s:${index}:${attr}`,
            type: "shift",
            source: attr,
            target: s.target,
            sourceHandle: "out",
            targetHandle: "shift",
            data: { index },
            selected: selection.type === "shift" && selection.index === index,
            markerEnd: { type: MarkerType.ArrowClosed, width: 14, height: 14 },
            ariaLabel: `Level shift ${attr}=${s.when[attr]} on ${s.target}`,
          });
        }
      });
    return out;
  }, [world, selection, shiftEdges]);

  const onNodesChange = useCallback(
    (changes: NodeChange<WNode>[]) => {
      setNodes((ns) =>
        applyNodeChanges(
          changes.filter((c) => c.type !== "remove"),
          ns,
        ),
      );
      for (const c of changes) {
        if (c.type === "select" && c.selected) {
          const w = useDoc.getState().doc.world;
          const sel = useUI.getState().selection;
          const type = c.id in w.attributes ? "attribute" : "node";
          if (!(sel.type === type && "id" in sel && sel.id === c.id)) select({ type, id: c.id });
        }
      }
    },
    [select],
  );

  const isValidConnection = useCallback((c: Connection | Edge) => {
    const problem = connectionProblem(useDoc.getState().doc.world, c.source, c.target);
    return problem === null;
  }, []);

  const onConnect = useCallback((c: Connection) => {
    const w = useDoc.getState().doc.world;
    const problem = connectionProblem(w, c.source, c.target);
    if (problem) {
      setRefused(problem);
      return;
    }
    setRefused(null);
    if (c.source in w.attributes) {
      addShift(c.source, defaultValue(w.attributes[c.source].values), c.target, 0.5);
      const idx = useDoc.getState().doc.world.shifts.length - 1;
      useUI.getState().select({ type: "shift", index: idx }, "delta");
    } else if (addArrow(c.source, c.target, 0.3)) {
      useUI.getState().select({ type: "arrow", id: `${c.source} -> ${c.target}` }, "slope");
    }
  }, []);

  return (
    <div className="relative h-full w-full" data-testid="world-canvas">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        onNodesChange={onNodesChange}
        onNodeDragStop={(_, node) =>
          setUI({ positions: { ...useUI.getState().positions, [node.id]: node.position } })
        }
        onConnect={onConnect}
        isValidConnection={isValidConnection}
        onEdgeClick={(_, e) => {
          if (e.type === "arrow") select({ type: "arrow", id: (e.data as ArrowData).key });
          else select({ type: "shift", index: (e.data as ShiftData).index });
        }}
        onPaneClick={() => select({ type: "world" })}
        deleteKeyCode={null}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        minZoom={0.25}
        maxZoom={2}
        proOptions={{ hideAttribution: true }}
        aria-label="World canvas"
      >
        <Background gap={18} size={1.2} color="var(--canvas-dot)" />
        <Controls showInteractive={false} position="bottom-left" />
      </ReactFlow>
      {refused ? (
        <div role="status" className="badge badge-bad absolute top-2 left-1/2 -translate-x-1/2">
          Connection refused: {refused}
        </div>
      ) : null}
      {!world.arrows.length && !Object.keys(world.nodes).length && !Object.keys(world.attributes).length ? (
        <div className="pointer-events-none absolute inset-0 grid place-items-center">
          <div className="max-w-sm text-center text-muted">
            <div className="font-semibold text-fg">An empty World</div>
            Add a latent or an attribute from the inspector on the right, then drag from a node's right handle
            to another node to draw an arrow.
          </div>
        </div>
      ) : null}
    </div>
  );
}
