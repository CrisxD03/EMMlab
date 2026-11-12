// Automatic layout with ELK (layered, left → right, orthogonal edge routing). ELK reserves room for
// edge labels, so labels sit between layers instead of on top of nodes or each other. Categorical
// attributes get their own lane above the causal graph, so level-shift edges never cross latent rows.
// Positions and routes live in the UI store, not in the World spec.

import type { ElkExtendedEdge, ElkNode } from "elkjs/lib/elk-api";
import type { World } from "../../api/types";
import { arrowKey, fmtTyped, normKey } from "../../state/worldOps";

export const NODE_W = 200;
export const NODE_H = 92;

export interface Pt {
  x: number;
  y: number;
}
export interface Route {
  points: Pt[];
  label?: { x: number; y: number; width: number; height: number };
  /** Top-left of source and target when the route was computed (stale once either node moves). */
  basis: { from: Pt; to: Pt };
}
export interface Layered {
  positions: Record<string, Pt>;
  routes: Record<string, Route>;
  width: number;
  height: number;
}
export type Direction = "RIGHT" | "DOWN";

export interface LNode {
  id: string;
  width: number;
  height: number;
}
export interface LEdge {
  id: string;
  from: string;
  to: string;
  label?: { width: number; height: number };
}

// ELK is large: load it on first layout, not with the app shell.
let elkPromise: Promise<{ layout: (g: ElkNode) => Promise<ElkNode> }> | null = null;
const getElk = () => {
  elkPromise ??= import("elkjs/lib/elk.bundled.js").then((m) => new m.default());
  return elkPromise;
};

/** Layered layout of a DAG with one port per edge end and reserved space for edge labels. */
export async function layered(
  nodes: LNode[],
  edges: LEdge[],
  opts: { nodeGap?: number; layerGap?: number; direction?: Direction } = {},
): Promise<Layered> {
  const down = opts.direction === "DOWN";
  const known = new Set(nodes.map((n) => n.id));
  const es = edges.filter((e) => known.has(e.from) && known.has(e.to) && e.from !== e.to);
  const ports = new Map<string, { id: string; side: string }[]>();
  for (const e of es) {
    ports.set(e.from, [...(ports.get(e.from) ?? []), { id: `${e.id}::s`, side: down ? "SOUTH" : "EAST" }]);
    ports.set(e.to, [...(ports.get(e.to) ?? []), { id: `${e.id}::t`, side: down ? "NORTH" : "WEST" }]);
  }
  const graph: ElkNode = {
    id: "root",
    layoutOptions: {
      "elk.algorithm": "layered",
      "elk.direction": down ? "DOWN" : "RIGHT",
      "elk.edgeRouting": "ORTHOGONAL",
      "elk.layered.spacing.nodeNodeBetweenLayers": String(opts.layerGap ?? 70),
      "elk.spacing.nodeNode": String(opts.nodeGap ?? 36),
      "elk.layered.spacing.edgeNodeBetweenLayers": "22",
      "elk.layered.spacing.edgeEdgeBetweenLayers": "12",
      "elk.spacing.edgeEdge": "12",
      "elk.spacing.edgeNode": "18",
      "elk.spacing.edgeLabel": "4",
      "elk.spacing.componentComponent": "60",
      "elk.layered.nodePlacement.strategy": "NETWORK_SIMPLEX",
      "elk.layered.edgeLabels.sideSelection": down ? "SMART_DOWN" : "ALWAYS_UP",
      "elk.edgeLabels.placement": "CENTER",
      "elk.layered.mergeEdges": "false",
      "elk.padding": "[top=16,left=16,bottom=16,right=16]",
    },
    children: nodes.map((n) => ({
      id: n.id,
      width: n.width,
      height: n.height,
      layoutOptions: { "elk.portConstraints": "FIXED_SIDE" },
      ports: (ports.get(n.id) ?? []).map((p) => ({
        id: p.id,
        width: 1,
        height: 1,
        layoutOptions: { "elk.port.side": p.side },
      })),
    })),
    edges: es.map(
      (e): ElkExtendedEdge => ({
        id: e.id,
        sources: [`${e.id}::s`],
        targets: [`${e.id}::t`],
        labels: e.label
          ? [{ id: `${e.id}::l`, text: " ", width: e.label.width, height: e.label.height }]
          : [],
      }),
    ),
  };
  const res = await (await getElk()).layout(graph);
  const positions: Record<string, Pt> = {};
  for (const c of res.children ?? []) positions[c.id] = { x: c.x ?? 0, y: c.y ?? 0 };
  const routes: Record<string, Route> = {};
  for (const e of (res.edges ?? []) as ElkExtendedEdge[]) {
    const s = e.sections?.[0];
    const src = es.find((x) => x.id === e.id);
    if (!s || !src) continue;
    const l = e.labels?.[0];
    routes[e.id] = {
      points: [s.startPoint, ...(s.bendPoints ?? []), s.endPoint].map((p) => ({ x: p.x, y: p.y })),
      label: l ? { x: l.x ?? 0, y: l.y ?? 0, width: l.width ?? 0, height: l.height ?? 0 } : undefined,
      basis: { from: positions[src.from], to: positions[src.to] },
    };
  }
  return { positions, routes, width: res.width ?? 0, height: res.height ?? 0 };
}

/** SVG path through route points with rounded corners. */
export function roundedPath(points: Pt[], r = 8): string {
  if (points.length < 2) return "";
  let d = `M${points[0].x},${points[0].y}`;
  for (let i = 1; i < points.length - 1; i++) {
    const [a, b, c] = [points[i - 1], points[i], points[i + 1]];
    const d1 = Math.hypot(b.x - a.x, b.y - a.y);
    const d2 = Math.hypot(c.x - b.x, c.y - b.y);
    const k = Math.min(r, d1 / 2, d2 / 2);
    const p1 = { x: b.x + ((a.x - b.x) / (d1 || 1)) * k, y: b.y + ((a.y - b.y) / (d1 || 1)) * k };
    const p2 = { x: b.x + ((c.x - b.x) / (d2 || 1)) * k, y: b.y + ((c.y - b.y) / (d2 || 1)) * k };
    d += ` L${p1.x},${p1.y} Q${b.x},${b.y} ${p2.x},${p2.y}`;
  }
  const last = points[points.length - 1];
  return `${d} L${last.x},${last.y}`;
}

/** Width of the arrow label on the World canvas (slope · β · twist badges), in px. */
export function arrowLabelWidth(w: World, key: string): number {
  const a = w.arrows.find((x) => arrowKey(x) === key);
  const slope = fmtTyped(a?.slope ?? a?.std ?? 0);
  let width = 30 + 7 * (slope.length + 6);
  for (const t of w.twists.filter((t) => normKey(t.arrow) === key)) {
    const text = `⤳ ${fmtTyped(t.delta)} | ${Object.entries(t.when)
      .map(([k, v]) => `${k}=${v}`)
      .join(" & ")}`;
    width += 20 + 6.4 * text.length;
  }
  return Math.round(width);
}

export const ATTR_LANE_GAP = 170;
const ATTR_SIDE_GAP = 150;

/**
 * World layout: continuous nodes by ELK (arrows routed, labels reserved), categorical attributes in
 * their own lane (above a left-to-right graph, left of a top-down one), each near the nodes it shifts.
 * With a viewport, both directions are tried and the one that fits larger on screen wins, unless the World
 * sets `layout_direction` or the caller asks to keep the current one.
 */
export async function worldLayout(
  w: World,
  sizes: Record<string, { width: number; height: number } | undefined>,
  viewport?: { width: number; height: number },
  keep?: Direction,
): Promise<{ positions: Record<string, Pt>; routes: Record<string, Route>; direction: Direction }> {
  // The World can force its direction; so can the caller, to redraw without flipping (e.g. after a display toggle).
  const forced = w.layout_direction ?? keep;
  if (forced) return worldLayoutDir(w, sizes, forced);
  const right = await worldLayoutDir(w, sizes, "RIGHT");
  if (!viewport?.width || !viewport.height || Object.keys(w.nodes).length < 4) return right;
  const down = await worldLayoutDir(w, sizes, "DOWN");
  const scale = (r: typeof right) => {
    const b = bounds(r.positions, sizes);
    return Math.min(viewport.width / b.width, viewport.height / b.height);
  };
  return scale(down) > scale(right) * 1.08 ? down : right;
}

function bounds(
  positions: Record<string, Pt>,
  sizes: Record<string, { width: number; height: number } | undefined>,
) {
  const ids = Object.keys(positions);
  const x0 = Math.min(...ids.map((i) => positions[i].x));
  const y0 = Math.min(...ids.map((i) => positions[i].y));
  const x1 = Math.max(...ids.map((i) => positions[i].x + (sizes[i]?.width ?? NODE_W)));
  const y1 = Math.max(...ids.map((i) => positions[i].y + (sizes[i]?.height ?? NODE_H)));
  return { width: Math.max(1, x1 - x0), height: Math.max(1, y1 - y0) };
}

async function worldLayoutDir(
  w: World,
  sizes: Record<string, { width: number; height: number } | undefined>,
  direction: Direction,
) {
  const size = (id: string, dw: number, dh: number) => ({
    width: sizes[id]?.width ?? dw,
    height: sizes[id]?.height ?? dh,
  });
  // Continuous attributes without arrows (e.g. a decoy) sit in the attribute lane, not in the graph.
  const linked = new Set(w.arrows.flatMap((a) => [a.from, a.to]));
  const laneNodes = Object.keys(w.nodes).filter((id) => w.nodes[id].kind === "attribute" && !linked.has(id));
  const cont = Object.keys(w.nodes)
    .filter((id) => !laneNodes.includes(id))
    .map((id) => ({ id, ...size(id, NODE_W, NODE_H) }));
  const edges: LEdge[] = w.arrows
    .filter((a) => a.from in w.nodes && a.to in w.nodes)
    .map((a) => {
      const key = arrowKey(a);
      return { id: key, from: a.from, to: a.to, label: { width: arrowLabelWidth(w, key), height: 24 } };
    });
  const res = await layered(cont, edges, {
    layerGap: direction === "DOWN" ? 36 : 56,
    nodeGap: 40,
    direction,
  });
  const positions = { ...res.positions };
  const vals = Object.values(positions);
  // Attribute lane, ordered by where the nodes each attribute shifts sit along the lane.
  const along = (id: string) =>
    direction === "RIGHT"
      ? (positions[id]?.x ?? 0) + (sizes[id]?.width ?? NODE_W) / 2
      : (positions[id]?.y ?? 0) + (sizes[id]?.height ?? NODE_H) / 2;
  const attrs = [...Object.keys(w.attributes), ...laneNodes].map((id) => {
    const targets = w.shifts.filter((s) => id in s.when && s.target in positions).map((s) => along(s.target));
    return {
      id,
      ...size(id, 180, 64),
      c: targets.length ? targets.reduce((x, y) => x + y, 0) / targets.length : 0,
    };
  });
  attrs.sort((a, b) => a.c - b.c);
  let cursor = Number.NEGATIVE_INFINITY;
  if (direction === "RIGHT") {
    const top = Math.min(0, ...vals.map((p) => p.y));
    const laneH = Math.max(0, ...attrs.map((a) => a.height));
    for (const a of attrs) {
      const x = Math.max(a.c - a.width / 2, cursor);
      positions[a.id] = { x, y: top - ATTR_LANE_GAP - laneH };
      cursor = x + a.width + 28;
    }
  } else {
    const left = Math.min(0, ...vals.map((p) => p.x));
    const laneW = Math.max(0, ...attrs.map((a) => a.width));
    for (const a of attrs) {
      const y = Math.max(a.c - a.height / 2, cursor);
      positions[a.id] = { x: left - ATTR_SIDE_GAP - laneW, y };
      cursor = y + a.height + 20;
    }
  }
  return { positions, routes: res.routes, direction };
}

/** Place a new node without moving anything: attributes extend the top lane, latents extend right. */
export function freeSlot(all: Pt[], sameKind: Pt[], column: "left" | "right"): Pt {
  if (!all.length) return { x: 0, y: column === "left" ? -NODE_H - ATTR_LANE_GAP : 0 };
  const xs = all.map((p) => p.x);
  const ys = all.map((p) => p.y);
  if (column === "left") {
    if (!sameKind.length) return { x: Math.min(...xs), y: Math.min(...ys) - NODE_H - ATTR_LANE_GAP };
    const last = sameKind.reduce((a, b) => (b.x > a.x ? b : a));
    return { x: last.x + 210, y: last.y };
  }
  if (!sameKind.length)
    return { x: Math.max(...xs) + NODE_W + 90, y: Math.max(0, ...ys.filter((y) => y >= 0)) };
  const last = sameKind[sameKind.length - 1];
  return { x: last.x + NODE_W + 90, y: last.y + (sameKind.length % 2 ? 60 : -60) };
}
