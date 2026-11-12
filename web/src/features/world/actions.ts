// High-level, undoable World edits used by the canvas and the inspector.

import type { CategoricalAttribute, ContinuousNode, World } from "../../api/types";
import { type EditKind, useDoc } from "../../state/doc";
import { useUI } from "../../state/ui";
import * as ops from "../../state/worldOps";
import { freeSlot } from "./layout";

const doc = () => useDoc.getState();
const ui = () => useUI.getState();

export function editWorld(label: string, kind: EditKind, fn: (w: World) => void) {
  doc().apply(label, kind, (d) => fn(d.world as World));
}

function place(name: string, column: "left" | "right") {
  const { positions } = ui();
  const w = doc().doc.world;
  const all = Object.values(positions);
  const same = Object.entries(positions)
    .filter(([k]) =>
      column === "left" ? k in w.attributes || w.nodes[k]?.kind === "attribute" : k in w.nodes,
    )
    .map(([, p]) => p);
  ui().set({ positions: { ...positions, [name]: freeSlot(all, same, column) } });
}

export function addContinuous(kind: "latent" | "attribute" = "latent"): string {
  const w = doc().doc.world;
  const name = ops.uniqueName(w, kind === "latent" ? "latent" : "num");
  const node: ContinuousNode = { root: { mean: 0, sd: 1 } };
  if (kind === "attribute") node.kind = "attribute";
  place(name, kind === "latent" ? "right" : "left");
  editWorld(`add ${kind} ${name}`, "structure", (w) => {
    w.nodes[name] = node;
  });
  ui().select({ type: "node", id: name }, "name");
  return name;
}

export function addCategorical(): string {
  const w = doc().doc.world;
  const name = ops.uniqueName(w, "attr");
  const attr: CategoricalAttribute = { level: "student", values: { low: 0.5, high: 0.5 } };
  place(name, "left");
  editWorld(`add attribute ${name}`, "structure", (w) => {
    w.attributes[name] = attr;
  });
  ui().select({ type: "attribute", id: name }, "name");
  return name;
}

export function renameElement(kind: "node" | "attribute", oldName: string, newName: string) {
  if (oldName === newName) return;
  doc().apply(`rename ${oldName} → ${newName}`, "structure", (d) => {
    if (kind === "node") ops.renameNode(d.world as World, oldName, newName, d.model);
    else ops.renameAttribute(d.world as World, oldName, newName);
    d.method.analysis.residualize_on = d.method.analysis.residualize_on.map((x) =>
      x === oldName ? newName : x,
    );
    if (d.search.binning[oldName]) {
      d.search.binning = ops.renameKeyInRecord(d.search.binning, oldName, newName);
    }
    if (d.search.attributes)
      d.search.attributes = d.search.attributes.map((x) => (x === oldName ? newName : x));
  });
  const { positions } = ui();
  ui().set({ positions: ops.renameKeyInRecord(positions, oldName, newName) });
  ui().select({ type: kind, id: newName });
}

export function deleteElement(kind: "node" | "attribute", name: string) {
  doc().apply(`delete ${name}`, "structure", (d) => {
    if (kind === "node") ops.removeNode(d.world as World, name, d.model);
    else ops.removeAttribute(d.world as World, name);
    d.method.analysis.residualize_on = d.method.analysis.residualize_on.filter((x) => x !== name);
    if (d.search.attributes) d.search.attributes = d.search.attributes.filter((x) => x !== name);
    delete d.search.binning[name];
    if (d.search.filter && name in d.search.filter) delete d.search.filter;
  });
  ui().select({ type: "world" });
}

export function addArrow(from: string, to: string, slope = 0.3): boolean {
  const w = doc().doc.world;
  if (ops.connectionProblem(w, from, to)) return false;
  editWorld(`arrow ${from} → ${to}`, "structure", (w) => ops.addArrow(w, from, to, slope));
  return true;
}

export function removeArrow(key: string) {
  editWorld(`remove arrow ${key}`, "structure", (w) => ops.removeArrow(w, key));
  if (ui().selection.type === "arrow") ui().select({ type: "world" });
}

export function setArrowSlope(key: string, value: number, mode: "slope" | "std" = "slope") {
  editWorld(`slope ${key}`, "slope", (w) => {
    const a = w.arrows.find((x) => ops.arrowKey(x) === key);
    if (!a) return;
    if (mode === "slope") {
      a.slope = value;
      delete a.std;
    } else {
      a.std = value;
      delete a.slope;
    }
  });
}

export function addShift(attr: string, value: string, target: string, delta = 0.5): boolean {
  const w = doc().doc.world;
  if (!w.attributes[attr] || !w.nodes[target]) return false;
  editWorld(`shift ${attr}=${value} → ${target}`, "structure", (w) =>
    ops.addShift(w, attr, value, target, delta),
  );
  return true;
}

export function removeShift(index: number) {
  editWorld("remove level shift", "structure", (w) => ops.removeShift(w, index));
  if (ui().selection.type === "shift") ui().select({ type: "world" });
}

/** A twist on one condition, or on two (`and`): then the slope changes only where both hold. */
export function addTwist(
  arrow: string,
  attr: string,
  value: string,
  delta: number,
  and?: [string, string],
): boolean {
  const w = doc().doc.world;
  if (!w.arrows.some((a) => ops.arrowKey(a) === arrow) || !w.attributes[attr]) return false;
  if (and && (and[0] === attr || !w.attributes[and[0]])) return false;
  editWorld(`twist on ${arrow}`, "slope", (w) => {
    w.twists.push({ arrow, when: and ? { [attr]: value, [and[0]]: and[1] } : { [attr]: value }, delta });
  });
  return true;
}

export function removeTwist(index: number) {
  editWorld("remove twist", "slope", (w) => {
    w.twists.splice(index, 1);
  });
}

/** Load a whole World (preset, YAML import), keeping or recomputing the Researcher's Model. */
export function relayout() {
  ui().set({ layoutSeq: ui().layoutSeq + 1 });
}
