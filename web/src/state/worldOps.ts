// Pure, mutating helpers on a World draft (used inside immer `produce`). They keep the World
// consistent while it is edited on the canvas: references follow renames, deleting a node removes
// its arrows/shifts/twists, and a node becomes a Child (recipe) or Root (distribution) automatically
// when it gains or loses its last incoming arrow or level shift.

import type { Arrow, ResearcherModel, World } from "../api/types";

export const arrowKey = (a: { from: string; to: string }) => `${a.from} -> ${a.to}`;

export function parseArrowKey(key: string): [string, string] {
  const i = key.indexOf("->");
  if (i < 0) return [key.trim(), ""];
  return [key.slice(0, i).trim(), key.slice(i + 2).trim()];
}

export function emptyWorld(name = "untitled"): World {
  return {
    name,
    description: "",
    schools: { n_schools: 200, students_per_school: { dist: "fixed", n: 35 }, allocation: "stratified" },
    attributes: {},
    nodes: {},
    arrows: [],
    shifts: [],
    twists: [],
  };
}

export function isChild(w: World, name: string): boolean {
  return w.arrows.some((a) => a.to === name) || w.shifts.some((s) => s.target === name);
}

/** Convert root <-> recipe for the named nodes so they match their position in the graph. */
export function syncRootRecipe(w: World, names: Iterable<string>) {
  for (const name of names) {
    const n = w.nodes[name];
    if (!n) continue;
    const child = isChild(w, name);
    if (child && !n.recipe) {
      n.recipe = { intercept: n.root?.mean ?? 0, noise_sd: n.root?.sd ?? 1 };
      delete n.root;
    } else if (!child && !n.root) {
      n.root = { mean: n.recipe?.intercept ?? 0, sd: n.recipe?.noise_sd ?? n.target?.sd ?? 1 };
      delete n.recipe;
      if (n.target) {
        delete n.target.sd;
        delete n.target.r2;
        if (Object.keys(n.target).length === 0) delete n.target;
      }
    }
  }
}

export function uniqueName(w: World, base: string): string {
  const taken = new Set([...Object.keys(w.nodes), ...Object.keys(w.attributes)]);
  for (let i = 1; ; i++) {
    const n = `${base}${i}`;
    if (!taken.has(n)) return n;
  }
}

export function nameTaken(w: World, name: string): boolean {
  return name in w.nodes || name in w.attributes;
}

export const validName = (name: string) => /^[A-Za-z_][A-Za-z0-9_]*$/.test(name);

/** True when adding from -> to would close a directed cycle among continuous nodes. */
export function wouldCycle(w: World, from: string, to: string): boolean {
  if (from === to) return true;
  const out = new Map<string, string[]>();
  for (const a of w.arrows) out.set(a.from, [...(out.get(a.from) ?? []), a.to]);
  const stack = [to];
  const seen = new Set<string>();
  while (stack.length) {
    const v = stack.pop()!;
    if (v === from) return true;
    if (seen.has(v)) continue;
    seen.add(v);
    stack.push(...(out.get(v) ?? []));
  }
  return false;
}

/** Why a canvas connection source -> target is not allowed (null = allowed). Used while dragging. */
export function connectionProblem(w: World, source: string, target: string): string | null {
  if (source === target) return "a node cannot cause itself";
  const tn = w.nodes[target];
  if (!tn) return "arrows and level shifts must end at a continuous node";
  const cat = w.attributes[source];
  if (cat) {
    if ((tn.level ?? "student") === "school" && (cat.level ?? "student") === "student")
      return "a school-level node cannot depend on a student-level attribute";
    return null;
  }
  const sn = w.nodes[source];
  if (!sn) return "unknown source";
  if (w.arrows.some((a) => a.from === source && a.to === target)) return "this arrow already exists";
  if ((tn.level ?? "student") === "school" && (sn.level ?? "student") === "student")
    return "a school-level node can only have school-level parents";
  if ((tn.kind ?? "latent") === "attribute" && (sn.kind ?? "latent") === "latent")
    return "a continuous attribute cannot be caused by a latent";
  if (wouldCycle(w, source, target)) return "this arrow would create a cycle";
  return null;
}

export function addArrow(w: World, from: string, to: string, slope = 0.3) {
  w.arrows.push({ from, to, slope });
  syncRootRecipe(w, [to]);
}

export function removeArrow(w: World, key: string) {
  const idx = w.arrows.findIndex((a) => arrowKey(a) === key);
  if (idx < 0) return;
  const [a] = w.arrows.splice(idx, 1);
  w.twists = w.twists.filter((t) => normKey(t.arrow) !== key);
  syncRootRecipe(w, [a.to]);
}

export const normKey = (k: string) => {
  const [a, b] = parseArrowKey(k);
  return `${a} -> ${b}`;
};

export function addShift(w: World, attr: string, value: string, target: string, delta = 0.5) {
  w.shifts.push({ when: { [attr]: value }, target, delta });
  syncRootRecipe(w, [target]);
}

export function removeShift(w: World, index: number) {
  const [s] = w.shifts.splice(index, 1);
  if (s) syncRootRecipe(w, [s.target]);
}

export function defaultValue(values: Record<string, number>): string {
  const keys = Object.keys(values);
  return keys[keys.length - 1] ?? "";
}

export function renameKeyInRecord<T>(
  rec: Record<string, T>,
  oldKey: string,
  newKey: string,
): Record<string, T> {
  const out: Record<string, T> = {};
  for (const [k, v] of Object.entries(rec)) out[k === oldKey ? newKey : k] = v;
  return out;
}

export function renameNode(w: World, oldName: string, newName: string, model?: ResearcherModel) {
  if (!(oldName in w.nodes)) return;
  w.nodes = renameKeyInRecord(w.nodes, oldName, newName);
  for (const a of w.arrows) {
    if (a.from === oldName) a.from = newName;
    if (a.to === oldName) a.to = newName;
  }
  for (const s of w.shifts) if (s.target === oldName) s.target = newName;
  for (const t of w.twists) {
    const [p, c] = parseArrowKey(t.arrow);
    t.arrow = `${p === oldName ? newName : p} -> ${c === oldName ? newName : c}`;
  }
  if (model) renameInModel(model, oldName, newName);
}

export function renameAttribute(w: World, oldName: string, newName: string) {
  if (!(oldName in w.attributes)) return;
  w.attributes = renameKeyInRecord(w.attributes, oldName, newName);
  for (const a of Object.values(w.attributes)) if (a.depends_on === oldName) a.depends_on = newName;
  for (const s of w.shifts) if (oldName in s.when) s.when = renameKeyInRecord(s.when, oldName, newName);
  for (const t of w.twists) if (oldName in t.when) t.when = renameKeyInRecord(t.when, oldName, newName);
}

export function renameAttrValue(w: World, attr: string, oldV: string, newV: string) {
  const at = w.attributes[attr];
  if (!at || !(oldV in at.values)) return;
  at.values = renameKeyInRecord(at.values, oldV, newV);
  if (at.table)
    for (const pv of Object.keys(at.table)) at.table[pv] = renameKeyInRecord(at.table[pv], oldV, newV);
  for (const other of Object.values(w.attributes)) {
    if (other.depends_on === attr && other.table) other.table = renameKeyInRecord(other.table, oldV, newV);
  }
  for (const s of w.shifts) if (s.when[attr] === oldV) s.when[attr] = newV;
  for (const t of w.twists) if (t.when[attr] === oldV) t.when[attr] = newV;
}

export function removeNode(w: World, name: string, model?: ResearcherModel) {
  const touched = new Set<string>();
  for (const a of w.arrows) if (a.from === name) touched.add(a.to);
  w.arrows = w.arrows.filter((a) => a.from !== name && a.to !== name);
  w.shifts = w.shifts.filter((s) => s.target !== name);
  w.twists = w.twists.filter((t) => !parseArrowKey(t.arrow).includes(name));
  delete w.nodes[name];
  syncRootRecipe(w, touched);
  if (model) removeFromModel(model, name);
}

export function removeAttribute(w: World, name: string) {
  const touched = new Set<string>();
  for (const s of w.shifts) if (name in s.when) touched.add(s.target);
  w.shifts = w.shifts.filter((s) => !(name in s.when));
  w.twists = w.twists.filter((t) => !(name in t.when));
  for (const a of Object.values(w.attributes)) {
    if (a.depends_on === name) {
      delete a.depends_on;
      delete a.table;
    }
  }
  delete w.attributes[name];
  syncRootRecipe(w, touched);
}

export function renameInModel(m: ResearcherModel, oldName: string, newName: string) {
  m.latents = m.latents.map((x) => (x === oldName ? newName : x));
  m.arrows = m.arrows.map((k) => {
    const [p, c] = parseArrowKey(k);
    return `${p === oldName ? newName : p} -> ${c === oldName ? newName : c}`;
  });
}

export function removeFromModel(m: ResearcherModel, name: string) {
  m.latents = m.latents.filter((x) => x !== name);
  m.arrows = m.arrows.filter((k) => !parseArrowKey(k).includes(name));
}

export function modelWouldCycle(m: ResearcherModel, from: string, to: string): boolean {
  const w = { arrows: m.arrows.map((k) => parseArrowKey(k)).map(([f, t]) => ({ from: f, to: t })) } as {
    arrows: Arrow[];
  };
  return wouldCycle(w as World, from, to);
}

/** Model latents that a Researcher's Model may use: observed, student-level latents of the World. */
export function modelCandidates(w: World): string[] {
  return Object.entries(w.nodes)
    .filter(
      ([, n]) =>
        (n.kind ?? "latent") === "latent" && (n.level ?? "student") === "student" && n.observed !== false,
    )
    .map(([k]) => k);
}

export function categoricalNames(w: World): string[] {
  return Object.keys(w.attributes);
}

/** Parse a number typed by a person: accepts unicode minus, leading dot, commas as decimal point. */
export function parseNumber(text: string): number | null {
  const t = text.trim().replace(/−/g, "-").replace(",", ".");
  if (t === "" || t === "-" || t === ".") return null;
  if (!/^[-+]?(\d+\.?\d*|\.\d+)(e[-+]?\d+)?$/i.test(t)) return null;
  const v = Number(t);
  return Number.isFinite(v) ? v : null;
}

/** Format a number for display with a real minus sign. */
export function fmt(x: number | null | undefined, d = 3): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return "–";
  const v = Math.abs(x) < 0.5 * 10 ** -d ? 0 : x;
  const s = v.toFixed(d);
  return s.startsWith("-") ? `−${s.slice(1)}` : s;
}

/** Compact typed-value formatting (no trailing zeros), e.g. 0.3, -0.4 -> "−0.4". */
export function fmtTyped(x: number | null | undefined): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return "";
  const s = String(Number(x.toPrecision(6)));
  return s.startsWith("-") ? `−${s.slice(1)}` : s;
}
