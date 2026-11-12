// Turning server validation issues into actionable UI (UX6): which element they name, how to show it
// on the canvas, and a one-click fix where one is obvious.

import type { Issue, World } from "../../api/types";
import { useDoc } from "../../state/doc";
import type { Selection } from "../../state/ui";
import { useUI } from "../../state/ui";
import { arrowKey, parseArrowKey, removeArrow, syncRootRecipe } from "../../state/worldOps";
import { editWorld } from "./actions";

export interface IssueTarget {
  selection: Selection | null;
  focusField?: string;
  highlight?: string[];
  /** Human-readable name of the element the issue is about. */
  label: string;
  yaml?: boolean;
}

export interface IssueFix {
  label: string;
  run: () => void;
}

const idx = (s: string) => Number(s);

export function issueTarget(issue: Issue, w: World): IssueTarget {
  const el = issue.element;
  let m = el.match(/^nodes\.([^.[\s]+)(?:\.(.+))?$/);
  if (m) {
    const [, name, rest] = m;
    let focus = rest;
    if (rest === "target") {
      const t = w.nodes[name]?.target ?? {};
      focus = t.sd !== undefined ? "target.sd" : t.r2 !== undefined ? "target.r2" : "target.icc";
    }
    return { selection: { type: "node", id: name }, focusField: focus, label: `node “${name}”` };
  }
  m = el.match(/^attributes\.([^.[\s]+)(?:\.(.+))?$/);
  if (m)
    return { selection: { type: "attribute", id: m[1] }, focusField: m[2], label: `attribute “${m[1]}”` };
  if (el === "attributes") {
    const names = (issue.message.split(":")[1] ?? "")
      .split("->")
      .map((x) => x.trim())
      .filter(Boolean);
    return {
      selection: names[0] ? { type: "attribute", id: names[0] } : null,
      highlight: names,
      label: "attribute dependencies",
    };
  }
  m = el.match(/^arrows(?:\[(\d+)\]|\.(\d+))/);
  if (m) {
    const i = idx(m[1] ?? m[2]);
    const keyFromEl = el.match(/\((.+)\)/)?.[1];
    const a = w.arrows[i];
    const key = keyFromEl ?? (a ? arrowKey(a) : undefined);
    return {
      selection: key ? { type: "arrow", id: key } : null,
      focusField: "slope",
      label: `arrow ${i + 1}${key ? ` (${key})` : ""}`,
    };
  }
  if (el === "arrows") {
    const names = cycleNodes(issue.message);
    const closing = names.length ? `${names[names.length - 1]} -> ${names[0]}` : undefined;
    return {
      selection: closing ? { type: "arrow", id: closing } : null,
      highlight: names,
      label: names.length ? `cycle ${names.join(" → ")}` : "arrows",
    };
  }
  m = el.match(/^shifts(?:\[(\d+)\]|\.(\d+))/);
  if (m) {
    const i = idx(m[1] ?? m[2]);
    return { selection: { type: "shift", index: i }, focusField: "delta", label: `level shift ${i + 1}` };
  }
  m = el.match(/^twists(?:\[(\d+)\]|\.(\d+))/);
  if (m) {
    const i = idx(m[1] ?? m[2]);
    const t = w.twists[i];
    return { selection: { type: "twist", index: i }, label: `twist ${i + 1}${t ? ` (${t.arrow})` : ""}` };
  }
  if (el === "yaml") return { selection: null, yaml: true, label: "YAML text" };
  if (el in w.nodes) return { selection: { type: "node", id: el }, label: `“${el}”` };
  if (el in w.attributes) return { selection: { type: "attribute", id: el }, label: `“${el}”` };
  return { selection: { type: "world" }, label: el || "World" };
}

function cycleNodes(message: string): string[] {
  const part = message.split(":").slice(1).join(":");
  return part
    .split("->")
    .map((x) => x.trim())
    .filter(Boolean);
}

export function showIssue(issue: Issue) {
  const w = useDoc.getState().doc.world;
  const t = issueTarget(issue, w);
  const ui = useUI.getState();
  if (t.yaml) {
    ui.set({ yamlOpen: true, tab: "world" });
    return;
  }
  ui.set({ tab: "world" });
  if (t.selection) ui.select(t.selection, t.focusField ?? null);
  if (t.highlight?.length) ui.set({ highlight: t.highlight });
}

export function issueFix(issue: Issue, w: World): IssueFix | null {
  const el = issue.element;
  const msg = issue.message;
  const node = el.match(/^nodes\.([^.[\s]+)/)?.[1];
  if (node && msg.includes("t_min =")) {
    const tmin = Number(msg.match(/t_min = ([\d.eE+-]+)/)?.[1]);
    if (Number.isFinite(tmin)) {
      const v = Math.ceil(tmin * 1.05 * 1000) / 1000;
      return {
        label: `Raise target sd to ${v}`,
        run: () =>
          editWorld("fix target", "spread", (w) => {
            const n = w.nodes[node];
            if (n?.target) n.target.sd = v;
          }),
      };
    }
  }
  if (node && msg.includes("so it is a Child: give a recipe")) {
    return {
      label: "Convert root to recipe",
      run: () =>
        editWorld("convert to recipe", "structure", (w) => {
          const n = w.nodes[node];
          if (!n) return;
          n.recipe = n.recipe ?? { intercept: n.root?.mean ?? 0, noise_sd: n.root?.sd ?? 1 };
          delete n.root;
        }),
    };
  }
  if (node && msg.includes("so it is a Root: give a root distribution")) {
    return {
      label: "Convert recipe to root distribution",
      run: () =>
        editWorld("convert to root", "structure", (w) => {
          const n = w.nodes[node];
          if (!n) return;
          delete n.recipe;
          syncRootRecipe(w, [node]);
        }),
    };
  }
  if (node && (msg.includes("has no recipe") || msg.includes("has no root distribution"))) {
    return {
      label: "Add the missing recipe/distribution",
      run: () => editWorld("fix node", "structure", (w) => syncRootRecipe(w, [node])),
    };
  }
  const attr = el.match(/^attributes\.([^.[\s]+)$/)?.[1];
  if (attr && msg.includes("proportions sum to")) {
    return {
      label: "Normalize proportions",
      run: () =>
        editWorld("normalize proportions", "other", (w) => {
          const a = w.attributes[attr];
          if (!a) return;
          const s = Object.values(a.values).reduce((x, y) => x + y, 0) || 1;
          for (const k of Object.keys(a.values)) a.values[k] = Math.round((a.values[k] / s) * 1e6) / 1e6;
        }),
    };
  }
  if (el === "arrows" && msg.includes("cycle")) {
    const names = cycleNodes(msg);
    if (names.length >= 2) {
      const key = `${names[names.length - 1]} -> ${names[0]}`;
      return {
        label: `Remove arrow ${key}`,
        run: () => editWorld("break cycle", "structure", (w) => removeArrow(w, key)),
      };
    }
  }
  const tw = el.match(/^twists(?:\[(\d+)\]|\.(\d+))/);
  if (tw) {
    const i = Number(tw[1] ?? tw[2]);
    return {
      label: "Remove this twist",
      run: () =>
        editWorld("remove twist", "slope", (w) => {
          w.twists.splice(i, 1);
        }),
    };
  }
  const ar = el.match(/^arrows\[(\d+)\] \((.+)\)$/);
  if (ar && (msg.includes("school-level") || msg.includes("cannot be caused") || msg.includes("duplicate"))) {
    const key = ar[2];
    const [p, c] = parseArrowKey(key);
    if (w.nodes[p] && w.nodes[c])
      return {
        label: `Remove arrow ${key}`,
        run: () => editWorld("remove arrow", "structure", (w) => removeArrow(w, key)),
      };
  }
  const sh = el.match(/^shifts\[(\d+)\]$/);
  if (sh) {
    const i = Number(sh[1]);
    return {
      label: "Remove this level shift",
      run: () =>
        editWorld("remove level shift", "structure", (w) => {
          const [s] = w.shifts.splice(i, 1);
          if (s) syncRootRecipe(w, [s.target]);
        }),
    };
  }
  return null;
}

/** Issues that concern one element (for inline display in the inspector). */
export function issuesFor(
  issues: Issue[],
  kind: "node" | "attribute" | "arrow" | "shift" | "twist",
  id: string | number,
  w: World,
) {
  return issues.filter((i) => {
    const t = issueTarget(i, w).selection;
    if (!t) return false;
    if (kind === "node" || kind === "attribute") return t.type === kind && t.id === id;
    if (kind === "arrow") return t.type === "arrow" && t.id === id;
    if (kind === "shift") return t.type === "shift" && t.index === id;
    return t.type === "twist" && t.index === id;
  });
}
