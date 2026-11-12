import { create } from "zustand";
import type { Issue, RunResult } from "../api/types";
import type { Direction, Route } from "../features/world/layout";

export type Tab = "world" | "model" | "search" | "results" | "experiments";

export type Selection =
  | { type: "world" }
  | { type: "node"; id: string }
  | { type: "attribute"; id: string }
  | { type: "arrow"; id: string } // "from -> to"
  | { type: "shift"; index: number }
  | { type: "twist"; index: number };

export type Theme = "system" | "light" | "dark";
export type EquationView = "paths" | "compact" | "primitives";
export type RippleKind = "slope" | "spread" | "change";

export interface Ripple {
  seq: number;
  editKind: string;
  /** element key -> kind: "arrow:<key>:raw", "arrow:<key>:std", "node:<name>:sd", "total:<key>" … */
  kinds: Record<string, RippleKind>;
}

interface UIState {
  tab: Tab;
  selection: Selection;
  /** Field to focus in the inspector after "show on canvas" or after creating an element. */
  focusField: string | null;
  theme: Theme;
  yamlOpen: boolean;
  galleryOpen: boolean;
  glossaryOpen: boolean;
  positions: Record<string, { x: number; y: number }>;
  /** ELK edge routes of the last layout (keyed by arrow key); stale routes fall back to curves. */
  routes: Record<string, Route>;
  /** Direction of the last auto-layout; handles follow it (left→right or top→down). */
  layoutDir: Direction;
  /** Level-shift edges on the canvas: auto = drawn when there are few, otherwise badges on nodes. */
  shiftEdges: "auto" | "show" | "hide";
  layoutSeq: number;
  /** Redraw the next layout in the current direction (set by display toggles so they do not flip it). */
  keepLayoutDir: boolean;
  /** Researcher's Model canvas: label each arrow with its expected whole-population coefficient. */
  modelCoefs: boolean;
  /** How equations are written (World and Results tabs share this): latent paths only, compact, or primitives. */
  equationView: EquationView;
  ripple: Ripple;
  run: RunResult | null;
  running: boolean;
  runIssues: Issue[];
  selectedFinding: number; // rank, 0 = full sample
  experimentId: string | null;
  highlight: string[]; // node names highlighted by "show on canvas" (e.g. a cycle)
  set: (patch: Partial<UIState>) => void;
  select: (s: Selection, focusField?: string | null) => void;
}

function initialTheme(): Theme {
  try {
    const t = localStorage.getItem("emmlab.theme");
    if (t === "light" || t === "dark" || t === "system") return t;
  } catch {
    /* ignore */
  }
  return "system";
}

export const useUI = create<UIState>((set) => ({
  tab: "world",
  selection: { type: "world" },
  focusField: null,
  theme: initialTheme(),
  yamlOpen: false,
  galleryOpen: false,
  glossaryOpen: false,
  positions: {},
  routes: {},
  layoutDir: "RIGHT",
  shiftEdges: "auto",
  layoutSeq: 0,
  keepLayoutDir: false,
  modelCoefs: true,
  equationView: "paths",
  ripple: { seq: 0, editKind: "load", kinds: {} },
  run: null,
  running: false,
  runIssues: [],
  selectedFinding: 1,
  experimentId: null,
  highlight: [],
  set: (patch) => set(patch),
  select: (selection, focusField = null) => set({ selection, focusField, highlight: [] }),
}));

export function applyTheme(theme: Theme) {
  const root = document.documentElement;
  if (theme === "system") delete root.dataset.theme;
  else root.dataset.theme = theme;
  try {
    localStorage.setItem("emmlab.theme", theme);
  } catch {
    /* ignore */
  }
}

export const SHIFT_EDGES_AUTO_MAX = 4;
export function shiftEdgesShown(setting: UIState["shiftEdges"], nShifts: number) {
  return setting === "show" || (setting === "auto" && nShifts <= SHIFT_EDGES_AUTO_MAX);
}
