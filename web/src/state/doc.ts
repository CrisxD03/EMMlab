// The document: World + Researcher's Model + method + search space + seed. Every edit goes through
// `apply`, which records an undo snapshot (U6) and the kind of change (for the ripple, U3).

import { type Draft, produce } from "immer";
import { create } from "zustand";
import type { MethodPreset, ResearcherModel, RunSpec, SearchSpace, World } from "../api/types";
import { emptyWorld } from "./worldOps";

export interface Doc {
  world: World;
  model: ResearcherModel;
  method: MethodPreset;
  search: SearchSpace;
  seed: number;
}

/** What kind of change an edit is. "slope" and "spread" drive the two ripple styles (U3). */
export type EditKind = "slope" | "spread" | "structure" | "other" | "load" | "yaml" | "history";

export const HISTORY_LIMIT = 200;

export const RECOMMENDED: MethodPreset = {
  name: "recommended",
  analysis: {
    normalisation: "none",
    standardize: "global_once",
    residualize_on: [],
    residualize_method: "total_effect",
    se: "cluster",
  },
  emm: {
    quality: "ent_z",
    baseline: "complement",
    paths: "all",
    algorithm: "beam",
    beam_width: 20,
    top_k: 10,
    interval_level: 0.95,
    exact_bootstrap_reps: 99,
    nominal_min_schools: 25,
    interval_se: "analysis",
    cover_selection: true,
    prune_refinements: true,
  },
};

export const DEFAULT_SEARCH: SearchSpace = {
  binning: {},
  default_binning: { n_bins: 5, one_sided: true, weighting: "students" },
  depth: 2,
  min_n: 100,
  min_n_complement: 100,
  min_schools: 0,
};

export function blankDoc(): Doc {
  return {
    world: emptyWorld(),
    model: { name: "model", latents: [], arrows: [] },
    method: structuredClone(RECOMMENDED),
    search: structuredClone(DEFAULT_SEARCH),
    seed: 0,
  };
}

export interface LastEdit {
  kind: EditKind;
  label: string;
  seq: number;
}

interface DocState {
  doc: Doc;
  past: Doc[];
  future: Doc[];
  lastEdit: LastEdit;
  apply: (label: string, kind: EditKind, recipe: (d: Draft<Doc>) => void) => void;
  load: (doc: Doc, label?: string) => void;
  undo: () => void;
  redo: () => void;
  reset: (doc?: Doc) => void;
}

const STORAGE_KEY = "emmlab.doc.v1";

export function savedDoc(): Doc | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as Doc) : null;
  } catch {
    return null;
  }
}

export const useDoc = create<DocState>((set, get) => ({
  doc: savedDoc() ?? blankDoc(),
  past: [],
  future: [],
  lastEdit: { kind: "load", label: "start", seq: 0 },
  apply: (label, kind, recipe) => {
    const { doc, past, lastEdit } = get();
    const next = produce(doc, recipe);
    if (next === doc) return;
    set({
      doc: next,
      past: [...past, doc].slice(-HISTORY_LIMIT),
      future: [],
      lastEdit: { kind, label, seq: lastEdit.seq + 1 },
    });
  },
  load: (doc, label = "load") => {
    const { doc: cur, past, lastEdit } = get();
    set({
      doc,
      past: [...past, cur].slice(-HISTORY_LIMIT),
      future: [],
      lastEdit: { kind: "load", label, seq: lastEdit.seq + 1 },
    });
  },
  undo: () => {
    const { doc, past, future, lastEdit } = get();
    if (!past.length) return;
    set({
      doc: past[past.length - 1],
      past: past.slice(0, -1),
      future: [doc, ...future],
      lastEdit: { kind: "history", label: "undo", seq: lastEdit.seq + 1 },
    });
  },
  reset: (doc) =>
    set({ doc: doc ?? blankDoc(), past: [], future: [], lastEdit: { kind: "load", label: "reset", seq: 0 } }),
  redo: () => {
    const { doc, past, future, lastEdit } = get();
    if (!future.length) return;
    set({
      doc: future[0],
      past: [...past, doc],
      future: future.slice(1),
      lastEdit: { kind: "history", label: "redo", seq: lastEdit.seq + 1 },
    });
  },
}));

let saveTimer: ReturnType<typeof setTimeout> | undefined;
useDoc.subscribe((s) => {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(s.doc));
    } catch {
      /* storage unavailable: the document still works in memory */
    }
  }, 300);
});

export function runSpecOf(doc: Doc): RunSpec {
  return { world: doc.world, model: doc.model, method: doc.method, search: doc.search, seed: doc.seed };
}

export function clearSavedDoc() {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch {
    /* ignore */
  }
}
