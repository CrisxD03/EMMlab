// Live truth for the canvas (U2): the World is posted to /api/world/truth ~80 ms after the last edit;
// every derived number on screen reads from here. Comparing successive responses gives the ripple:
// which derived values changed, and whether the edit was a slope change or a spread/noise change (U3).

import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { createContext, type ReactNode, useContext, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api/client";
import type { Derived, Issue, TruthResponse, Unfolded } from "../api/types";
import { type EditKind, useDoc } from "./doc";
import { type RippleKind, useUI } from "./ui";

export interface TruthState {
  /** Latest response (may have issues). */
  data?: TruthResponse;
  /** Derived values of the latest valid World (stale when the current World has issues). */
  derived?: Derived;
  equations: string[];
  /** Generating equations of the latest valid World, in direct and unfolded (roots) form. */
  unfolded?: Unfolded;
  hash?: string;
  issues: Issue[];
  stale: boolean;
  pending: boolean;
}

const TruthCtx = createContext<TruthState>({ equations: [], issues: [], stale: false, pending: false });

export const useTruth = () => useContext(TruthCtx);

export const DEBOUNCE_MS = 80;

function changed(a: number | undefined, b: number | undefined): boolean {
  if (a === undefined || b === undefined) return a !== b;
  return Math.abs(a - b) > 1e-9;
}

export function computeRipple(
  prev: Derived | undefined,
  next: Derived,
  kind: EditKind,
): Record<string, RippleKind> {
  const out: Record<string, RippleKind> = {};
  if (!prev) return out;
  const tag: RippleKind =
    kind === "slope" || kind === "structure" ? "slope" : kind === "spread" ? "spread" : "change";
  for (const [key, n] of Object.entries(next.arrows)) {
    const p = prev.arrows[key];
    const groups = Object.keys(n.groups);
    const raw =
      !p ||
      changed(p.avg_slope, n.avg_slope) ||
      changed(p.avg_total, n.avg_total) ||
      changed(p.baseline_slope, n.baseline_slope) ||
      groups.some(
        (g) =>
          changed(p.groups[g]?.slope, n.groups[g].slope) || changed(p.groups[g]?.total, n.groups[g].total),
      );
    const std = !p || changed(p.std, n.std) || groups.some((g) => changed(p.groups[g]?.std, n.groups[g].std));
    if (raw) out[`arrow:${key}:raw`] = tag;
    if (std) out[`arrow:${key}:std`] = tag;
  }
  // A spread/noise change leaves every raw relationship alone: only standardized coefficients ripple.
  if (tag === "spread") {
    for (const k of Object.keys(out)) if (!k.endsWith(":std")) delete out[k];
    return out;
  }
  for (const [name, n] of Object.entries(next.nodes)) {
    const p = prev.nodes[name];
    for (const f of ["mean", "sd", "r2", "icc"] as const) {
      if (!p || changed(p[f], n[f])) out[`node:${name}:${f}`] = tag;
    }
  }
  for (const [key, v] of Object.entries(next.total_effects)) {
    if (changed(prev.total_effects[key], v)) out[`total:${key}`] = tag;
  }
  return out;
}

export function TruthProvider({ children }: { children: ReactNode }) {
  const world = useDoc((s) => s.doc.world);
  const key = useMemo(() => JSON.stringify(world), [world]);
  const [debKey, setDebKey] = useState(key);
  // Leading-edge debounce: a single edit is sent at once; a burst of edits is coalesced (≤ 1 request per 80 ms).
  const lastSent = useRef(0);
  useEffect(() => {
    if (key === debKey) return;
    const wait = Math.max(0, DEBOUNCE_MS - (performance.now() - lastSent.current));
    const t = setTimeout(() => {
      lastSent.current = performance.now();
      setDebKey(key);
    }, wait);
    return () => clearTimeout(t);
  }, [key, debKey]);

  const q = useQuery({
    queryKey: ["truth", debKey],
    queryFn: () => api.truth(JSON.parse(debKey)),
    placeholderData: keepPreviousData,
    staleTime: Number.POSITIVE_INFINITY,
    gcTime: 120_000,
  });

  const lastOk = useRef<TruthResponse | undefined>(undefined);
  const prevDerived = useRef<Derived | undefined>(undefined);
  const data = q.data;
  if (data?.ok) lastOk.current = data;

  useEffect(() => {
    if (!data?.ok || !data.derived) return;
    if (prevDerived.current === data.derived) return;
    const kind = useDoc.getState().lastEdit.kind;
    const kinds = computeRipple(prevDerived.current, data.derived, kind);
    prevDerived.current = data.derived;
    const r = useUI.getState().ripple;
    useUI.getState().set({ ripple: { seq: r.seq + 1, editKind: kind, kinds } });
  }, [data]);

  const value = useMemo<TruthState>(() => {
    const ok = lastOk.current;
    return {
      data,
      derived: ok?.derived,
      equations: data?.ok ? (data.equations ?? []) : (ok?.equations ?? []),
      unfolded: data?.ok ? data.unfolded : ok?.unfolded,
      hash: data?.ok ? data.hash : undefined,
      issues: data && !data.ok ? data.issues : [],
      stale: !!data && !data.ok,
      pending: key !== debKey || q.isFetching,
    };
  }, [data, key, debKey, q.isFetching]);

  return <TruthCtx.Provider value={value}>{children}</TruthCtx.Provider>;
}
