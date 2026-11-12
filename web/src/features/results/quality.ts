import { useQuery } from "@tanstack/react-query";
import { api } from "../../api/client";
import type { AnalysisConfig, EMMConfig, FindingPath, QualityInfo, QualityName } from "../../api/types";

/** Every quality function as the server defines it (emm.QUALITY_FUNCTIONS); fetched once. */
export function useQualities() {
  return useQuery({ queryKey: ["qualities"], queryFn: api.qualities, staleTime: Number.POSITIVE_INFINITY });
}

/** One quality function's description and deviation rule (undefined while loading). */
export function useQualityInfo(name: QualityName): QualityInfo | undefined {
  return useQualities().data?.find((q) => q.name === name);
}

export const usesIntervals = (info: QualityInfo | undefined) => !!info?.uses_intervals;

/** A path deviates when the quality function's own test says so: disjoint intervals for `current`, |z| ≥ 1.96 otherwise. */
export const isDeviating = (p: FindingPath, info: QualityInfo | undefined) =>
  usesIntervals(info) ? p.disjoint : Math.abs(p.z) >= 1.96;

export const deviatingRule = (info: QualityInfo | undefined, level: number) =>
  usesIntervals(info)
    ? `the ${Math.round(level * 100)}% intervals of subgroup and complement do not overlap`
    : "the difference is significant at 95% (|z| ≥ 1.96)";

/** Which standard errors the intervals behind the quality function use. */
export const intervalSeLabel = (emm: EMMConfig, analysis: AnalysisConfig) =>
  emm.interval_se === "naive"
    ? "student-level (naive) standard errors, as plspm reports them"
    : analysis.se === "cluster"
      ? "cluster-robust standard errors (schools are the clusters)"
      : `${analysis.se} standard errors`;
