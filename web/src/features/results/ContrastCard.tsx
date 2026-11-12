import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { api } from "../../api/client";
import type {
  ContrastCompare,
  ContrastFit,
  ContrastResult,
  Finding,
  RunResult,
  VerdictKind,
} from "../../api/types";
import { Empty, Term } from "../../components/ui";
import { fmt } from "../../state/worldOps";
import { deviatingRule, intervalSeLabel, useQualityInfo } from "./quality";
import { TierBadge, TierLegend, TooSmallBadge } from "./TierBadge";

const KIND_LABEL: Record<VerdictKind, string> = {
  conjunction_specific: "conjunction-specific",
  carried_by: "carried by part of it",
  shared: "not isolated",
  single: "single condition",
};
const KIND_BADGE: Record<VerdictKind, string> = {
  conjunction_specific: "badge-twist",
  carried_by: "badge-warn",
  shared: "badge-good",
  single: "",
};

const interval = (f: ContrastFit | null | undefined) =>
  f ? `${fmt(f.b)} [${fmt(f.ci[0])}, ${fmt(f.ci[1])}]` : "–";

function Cell({ fit, cmp }: { fit: ContrastFit | null | undefined; cmp?: ContrastCompare | null }) {
  return (
    <td className="r num" data-deviating={cmp?.deviating ? "true" : "false"}>
      {interval(fit)}
      {cmp?.deviating ? (
        <span className="badge badge-twist ml-1" title={`differs from the finding (z = ${fmt(cmp.z, 1)})`}>
          ≠
        </span>
      ) : null}
    </td>
  );
}

function Verdicts({ c }: { c: ContrastResult }) {
  if (!c.focal_paths.length)
    return (
      <p className="m-0 mb-2 text-[12.5px] text-muted" data-testid="contrast-none">
        No path of this subgroup deviates from its {c.baseline} under the rule above, so there is no deviation
        to decompose.
      </p>
    );
  return (
    <ul className="m-0 mb-3 flex list-none flex-col gap-1.5 p-0" data-testid="contrast-verdicts">
      {c.focal_paths.map((p) => {
        const v = c.verdicts[p];
        return (
          <li key={p} className="flex flex-wrap items-baseline gap-x-2 text-[12.5px]" data-verdict={v.kind}>
            <code>{p}</code>
            <span className={`badge ${KIND_BADGE[v.kind]}`}>{KIND_LABEL[v.kind]}</span>
            <span>{v.text}</span>
            {v.note ? (
              <span className="text-warn" data-testid="contrast-note">
                {v.note}
              </span>
            ) : null}
            {v.low_power ? (
              <span className="text-warn">
                Some slices are small, so “not deviating” may only mean “not enough data”.
              </span>
            ) : null}
          </li>
        );
      })}
    </ul>
  );
}

/** The finding decomposed: every sign pattern of its conditions, the complement and each condition on its own. */
export function ContrastCard({ r, finding }: { r: RunResult; finding: Finding }) {
  const q = useQuery({
    queryKey: ["contrast", r.run_key, finding.rank],
    queryFn: () => api.contrast(r.run_key, finding.rank),
  });
  const [showAll, setShowAll] = useState(false);
  const c = q.data;
  const emm = r.spec.method.emm;
  const qinfo = useQualityInfo(emm.quality);
  return (
    <section className="card min-w-0 p-3" aria-label="Contrastive analysis" data-testid="contrast">
      <h2 className="mt-0 mb-1 font-semibold text-[14px]">
        <Term term="Contrastive analysis">Contrastive analysis</Term> · <code>{finding.description}</code>
      </h2>
      <p className="mt-0 mb-2 text-[12px] text-muted">
        Which part of the conjunction carries the deviation? The Researcher's Model is refitted on the finding
        and on the slices that flip one condition at a time; a ≠ marks a path where the slice differs from the
        finding ({deviatingRule(qinfo, emm.interval_level)}). Not a new search.
      </p>
      {q.isError ? <p className="text-bad">{String((q.error as Error).message)}</p> : null}
      {!c ? (
        <Empty>Computing the slices…</Empty>
      ) : (
        <ContrastBody c={c} showAll={showAll} setShowAll={setShowAll} r={r} />
      )}
    </section>
  );
}

function ContrastBody({
  c,
  showAll,
  setShowAll,
  r,
}: {
  c: ContrastResult;
  showAll: boolean;
  setShowAll: (v: boolean) => void;
  r: RunResult;
}) {
  const rows = showAll || !c.focal_paths.length ? c.paths : c.focal_paths;
  return (
    <>
      <Verdicts c={c} />
      <div className="overflow-x-auto">
        <table className="data" data-testid="contrast-table">
          <thead>
            <tr>
              <th>path</th>
              {c.slices.map((s) => (
                <th key={s.key} className="r" data-slice={s.key}>
                  {s.label}
                  {s.role === "finding" ? <span className="badge badge-twist ml-1">finding</span> : null}
                  {s.role === "complement" ? <span className="badge ml-1">EMM baseline</span> : null}
                </th>
              ))}
            </tr>
            <tr>
              <th className="font-normal text-muted">students · schools</th>
              {c.slices.map((s) => (
                <th key={s.key} className="r font-normal text-muted">
                  {s.n} · {s.schools}
                  {s.tier !== "ok" ? (
                    <span className="ml-1">
                      <TierBadge tier={s.tier} />
                    </span>
                  ) : null}
                  {!s.reliable ? <TooSmallBadge /> : null}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((p) => (
              <tr key={p} data-path={p} data-focal={c.focal_paths.includes(p) ? "true" : "false"}>
                <td>{p}</td>
                {c.slices.map((s) => (
                  <Cell key={s.key} fit={s.fits[p]} cmp={s.vs_finding?.[p]} />
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {c.focal_paths.length && c.paths.length > c.focal_paths.length ? (
        <label className="mt-1.5 flex items-center gap-1.5 text-[12px]">
          <input type="checkbox" checked={showAll} onChange={(e) => setShowAll(e.target.checked)} />
          Show all {c.paths.length} paths
        </label>
      ) : null}
      {c.singles.length ? (
        <div className="mt-3 overflow-x-auto" data-testid="contrast-singles">
          <div className="mb-1 font-semibold text-[12.5px]">
            Each condition on its own, against everyone else
          </div>
          <table className="data">
            <thead>
              <tr>
                <th>path</th>
                {c.singles.map((s) => (
                  <th key={s.condition} colSpan={2} className="r">
                    {s.condition} · {s.alone.n} students
                  </th>
                ))}
              </tr>
              <tr>
                <th />
                {c.singles.map((s) => (
                  <th key={s.condition} colSpan={2} className="r font-normal text-muted">
                    alone vs the rest
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((p) => (
                <tr key={p}>
                  <td>{p}</td>
                  {c.singles.map((s) => (
                    <SingleCells key={s.condition} s={s} path={p} />
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
      <TierLegend />
      <p className="mt-2 mb-0 text-[11.5px] text-faint">
        Wald {Math.round(c.interval_level * 100)}% intervals with{" "}
        {intervalSeLabel(r.spec.method.emm, r.spec.method.analysis)}. “Too small”: fewer than 20 students or a
        fit that failed. A condition shows as needed when dropping it removes the deviation; a slice with few
        schools has wide intervals, so read “not deviating” together with its size.
      </p>
    </>
  );
}

function SingleCells({ s, path }: { s: ContrastResult["singles"][number]; path: string }) {
  const cmp = s.vs_rest[path];
  return (
    <>
      <td className="r num">{interval(s.alone.fits[path])}</td>
      <td className="r num" data-deviating={cmp?.deviating ? "true" : "false"}>
        {interval(s.rest.fits[path])}
        {cmp?.deviating ? <span className="badge badge-twist ml-1">≠</span> : null}
      </td>
    </>
  );
}
