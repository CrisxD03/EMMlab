import { Term, Tip } from "../../components/ui";

const TIER_TEXT: Record<string, string> = {
  ok: "ok: at least 50 effective schools, enough clusters for its interval to be trusted",
  fragile: "fragile: 20–49 effective schools; the interval may be too narrow",
  unreliable: "unreliable: fewer than 20 effective schools; do not trust the interval",
};

/** Cluster tier of a group (effective number of schools), with its meaning on hover. */
export function TierBadge({ tier }: { tier: string | undefined }) {
  if (!tier) return null;
  const cls = tier === "ok" ? "badge-good" : tier === "fragile" ? "badge-warn" : "badge-bad";
  const term = tier === "fragile" ? "Fragile" : tier === "unreliable" ? "Unreliable" : null;
  return (
    <span className={`badge ${cls}`} data-testid="tier">
      {term ? (
        <Term term={term}>{tier}</Term>
      ) : (
        <Tip content={TIER_TEXT[tier] ?? tier}>
          <span>{tier}</span>
        </Tip>
      )}
    </span>
  );
}

/** A slice that could not be fitted reliably. */
export function TooSmallBadge() {
  return (
    <span className="badge badge-bad ml-1" data-testid="too-small">
      <Tip content="too small: fewer than 20 students, or the regression could not be fitted on this slice">
        <span>too small</span>
      </Tip>
    </span>
  );
}

/** One line under every table that shows tiers. */
export function TierLegend() {
  return (
    <p className="mt-1 mb-0 text-[11.5px] text-faint" data-testid="tier-legend">
      Clusters: ok ≥ 50 effective schools · fragile 20–49 · unreliable &lt; 20 · too small: fewer than 20
      students or the fit failed.
    </p>
  );
}
