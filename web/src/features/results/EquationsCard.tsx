import type { Finding, RunResult, UnfoldedEq } from "../../api/types";
import {
  Equation,
  formatG,
  modelEquation,
  type NodeEquation,
  pathsEquation,
  subgroupEquation,
  unfoldedEquation,
} from "../../components/Equation";
import { EquationViewSwitch, PrimitivesLegend } from "../../components/NodeEquationView";
import { Empty } from "../../components/ui";
import { useUI } from "../../state/ui";
import { parseArrowKey } from "../../state/worldOps";

const INTRO = {
  paths:
    "Paths: only the latent arrows, as the fitted equations have them; the twists stay because they change a slope, the level shifts are only named.",
  compact: "Compact: each node in terms of its parents, level shifts and twists.",
  primitives:
    "Primitives: the truth with every parent substituted until only attribute indicators and noise sources remain; the fits unfolded the same way into the model's starting latents and each latent's residual, so every coefficient is a total effect.",
};

/** What generated the data (the World's equations) next to what the Researcher's Model fitted. */
export function EquationsCard({ r, finding }: { r: RunResult; finding: Finding | null }) {
  const view = useUI((s) => s.equationView);
  const gen = r.generating;
  const fu = r.fitted_unfolded;
  const children = [...new Set(r.spec.model.arrows.map((k) => parseArrowKey(k)[1]))];
  const columns = finding
    ? ["generating (truth)", "fitted: full sample", `fitted: ${finding.description}`, "fitted: complement"]
    : ["generating (truth)", "fitted: full sample"];
  const fullSample = (child: string) => {
    const rows = r.estimation.filter((e) => parseArrowKey(e.path)[1] === child);
    return rows.length
      ? modelEquation(
          child,
          rows.map((e) => parseArrowKey(e.path)[0]),
          rows.map((e) => e.estimate),
        )
      : null;
  };
  const truth = (child: string): NodeEquation | null => {
    if (!gen) return null;
    if (view === "primitives")
      return gen.equations.roots[child] ? unfoldedEquation(gen.equations.roots[child]) : null;
    const d = gen.equations.direct[child];
    if (!d) return null;
    return view === "paths" ? pathsEquation(d) : unfoldedEquation(d);
  };
  const fitted = (child: string): (NodeEquation | null)[] => {
    if (view === "primitives") {
      const un = (eq: UnfoldedEq | undefined) => (eq ? unfoldedEquation(eq) : null);
      const sides = finding ? fu?.findings[String(finding.rank)] : undefined;
      return [
        un(fu?.full[child]),
        ...(finding ? [un(sides?.subgroup[child]), un(sides?.complement[child])] : []),
      ];
    }
    return [
      fullSample(child),
      ...(finding
        ? [subgroupEquation(child, finding.paths, "b_g"), subgroupEquation(child, finding.paths, "b_b")]
        : []),
    ];
  };
  return (
    <section
      className="card min-w-0 p-3"
      aria-label="Generating and fitted equations"
      data-testid="equations-card"
    >
      <div className="mb-1 flex flex-wrap items-center gap-3">
        <h2 className="m-0 font-semibold text-[14px]">
          Equations · what generated the data vs what the model fitted
        </h2>
        <span className="flex-1" />
        <EquationViewSwitch />
      </div>
      <p className="mt-0 mb-2 text-[12px] text-muted">
        The generating equations are the World that made this run's data. {INTRO[view]} The gap between the
        truth and the fit is where spurious findings live.
      </p>
      {!gen ? (
        <Empty>This run was stored without its generating equations; run it again to see them.</Empty>
      ) : (
        <div className="overflow-x-auto">
          <div
            className="grid items-start gap-x-4 gap-y-2 text-[12px]"
            style={{ gridTemplateColumns: `repeat(${columns.length}, minmax(260px, 1fr))` }}
          >
            {columns.map((c) => (
              <div key={c} className="font-semibold text-muted">
                {c}
              </div>
            ))}
            {children.flatMap((child) =>
              [truth(child), ...fitted(child)].map((eq, i) => (
                <div key={`${child}-${columns[i]}`} className="min-w-0">
                  {eq ? (
                    <Equation
                      eq={eq}
                      testId={i === 0 ? `generating-eq-${child}` : i === 1 ? `fitted-eq-${child}` : undefined}
                    />
                  ) : (
                    "–"
                  )}
                </div>
              )),
            )}
          </div>
          {view === "primitives" ? (
            <>
              <PrimitivesLegend unfolded={gen} />
              <p className="mt-1 mb-0 text-[11.5px] text-muted">
                Fitted primitives: the model's starting latents (
                {r.spec.model.latents.filter((l) => !children.includes(l)).join(", ")}) and each latent's
                residual <code>e_x</code>; the model has no attributes, so their effects end up in the
                residuals.
                {!fu ? " This run was stored without them; run it again." : ""}
              </p>
            </>
          ) : null}
          <p className="mt-2 mb-0 text-[11.5px] text-faint">
            Fitted coefficients are on the analysis scale (standardized once, or within each group under{" "}
            <code>current</code>); generating coefficients are in the World's raw units. Noise sd:{" "}
            {gen.primitives.noise
              .filter((n) => n.kind === "e")
              .map((n) => `${n.node} ${n.sd === null ? "–" : formatG(n.sd, 3)}`)
              .join(", ")}
            .
          </p>
        </div>
      )}
    </section>
  );
}
