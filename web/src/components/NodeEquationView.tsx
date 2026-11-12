import type { Derived, Unfolded, World } from "../api/types";
import { useTruth } from "../state/truth";
import { type EquationView, useUI } from "../state/ui";
import { Equation, nodeEquation, pathsEquation, unfoldedEquation } from "./Equation";

/** The World's equation for one node, in the view chosen by the shared switch (paths / compact / primitives). */
export function NodeEquationView({
  name,
  world,
  derived,
  testId,
}: {
  name: string;
  world: World;
  derived?: Derived;
  testId?: string;
}) {
  const view = useUI((s) => s.equationView);
  const { unfolded } = useTruth();
  if (view === "primitives" && unfolded?.equations.roots[name])
    return <Equation eq={unfoldedEquation(unfolded.equations.roots[name])} testId={testId} />;
  if (view === "paths" && unfolded?.equations.direct[name])
    return <Equation eq={pathsEquation(unfolded.equations.direct[name])} testId={testId} />;
  const eq = nodeEquation(world, name, derived);
  return eq ? <Equation eq={eq} testId={testId} /> : null;
}

const VIEWS: { value: EquationView; label: string; title: string }[] = [
  {
    value: "paths",
    label: "Paths",
    title: "Only the latent arrows (and twists), like the fitted equations; level shifts are named",
  },
  { value: "compact", label: "Compact", title: "Each node in terms of its parents, level shifts and twists" },
  {
    value: "primitives",
    label: "Primitives",
    title:
      "Every parent substituted until only attribute indicators and noise sources remain, each factor once",
  },
];

/** The equation-view switch shared by the World and Results tabs. */
export function EquationViewSwitch({ className = "" }: { className?: string }) {
  const view = useUI((s) => s.equationView);
  const set = useUI((s) => s.set);
  const { data, unfolded } = useTruth();
  const serverLacksIt = view !== "compact" && !!data?.ok && !unfolded;
  return (
    <div className={`flex items-center gap-1.5 text-[12px] ${className}`}>
      <span className="text-muted">Equations</span>
      <fieldset className="m-0 inline-flex overflow-hidden rounded-md border border-line p-0">
        <legend className="sr-only">Equation view</legend>
        {VIEWS.map((v) => (
          <label
            key={v.value}
            title={v.title}
            className="cursor-pointer px-2 py-0.5"
            style={view === v.value ? { background: "var(--accent)", color: "var(--accent-fg)" } : undefined}
          >
            <input
              type="radio"
              name="equation-view"
              className="sr-only"
              checked={view === v.value}
              onChange={() => set({ equationView: v.value })}
            />
            {v.label}
          </label>
        ))}
      </fieldset>
      {serverLacksIt ? (
        <span className="badge badge-warn" role="status" data-testid="unfold-server-old">
          the server sent no generating equations: restart it
        </span>
      ) : null}
    </div>
  );
}

/** What the unfolded equations are written in: attributes with their proportions, and the noise sources. */
export function PrimitivesLegend({ unfolded }: { unfolded: Unfolded }) {
  const { attributes, noise } = unfolded.primitives;
  return (
    <div className="mt-2 rounded-md border border-line p-2 text-[11.5px] text-muted" data-testid="primitives">
      <div className="mb-0.5 font-semibold text-text">Primitives</div>
      <ul className="m-0 list-none p-0">
        {attributes.map((a) => (
          <li key={a.name}>
            <code>[{a.name}=value]</code> is 1 when the {a.level}'s {a.name} is that value (
            {Object.entries(a.values)
              .map(([v, p]) => `${v} ${p}`)
              .join(", ")}
            {a.depends_on ? `; drawn given ${a.depends_on}` : ""})
          </li>
        ))}
        {noise.map((n) => (
          <li key={n.symbol}>
            <code>{n.symbol.replace(":", "_")}</code>{" "}
            {n.kind === "e" ? `noise of ${n.node}` : `school random intercept of ${n.node}`}, sd{" "}
            {n.sd === null ? "–" : n.sd.toFixed(3)}
          </li>
        ))}
      </ul>
    </div>
  );
}
