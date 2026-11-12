import * as Plot from "@observablehq/plot";
import { useEffect, useRef } from "react";

/** Renders an Observable Plot figure; re-renders when `deps` change (compared by value). */
export function PlotFigure({
  make,
  deps,
  label,
  testId,
}: {
  make: () => Plot.PlotOptions;
  deps: unknown[];
  label: string;
  testId?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const makeRef = useRef(make);
  makeRef.current = make;
  const depKey = JSON.stringify(deps);
  useEffect(() => {
    if (!ref.current || !depKey) return;
    const el = Plot.plot({
      style: { background: "transparent", color: "var(--fg)", fontSize: "11px", overflow: "visible" },
      ...makeRef.current(),
    });
    // The figure is one image to assistive tech; Plot's inner aria-labels on <g> are not allowed there.
    for (const n of el.querySelectorAll("[aria-label]")) n.removeAttribute("aria-label");
    el.setAttribute("role", "img");
    el.setAttribute("aria-label", label);
    ref.current.replaceChildren(el);
    return () => el.remove();
  }, [depKey, label]);
  return <div ref={ref} className="plot" data-testid={testId} />;
}

export { Plot };
