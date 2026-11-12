import { screen, waitFor, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { formatG } from "../src/components/Equation";
import { useDoc } from "../src/state/doc";
import { useUI } from "../src/state/ui";
import { actionCounter, openPreset, renderApp } from "./utils";

describe("U12 preset gallery on first launch; minimal + Run gives a scorecard with no other input", () => {
  it("shows the gallery with all presets and null variants for twisted Worlds", async () => {
    const { user } = renderApp();
    const gallery = await screen.findByTestId("gallery");
    for (const p of ["minimal", "pisa_like", "simpson", "level_shift_only", "variance_decoy"]) {
      expect(await within(gallery).findByTestId(`preset-${p}`)).toBeInTheDocument();
    }
    expect(within(gallery).getByRole("button", { name: "Open minimal null variant" })).toBeInTheDocument();
    expect(within(gallery).getByRole("button", { name: "Open pisa_like null variant" })).toBeInTheDocument();
    expect(
      within(within(gallery).getByTestId("preset-level_shift_only")).queryByRole("button", {
        name: /null variant/,
      }),
    ).toBeNull();
    await user.click(within(gallery).getByRole("button", { name: "Open minimal null variant" }));
    await waitFor(() => expect(useDoc.getState().doc.world.name).toBe("minimal__null"));
    expect(useDoc.getState().doc.world.twists).toEqual([]);
  });

  it("open minimal, press Run: scorecard", async () => {
    const { user } = renderApp();
    const a = actionCounter(user);
    const gallery = await screen.findByTestId("gallery");
    await a.click(await within(gallery).findByRole("button", { name: "Open minimal" }), "Open minimal");
    await waitFor(() => expect(screen.queryByTestId("gallery")).toBeNull());
    await a.click(screen.getByRole("button", { name: "Run" }), "Run");
    const sc = await screen.findByTestId("scorecard", {}, { timeout: 30000 });
    expect(a.count).toBe(2);
    expect(within(sc).getByTestId("one-draw")).toHaveTextContent("one draw — not evidence");
    expect(within(sc).getByTestId("hits")).toHaveTextContent("rich schools flip");
  });
});

describe("U9 results: subgroup diagram, coefficient hover with truths + provenance, contrast picture, planted slopes, scorecard", () => {
  it("shows findings, highlights changed arrows, explains coefficients and plots groups", async () => {
    const { user } = renderApp();
    await openPreset(user, "minimal");
    await user.click(screen.getByRole("button", { name: "Run" }));
    await screen.findByTestId("scorecard", {}, { timeout: 30000 });
    const r = useUI.getState().run!;
    const f = r.emm.findings[0];
    expect(f.description).toBe("ses=rich");

    // scorecard: hits / misses / false positives by class / cluster tier badges
    const sc = screen.getByTestId("scorecard");
    expect(within(sc).getByTestId("misses")).toHaveTextContent("Misses · 0");
    for (const c of ["null", "decoy", "phantom_downstream", "variance_artifact", "level_shift_leak", "proxy"])
      expect(sc.querySelector(`[data-fp-class="${c}"]`)).not.toBeNull();
    expect(within(screen.getByTestId("finding-1")).getByTestId("tier")).toHaveTextContent("ok");

    // the quality function is confirmed in the table, with the count of deviating paths and their intervals
    expect(screen.getByTestId("quality-rule")).toHaveTextContent("ent_z");
    expect(screen.getByTestId("deviating-1")).toHaveTextContent("1 / 3");
    const paths = within(screen.getByTestId("subgroup-paths"));
    expect(paths.getByText("feedback -> achievement").closest("tr")).toHaveAttribute(
      "data-deviating",
      "true",
    );
    expect(paths.getByText("feedback -> selfeff").closest("tr")).toHaveAttribute("data-deviating", "false");

    // diagram for finding 1: the deviating (twisted) arrow is highlighted
    const diag = screen.getByTestId("subgroup-diagram");
    expect(within(diag).getByTestId("diagram-edge-feedback -> achievement")).toHaveAttribute(
      "data-changed",
      "true",
    );
    expect(within(diag).getByTestId("diagram-edge-feedback -> selfeff")).toHaveAttribute(
      "data-changed",
      "false",
    );
    expect(within(diag).getByTestId("diagram-edge-selfeff -> achievement")).toHaveAttribute(
      "data-changed",
      "false",
    );

    // hover a coefficient: estimand label, T1/T2/T3, true Δ and provenance
    const coef = within(within(diag).getByTestId("diagram-edge-feedback -> achievement")).getAllByRole(
      "button",
    )[0];
    await user.hover(coef);
    const tip = await screen.findByRole("tooltip", {}, { timeout: 5000 });
    for (const s of [
      "pooled slope",
      "T1 structural",
      "T2 causal target",
      "T3 step estimand",
      "true Δ",
      "provenance",
      r.run_key,
      r.manifest.world,
    ])
      expect(tip).toHaveTextContent(s);
    await user.unhover(coef);

    // subgroup equations (U13 per subgroup): numbers equal the finding's b_g
    const eq = screen.getByTestId("subgroup-eq-achievement");
    const bg = f.paths.find((p) => p.path === "feedback -> achievement")!.b_g;
    expect(eq.getAttribute("data-equation")).toContain(`${formatG(bg, 3, true)}·feedback`);
    expect(eq.querySelector(".katex")).not.toBeNull();

    // contrast picture for the deviating path: finding vs complement, on one axis, with an SVG download
    const pic = await screen.findByTestId("contrast-plot-feedback -> achievement", {}, { timeout: 20000 });
    expect(pic.querySelector('[data-cell="finding"]')).not.toBeNull();
    expect(pic.querySelector('[data-cell="complement"]')).not.toBeNull();
    expect(pic.querySelector('[data-pair="finding|complement"]')).toHaveAttribute("data-deviating", "true");
    expect(within(pic).getByRole("button", { name: "Download SVG" })).toBeInTheDocument();

    // planted vs found per path, and how far the search got
    expect(screen.getByTestId("planted-feedback -> achievement")).toHaveTextContent("0.2 − 0.4·[ses=rich]");
    expect(screen.getByTestId("search-coverage")).toHaveTextContent("candidate subgroups");
    expect(screen.getAllByTestId("tier-legend")[0]).toHaveTextContent("fragile 20–49");

    // full sample: estimates vs T3
    await user.click(screen.getByRole("button", { name: "Full sample" }));
    expect(
      within(screen.getByTestId("subgroup-diagram")).getByTestId("diagram-edge-feedback -> achievement"),
    ).toHaveAttribute("data-changed", "false");
    expect(screen.getByTestId("estimation").querySelectorAll("tbody tr")).toHaveLength(3);
  });
});

describe("U11 report export: self-contained HTML", () => {
  it("opens and downloads the run report", async () => {
    const { user } = renderApp();
    await openPreset(user, "minimal");
    await user.click(screen.getByRole("button", { name: "Run" }));
    await screen.findByTestId("scorecard", {}, { timeout: 30000 });
    const key = useUI.getState().run!.run_key;
    const open = screen.getByTestId("open-run-report");
    expect(open).toHaveAttribute("href", `/api/run/${key}/report`);
    expect(open).toHaveAttribute("target", "_blank");
    const spy = vi.spyOn(URL, "createObjectURL");
    await user.click(screen.getByRole("button", { name: "Download report (.html)" }));
    await waitFor(() => expect(spy).toHaveBeenCalled());
    const blob = spy.mock.calls[0][0] as Blob;
    const html = await blob.text();
    expect(html.toLowerCase()).toMatch(/^<!doctype html>/);
    expect(html).toContain("one draw — not evidence");
    expect(html).not.toMatch(/(src|href)=["']?https?:/i);
    expect(html).not.toMatch(/<script[^>]+src=/i);
    expect(html).not.toMatch(/<link[^>]+stylesheet/i);
  });
});

describe("U18 planted vs found per path, how far the search got, cluster tiers explained", () => {
  it("shows the World's slope next to the estimate, the search coverage line and the tier legend", async () => {
    const { user } = renderApp();
    await openPreset(user, "minimal");
    await user.click(screen.getByRole("button", { name: "Run" }));
    await screen.findByTestId("scorecard", {}, { timeout: 30000 });
    expect(screen.getByTestId("planted-feedback -> achievement")).toHaveTextContent("0.2 − 0.4·[ses=rich]");
    expect(screen.getByTestId("planted-feedback -> selfeff")).toHaveTextContent("0.3");
    const cov = useUI.getState().run!.emm.coverage!;
    expect(screen.getByTestId("search-coverage")).toHaveTextContent(
      `Scored ${cov.scored} candidate subgroups`,
    );
    expect(screen.getAllByTestId("tier-legend")[0]).toHaveTextContent("fragile 20–49");
  });
});
