import { screen, waitFor, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { Preset } from "../src/api/types";
import { useDoc } from "../src/state/doc";
import { actionCounter, bridge, inspector, outline, renderApp, settleTruth } from "./utils";

describe("UX1 task script A: build World A from a blank World in ≤ 40 UI actions without YAML", () => {
  it("builds minimal's structure and reads the total effect feedback → achievement", async () => {
    const { user } = renderApp();
    await user.click(await screen.findByRole("button", { name: "Start blank" }, { timeout: 20000 }));
    await waitFor(() => expect(screen.queryByTestId("gallery")).toBeNull());
    const a = actionCounter(user);
    const o = outline();
    const ins = inspector;

    // 3 latents
    for (const n of ["feedback", "selfeff", "achievement"]) {
      await a.click(o.getByRole("button", { name: "+ Latent" }), "+ Latent");
      await a.fill(await ins().findByLabelText("Name"), n);
    }
    // 1 school attribute ses: poor / rich
    await a.click(o.getByRole("button", { name: "+ Categorical attribute" }), "+ Categorical");
    await a.fill(await ins().findByLabelText("Name"), "ses");
    await a.select(ins().getByLabelText("Level"), "school");
    await a.fill(ins().getByLabelText("Value 1 name"), "poor");
    await a.fill(ins().getByLabelText("Value 2 name"), "rich");

    // arrows F→S .3, F→Y .2 (from feedback), S→Y .25 (from selfeff)
    await a.click(o.getByRole("button", { name: "Latent feedback" }), "feedback");
    await a.select(ins().getByLabelText("Add arrow from feedback to"), "selfeff");
    await a.fill(ins().getByLabelText("Slope of new arrow from feedback"), "0.3");
    await a.select(ins().getByLabelText("Add arrow from feedback to"), "achievement");
    await a.fill(ins().getByLabelText("Slope of new arrow from feedback"), "0.2");
    await a.click(o.getByRole("button", { name: "Latent selfeff" }), "selfeff");
    await a.select(ins().getByLabelText("Add arrow from selfeff to"), "achievement");
    await a.fill(ins().getByLabelText("Slope of new arrow from selfeff"), "0.25");

    // level shifts rich → feedback .6, rich → achievement .5 (value defaults to the last value, rich)
    await a.click(o.getByRole("button", { name: "Attribute ses" }), "ses");
    await a.select(ins().getByLabelText("Shift node"), "feedback");
    await a.fill(ins().getByLabelText("Delta of new level shift"), "0.6");
    await a.select(ins().getByLabelText("Shift node"), "achievement");
    await a.fill(ins().getByLabelText("Delta of new level shift"), "0.5");

    // twist F→Y in rich schools: −.4 (attribute and value default to ses / rich)
    await a.click(o.getByRole("button", { name: "Arrow feedback -> achievement" }), "arrow F→Y");
    await a.fill(ins().getByLabelText("Delta of new twist"), "-0.4");

    // noise SDs as in World A (feedback keeps 1.0)
    await a.click(o.getByRole("button", { name: "Latent selfeff" }), "selfeff");
    await a.fill(ins().getByLabelText("Noise SD"), "0.9");
    await a.click(o.getByRole("button", { name: "Latent achievement" }), "achievement");
    await a.fill(ins().getByLabelText("Noise SD"), "0.8");

    // read the total effect
    await a.click(o.getByRole("button", { name: "Arrow feedback -> achievement" }), "arrow F→Y");
    await settleTruth();
    await waitFor(
      () => expect(screen.getByTestId("total-feedback -> achievement")).toHaveTextContent("0.075"),
      {
        timeout: 10000,
      },
    );
    const groups = screen.getByTestId("arrow-groups");
    expect(within(groups).getByTestId("total-feedback -> achievement-ses=poor")).toHaveTextContent("0.275");
    expect(within(groups).getByTestId("total-feedback -> achievement-ses=rich")).toHaveTextContent("−0.125");

    console.info(`UX1 actions: ${a.count}\n  ${a.log.join("\n  ")}`);
    expect(a.count).toBeLessThanOrEqual(40);
    expect(screen.queryByTestId("yaml-panel")).toBeNull(); // YAML never opened

    // the World built by hand has World A's structure
    const minimal = (await bridge.json<Preset[]>("GET", "/api/presets")).find(
      (p) => p.name === "minimal",
    )!.world;
    const w = useDoc.getState().doc.world;
    const norm = (x: unknown) => JSON.stringify(x);
    expect(norm(w.arrows.map((x) => [x.from, x.to, x.slope]).sort())).toBe(
      norm(minimal.arrows.map((x) => [x.from, x.to, x.slope]).sort()),
    );
    expect(norm(w.shifts)).toBe(norm(minimal.shifts));
    expect(w.twists.map((t) => [t.arrow, t.when, t.delta])).toEqual(
      minimal.twists.map((t) => [t.arrow, t.when, t.delta]),
    );
    expect(Object.keys(w.attributes.ses.values)).toEqual(["poor", "rich"]);
    expect(w.attributes.ses.level).toBe("school");
    expect(w.nodes.selfeff.recipe?.noise_sd).toBe(0.9);
    expect(w.nodes.achievement.recipe?.noise_sd).toBe(0.8);
    expect(w.nodes.feedback.recipe).toEqual({ intercept: 0, noise_sd: 1 });
  });
});
