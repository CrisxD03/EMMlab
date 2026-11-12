import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import type userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import type { TruthResponse, World } from "../src/api/types";
import { useDoc } from "../src/state/doc";
import { useUI } from "../src/state/ui";
import { connectionProblem } from "../src/state/worldOps";
import { bridge, inspector, openPreset, outline, renderApp, settleTruth } from "./utils";

type User = ReturnType<typeof userEvent.setup>;
const world = () => useDoc.getState().doc.world;

async function fill(user: User, el: Element, text: string) {
  await user.clear(el);
  await user.type(el, `${text}{Enter}`);
}

async function startBlank(user: User) {
  await user.click(await screen.findByRole("button", { name: "Start blank" }, { timeout: 20000 }));
  await waitFor(() => expect(screen.queryByTestId("gallery")).toBeNull());
}

describe("U1 canvas: nodes, arrows, level shifts, twists, decoys and the inspector", () => {
  it("builds and edits every World element from the canvas inspector", async () => {
    const { user } = renderApp();
    await startBlank(user);
    const o = outline();

    // add + rename latents (name field is focused and selected after "+ Latent")
    await user.click(o.getByRole("button", { name: "+ Latent" }));
    let name = await inspector().findByLabelText("Name");
    await waitFor(() => expect(name).toHaveFocus());
    await fill(user, name, "x");
    await user.click(o.getByRole("button", { name: "+ Latent" }));
    name = await inspector().findByLabelText("Name");
    await fill(user, name, "y");
    expect(Object.keys(world().nodes)).toEqual(["x", "y"]);
    expect(await screen.findByTestId("node-x")).toBeInTheDocument();
    expect(await screen.findByTestId("node-y")).toBeInTheDocument();

    // root distribution of x
    await user.click(o.getByRole("button", { name: "Latent x" }));
    await fill(user, inspector().getByLabelText("Mean"), "1");
    await fill(user, inspector().getByLabelText("SD"), "2");
    expect(world().nodes.x.root).toEqual({ mean: 1, sd: 2 });
    // level and decoy role
    await user.selectOptions(inspector().getByLabelText("Level"), "school");
    await user.selectOptions(inspector().getByLabelText("Role"), "decoy");
    expect(world().nodes.x.level).toBe("school");
    expect(world().nodes.x.role).toBe("decoy");
    expect(within(screen.getByTestId("node-x")).getByText("school")).toBeInTheDocument();
    expect(within(screen.getByTestId("node-x")).getByText("decoy")).toBeInTheDocument();
    await user.selectOptions(inspector().getByLabelText("Role"), "normal");

    // draw an arrow with the inspector control ("add arrow from the selected node")
    await user.selectOptions(inspector().getByLabelText("Add arrow from x to"), "y");
    await fill(user, inspector().getByLabelText("Slope of new arrow from x"), "0.4");
    expect(world().arrows).toEqual([{ from: "x", to: "y", slope: 0.4 }]);
    // y became a Child: it now has a recipe instead of a root distribution
    expect(world().nodes.y.recipe).toBeDefined();
    expect(world().nodes.y.root).toBeUndefined();
    expect(await screen.findByTestId("edge-x -> y")).toBeInTheDocument();
    // the canvas exposes drag handles on both sides of a node
    expect(screen.getByTestId("node-y").querySelector(".react-flow__handle.source")).not.toBeNull();
    expect(screen.getByTestId("node-y").querySelector(".react-flow__handle.target")).not.toBeNull();

    // set the slope
    await fill(user, inspector().getByLabelText("Slope x -> y"), "0.5");
    expect(world().arrows[0].slope).toBe(0.5);
    expect(within(screen.getByTestId("edge-x -> y")).getByText("0.5")).toBeInTheDocument();

    // child recipe and targets of y
    await user.click(o.getByRole("button", { name: "Latent y" }));
    await fill(user, inspector().getByLabelText("Intercept"), "0.1");
    await fill(user, inspector().getByLabelText("Noise SD"), "0.7");
    expect(world().nodes.y.recipe).toEqual({ intercept: 0.1, noise_sd: 0.7 });
    await fill(user, inspector().getByLabelText("Target SD"), "1.5");
    expect(world().nodes.y.target).toEqual({ sd: 1.5 });
    expect(world().nodes.y.recipe?.noise_sd).toBeUndefined();
    await fill(user, inspector().getByLabelText("Target ICC"), "0.2");
    await fill(user, inspector().getByLabelText("Target R²"), "0.3");
    expect(world().nodes.y.target).toEqual({ r2: 0.3, icc: 0.2 });
    await settleTruth();
    await waitFor(() => expect(Number(screen.getByTestId("derived-sd-y").dataset.value)).toBeGreaterThan(0));

    // a continuous attribute
    await user.click(o.getByRole("button", { name: "+ Continuous attribute" }));
    await fill(user, await inspector().findByLabelText("Name"), "size");
    expect(world().nodes.size.kind).toBe("attribute");

    // a categorical attribute: level, values and proportions
    await user.click(o.getByRole("button", { name: "+ Categorical attribute" }));
    await fill(user, await inspector().findByLabelText("Name"), "g");
    await user.selectOptions(inspector().getByLabelText("Level"), "school");
    await fill(user, inspector().getByLabelText("Value 1 name"), "lo");
    await fill(user, inspector().getByLabelText("New value name"), "mid");
    await fill(user, inspector().getByLabelText("Proportion of lo"), "0.3");
    await fill(user, inspector().getByLabelText("Proportion of high"), "0.4");
    await fill(user, inspector().getByLabelText("Proportion of mid"), "0.3");
    expect(world().attributes.g).toMatchObject({ level: "school", values: { lo: 0.3, high: 0.4, mid: 0.3 } });
    await user.selectOptions(inspector().getByLabelText("Role"), "decoy");
    expect(world().attributes.g.role).toBe("decoy");
    await user.selectOptions(inspector().getByLabelText("Role"), "normal");

    // level shift from the attribute
    await user.selectOptions(inspector().getByLabelText("Value"), "high");
    await user.selectOptions(inspector().getByLabelText("Shift node"), "y");
    await fill(user, inspector().getByLabelText("Delta of new level shift"), "0.25");
    expect(world().shifts).toEqual([{ when: { g: "high" }, target: "y", delta: 0.25 }]);
    expect(await screen.findByTestId("shift-0")).toHaveTextContent("g=high");

    // twist on the arrow (badge on the arrow)
    await user.click(o.getByRole("button", { name: "Arrow x -> y" }));
    await user.selectOptions(inspector().getByLabelText("Twist value"), "high");
    await fill(user, inspector().getByLabelText("Delta of new twist"), "-0.2");
    expect(world().twists).toEqual([{ arrow: "x -> y", when: { g: "high" }, delta: -0.2 }]);
    expect(await screen.findByTestId("twist-badge-x -> y")).toHaveTextContent("g=high");

    // rename x: arrows and twists follow
    await user.click(o.getByRole("button", { name: "Latent x" }));
    await fill(user, inspector().getByLabelText("Name"), "z");
    expect(world().arrows[0]).toMatchObject({ from: "z", to: "y" });
    expect(world().twists[0].arrow).toBe("z -> y");
    expect(await screen.findByTestId("edge-z -> y")).toBeInTheDocument();

    // remove: the attribute takes its shift and twist with it; the node goes away
    await user.click(o.getByRole("button", { name: "Attribute g" }));
    await user.click(inspector().getByRole("button", { name: "Delete g" }));
    expect(world().attributes.g).toBeUndefined();
    expect(world().shifts).toEqual([]);
    expect(world().twists).toEqual([]);
    await user.click(o.getByRole("button", { name: "Continuous attribute size" }));
    await user.click(inspector().getByRole("button", { name: "Delete size" }));
    expect(world().nodes.size).toBeUndefined();
    await waitFor(() => expect(screen.queryByTestId("node-size")).toBeNull());
    // x is now a school-level parent with SD 2: y's ICC target became infeasible; clearing it fixes the World
    await user.click(o.getByRole("button", { name: "Latent y" }));
    await fill(user, inspector().getByLabelText("Target ICC"), "");
    expect(world().nodes.y.target).toEqual({ r2: 0.3 });
    await settleTruth();
    expect(screen.queryByTestId("issues-panel")).toBeNull();
  });
});

describe("U2 live ripple: slope edits update every derived value on the canvas", () => {
  it("updates downstream means/R²/total/standardized values after a slope edit (pisa_like; SDs are fixed by targets there)", async () => {
    const { user } = renderApp();
    await openPreset(user, "pisa_like");
    await user.click(outline().getByRole("button", { name: "Arrow feedback -> enjoyment" }));
    const read = () => {
      const node = (n: string, what: string) =>
        within(screen.getByTestId(`node-${n}`))
          .getByLabelText(new RegExp(`^${what} of ${n}`))
          .getAttribute("data-value");
      return {
        enjoymentMean: node("enjoyment", "Mean"),
        enjoymentR2: node("enjoyment", "R²"),
        interestMean: node("interest", "Mean"),
        achievementMean: node("achievement", "Mean"),
        total: screen.getByTestId("total-feedback -> enjoyment").getAttribute("data-value"),
        std: screen.getByTestId("std-feedback -> enjoyment").getAttribute("data-value"),
        edgeStd: within(screen.getByTestId("edge-feedback -> enjoyment"))
          .getByRole("status", { hidden: true })
          .getAttribute("data-value"),
      };
    };
    const times: number[] = [];
    let before = read();
    for (const v of ["0.35", "0.42", "0.5", "0.3", "0.6", "0.45", "0.38", "0.55"]) {
      const input = inspector().getByLabelText("Slope");
      await user.clear(input);
      await user.type(input, v);
      const t0 = performance.now();
      fireEvent.keyDown(input, { key: "Enter" });
      await waitFor(() => expect(read().total).not.toBe(before.total), { timeout: 10000, interval: 5 });
      times.push(performance.now() - t0);
      const after = read();
      for (const k of Object.keys(after) as (keyof typeof after)[]) expect(after[k], k).not.toBe(before[k]);
      before = after;
    }
    times.sort((a, b) => a - b);
    const p95 = times[Math.ceil(0.95 * times.length) - 1];
    console.info(
      `U2 edit-to-render in jsdom via bridge: p95 ${p95.toFixed(0)} ms (${times.map((t) => t.toFixed(0)).join(", ")})`,
    );
    // Functional check here (every derived value re-rendered after each edit). The p95 < 150 ms target is
    // asserted in a real browser against `lab serve` (e2e/ux.spec.ts); jsdom + the stdin bridge only get a
    // loose bound that catches hangs or lost updates.
    expect(p95).toBeLessThan(5000);
  });
});

describe("U3 ripple distinguishes slope changes from spread/noise changes", () => {
  it("a slope edit marks raw relationships (data-ripple=slope); a noise edit marks only standardized coefficients", async () => {
    const { user } = renderApp();
    await openPreset(user, "minimal");
    await user.click(outline().getByRole("button", { name: "Arrow feedback -> selfeff" }));
    await fill(user, inspector().getByLabelText("Slope"), "0.45");
    await waitFor(() =>
      expect(screen.getByTestId("edge-feedback -> selfeff")).toHaveAttribute("data-ripple", "slope"),
    );
    expect(document.querySelector('.react-flow__edge [data-ripple="slope"]')).not.toBeNull();
    expect(screen.getByTestId("total-feedback -> selfeff")).toHaveAttribute("data-ripple", "slope");
    expect(document.querySelectorAll('[data-ripple="spread"]').length).toBe(0);

    await user.click(outline().getByRole("button", { name: "Latent selfeff" }));
    await fill(user, inspector().getByLabelText("Noise SD"), "1.3");
    await waitFor(() =>
      expect(document.querySelectorAll('[data-ripple="spread"]').length).toBeGreaterThan(0),
    );
    // nothing raw is highlighted; every highlighted element is a standardized coefficient
    expect(document.querySelectorAll('[data-ripple="slope"]').length).toBe(0);
    expect(document.querySelectorAll(".react-flow__edge [data-ripple]").length).toBe(0);
    for (const el of document.querySelectorAll('[data-ripple="spread"]')) {
      expect(el.getAttribute("aria-label")).toMatch(/^Standardized coefficient/);
    }
    // the std coefficient of selfeff -> achievement moved, its raw slope did not
    const edge = screen.getByTestId("edge-selfeff -> achievement");
    expect(edge).not.toHaveAttribute("data-ripple");
    expect(within(edge).getByLabelText(/^Standardized coefficient/)).toHaveAttribute("data-ripple", "spread");
  });
});

describe("U4 impossible states are prevented at input time", () => {
  it("refuses cycle-creating connections (the same check backs the canvas isValidConnection)", async () => {
    const { user } = renderApp();
    await openPreset(user, "minimal");
    const w = world();
    expect(connectionProblem(w, "achievement", "feedback")).toMatch(/cycle/);
    expect(connectionProblem(w, "selfeff", "feedback")).toMatch(/cycle/);
    expect(connectionProblem(w, "feedback", "feedback")).toMatch(/itself/);
    expect(connectionProblem(w, "ses", "selfeff")).toBeNull(); // attribute -> latent = level shift
    await user.click(outline().getByRole("button", { name: "Latent achievement" }));
    const sel = inspector().getByLabelText("Add arrow from achievement to");
    expect(
      within(sel).getByRole("option", { name: /feedback \(this arrow would create a cycle\)/ }),
    ).toBeInTheDocument();
    await user.selectOptions(sel, "feedback");
    expect(inspector().getByRole("alert")).toHaveTextContent("would create a cycle");
    await user.click(within(screen.getByTestId("add-arrow")).getByRole("button", { name: "Add" }));
    expect(world().arrows).toHaveLength(3);
  });

  it("flags an infeasible target SD inline with t_min and offers a fix", async () => {
    const { user } = renderApp();
    await openPreset(user, "minimal");
    await user.click(outline().getByRole("button", { name: "Latent achievement" }));
    await fill(user, inspector().getByLabelText("Target SD"), "0.2");
    const inline = await screen.findByTestId("inline-issues", {}, { timeout: 10000 });
    expect(inline).toHaveTextContent("t_min = 0.376");
    expect(inspector().getByLabelText("Target SD")).toHaveAttribute("aria-invalid", "true");
    await user.click(within(inline).getByRole("button", { name: /Raise target sd to/ }));
    await waitFor(() => expect(screen.queryByTestId("inline-issues")).toBeNull(), { timeout: 10000 });
    expect(world().nodes.achievement.target?.sd).toBeGreaterThan(0.376);
  });

  it("twist UI only offers existing arrows and categorical attributes", async () => {
    const { user } = renderApp();
    await openPreset(user, "pisa_like");
    // twists are created from an arrow's inspector, so they can only sit on existing arrows
    expect(screen.queryByTestId("add-twist")).toBeNull();
    await user.click(outline().getByRole("button", { name: "Arrow feedback -> achievement" }));
    const attr = inspector().getByLabelText("Twist attribute");
    const options = within(attr)
      .getAllByRole("option")
      .map((o) => o.getAttribute("value"));
    expect(options.sort()).toEqual(Object.keys(world().attributes).sort());
    expect(options).not.toContain("school_size"); // continuous attribute
    await user.selectOptions(attr, "immigrant");
    const vals = within(inspector().getByLabelText("Twist value"))
      .getAllByRole("option")
      .map((o) => o.getAttribute("value"));
    expect(vals).toEqual(["native", "immigrant"]);
  });
});

describe("U6 undo/redo for every canvas edit (≥50 steps, keyboard shortcuts)", () => {
  it("undoes 50 slope edits with Ctrl+Z and redoes with Ctrl+Shift+Z, and undoes structural edits", async () => {
    const { user } = renderApp();
    await openPreset(user, "minimal");
    await user.click(outline().getByRole("button", { name: "Arrow feedback -> selfeff" }));
    const slope = () => world().arrows.find((a) => a.from === "feedback" && a.to === "selfeff")!.slope;
    for (let i = 1; i <= 55; i++) {
      const input = inspector().getByLabelText("Slope");
      fireEvent.change(input, { target: { value: (i / 100).toFixed(2) } });
      fireEvent.keyDown(input, { key: "Enter" });
    }
    expect(slope()).toBe(0.55);
    act(() => (document.activeElement as HTMLElement | null)?.blur());
    for (let i = 0; i < 50; i++) await user.keyboard("{Control>}z{/Control}");
    expect(slope()).toBe(0.05);
    for (let i = 0; i < 3; i++) await user.keyboard("{Control>}{Shift>}z{/Shift}{/Control}");
    expect(slope()).toBe(0.08);
    await user.keyboard("{Control>}y{/Control}");
    expect(slope()).toBe(0.09);

    // structural edits: add node, delete arrow, twist, rename — each one undo step
    const before = JSON.stringify(world());
    await user.click(outline().getByRole("button", { name: "+ Latent" }));
    await fill(user, await inspector().findByLabelText("Name"), "extra");
    await user.click(outline().getByRole("button", { name: "Arrow feedback -> achievement" }));
    await user.click(inspector().getByRole("button", { name: "Remove twist 1" }));
    await user.click(inspector().getByRole("button", { name: "Delete arrow" }));
    expect(world().arrows).toHaveLength(2);
    act(() => (document.activeElement as HTMLElement | null)?.blur());
    await user.click(screen.getByRole("button", { name: "Undo" }));
    expect(world().arrows).toHaveLength(3);
    for (let i = 0; i < 3; i++) await user.keyboard("{Control>}z{/Control}");
    expect(JSON.stringify(world())).toBe(before);
    await user.click(screen.getByRole("button", { name: "Redo" }));
    expect(world().nodes.latent1).toBeDefined();
  });
});

describe("U13 equations rendered per node with KaTeX, numbers matching the spec", () => {
  for (const preset of ["minimal", "pisa_like"]) {
    it(`renders ${preset}'s node equations identical to the server's`, async () => {
      const { user } = renderApp();
      await openPreset(user, preset);
      useUI.setState({ equationView: "compact" }); // the server's equation strings are the compact form
      const truth = await bridge.json<TruthResponse>("POST", "/api/world/truth", { world: world() as World });
      const eqs = await screen.findByTestId("world-equations");
      const shown = [...eqs.querySelectorAll("[data-equation]")];
      expect(shown.length).toBe(Object.keys(world().nodes).length);
      const server = new Map((truth.equations ?? []).map((e) => [e.split(" ")[0], e]));
      for (const el of shown) {
        const plain = el.getAttribute("data-equation")!;
        expect(plain).toBe(server.get(plain.split(" ")[0]));
        expect(el.querySelector(".katex")).not.toBeNull();
      }
      if (preset === "minimal") {
        const ach = screen.getByTestId("equation-achievement");
        const tex = ach.querySelector("annotation")?.textContent ?? "";
        for (const n of ["0.25", "0.2", "0.5", "0.4", "0.8"]) expect(tex).toContain(n);
      }
    });
  }
});
