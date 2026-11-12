import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { useDoc } from "../src/state/doc";
import { openPreset, renderApp } from "./utils";

const doc = () => useDoc.getState().doc;

describe("U7 Researcher's Model canvas", () => {
  it("copies from the World (dropping attribute arrows and twists), edits arrows, and labels residualization", async () => {
    const { user } = renderApp();
    await openPreset(user, "pisa_like");
    await user.click(screen.getByRole("tab", { name: "Researcher's Model" }));
    // start from an empty model, then copy
    useDoc.getState().apply("clear model", "other", (d) => {
      d.model = { name: "m", latents: [], arrows: [] };
    });
    await user.click(await screen.findByRole("button", { name: "Copy from World" }));
    await waitFor(() => expect(doc().model.latents.length).toBe(5));
    // unobserved prior_ach and the school_size attribute are not in the model; no attribute arrows/twists
    expect(doc().model.latents).toEqual(["feedback", "selfeff", "enjoyment", "interest", "achievement"]);
    expect(doc().model.arrows).not.toContain("prior_ach -> feedback");
    expect(doc().model.arrows).toHaveLength(9);
    for (const k of doc().model.arrows) expect(k).not.toMatch(/gender|school_type|prior_ach/);
    expect(JSON.stringify(doc().model)).not.toContain("twist");
    await waitFor(() => expect(screen.getAllByTestId(/^model-node-/)).toHaveLength(5));
  });

  it("removes/adds arrows and clicking a node residualizes on it: triangle keeps R², chain drops it", async () => {
    const { user } = renderApp();
    await openPreset(user, "minimal");
    await user.click(screen.getByRole("tab", { name: "Researcher's Model" }));
    await user.click(await screen.findByRole("button", { name: "Copy from World" }));
    await waitFor(() => expect(doc().model.arrows).toHaveLength(3));

    // click the feedback node on the canvas
    const node = await screen.findByTestId("model-node-feedback");
    fireEvent.click(node.closest(".react-flow__node")!);
    await waitFor(() => expect(doc().method.analysis.residualize_on).toEqual(["feedback"]));
    expect(node).toHaveAttribute("data-residualized", "true");
    const tri = await screen.findByTestId("case-achievement", {}, { timeout: 10000 });
    expect(tri).toHaveTextContent("triangle");
    const r2 = within(screen.getByTestId("r2-achievement")).getAllByRole("status", { hidden: true });
    expect(Number(r2[0].dataset.value)).toBeCloseTo(Number(r2[1].dataset.value), 10);

    // remove feedback -> achievement: now feedback only reaches achievement through selfeff (chain)
    await user.click(screen.getByRole("button", { name: "Remove model arrow feedback -> achievement" }));
    await waitFor(() => expect(screen.getByTestId("case-achievement")).toHaveTextContent("chain"), {
      timeout: 10000,
    });
    await waitFor(() => {
      const v = within(screen.getByTestId("r2-achievement")).getAllByRole("status", { hidden: true });
      expect(Number(v[1].dataset.value)).toBeLessThan(Number(v[0].dataset.value));
    });

    // add it back with the inspector control
    await user.selectOptions(screen.getByLabelText("Model arrow from"), "feedback");
    await user.selectOptions(screen.getByLabelText("to"), "achievement");
    await user.click(
      within(screen.getByRole("complementary", { name: "Model inspector" })).getByRole("button", {
        name: "Add",
      }),
    );
    expect(doc().model.arrows).toContain("feedback -> achievement");
    await waitFor(() => expect(screen.getByTestId("case-achievement")).toHaveTextContent("triangle"), {
      timeout: 10000,
    });
    // equations per node rendered with KaTeX
    expect(screen.getByTestId("model-equation-achievement").querySelector(".katex")).not.toBeNull();
    // click again: residualization off
    fireEvent.click(screen.getByTestId("model-node-feedback").closest(".react-flow__node")!);
    await waitFor(() => expect(doc().method.analysis.residualize_on).toEqual([]));
  });
});

describe("U7 SEM coefficients on the Researcher's Model canvas", () => {
  it("labels every arrow with its expected coefficient, follows residualization, and can be toggled off", async () => {
    const { user } = renderApp();
    await openPreset(user, "minimal");
    await user.click(screen.getByRole("tab", { name: "Researcher's Model" }));
    await user.click(await screen.findByRole("button", { name: "Copy from World" }));
    await waitFor(() => expect(doc().model.arrows).toHaveLength(3));

    const value = (arrow: string) =>
      within(screen.getByTestId(`model-edge-${arrow}`))
        .getAllByRole("status", { hidden: true })
        .at(-1)!.dataset.value;
    // every arrow carries a coefficient; before any residualization nothing is marked as changed
    await waitFor(() => expect(screen.getAllByTestId(/^model-edge-/)).toHaveLength(3), { timeout: 10000 });
    for (const el of screen.getAllByTestId(/^model-edge-/))
      expect(el).toHaveAttribute("data-changed", "false");
    const before = Number(value("feedback -> achievement"));
    expect(Number(value("feedback -> selfeff"))).toBeCloseTo(0.329, 2); // pooled (no normalisation)

    // residualizing on feedback moves credit to feedback -> achievement (triangle) and marks it changed
    fireEvent.click(screen.getByTestId("model-node-feedback").closest(".react-flow__node")!);
    await waitFor(
      () =>
        expect(screen.getByTestId("model-edge-feedback -> achievement")).toHaveAttribute(
          "data-changed",
          "true",
        ),
      { timeout: 10000 },
    );
    expect(Number(value("feedback -> achievement"))).toBeGreaterThan(before + 0.05);
    expect(screen.getByTestId("model-edge-feedback -> selfeff")).toHaveAttribute("data-changed", "false");

    // the toggle hides the labels again
    await user.click(screen.getByRole("checkbox", { name: "Show SEM coefficients on the diagram" }));
    await waitFor(() => expect(screen.queryAllByTestId(/^model-edge-/)).toHaveLength(0));
  });
});

describe("U8 Search Space panel", () => {
  it("selects attributes, bins numeric ones with histogram and cut points, sets depth/min sizes/filter and method", async () => {
    const { user } = renderApp();
    await openPreset(user, "pisa_like");
    await user.click(screen.getByRole("tab", { name: "Search & method" }));
    const card = await screen.findByTestId("descriptor-school_size", {}, { timeout: 20000 });
    // histogram (Observable Plot SVG) with cut-point rules
    await waitFor(() =>
      expect(within(card).getByTestId("hist-school_size").querySelector("svg")).not.toBeNull(),
    );
    const cuts5 = within(card).getByTestId("cuts-school_size").textContent!;
    expect(cuts5.split(",").length).toBeGreaterThanOrEqual(3);
    // per-attribute binning
    await user.clear(within(card).getByLabelText("Bins"));
    await user.type(within(card).getByLabelText("Bins"), "3{Enter}");
    expect(doc().search.binning.school_size.n_bins).toBe(3);
    await waitFor(
      () =>
        expect(
          within(screen.getByTestId("descriptor-school_size")).getByTestId("cuts-school_size").textContent,
        ).not.toBe(cuts5),
      {
        timeout: 20000,
      },
    );
    await user.selectOptions(
      within(screen.getByTestId("descriptor-school_size")).getByLabelText("Weighting"),
      "schools",
    );
    await user.click(within(screen.getByTestId("descriptor-school_size")).getByLabelText("One-sided"));
    expect(doc().search.binning.school_size).toEqual({ n_bins: 3, one_sided: false, weighting: "schools" });
    // categorical descriptors show their value shares
    expect(
      within(screen.getByTestId("descriptor-gender")).getByRole("list", { name: "Value shares of gender" }),
    ).toBeInTheDocument();

    // attribute selection
    await user.click(within(screen.getByTestId("descriptor-football")).getByRole("checkbox"));
    expect(doc().search.attributes).toBeDefined();
    expect(doc().search.attributes).not.toContain("football");
    expect(doc().search.attributes).toContain("gender");

    // depth, minimum sizes, filter
    await user.selectOptions(screen.getByLabelText("Depth (conditions per subgroup)"), "1");
    await user.clear(screen.getByLabelText("Min students (n)"));
    await user.type(screen.getByLabelText("Min students (n)"), "150{Enter}");
    await user.clear(screen.getByLabelText("Min complement n"));
    await user.type(screen.getByLabelText("Min complement n"), "120{Enter}");
    await user.clear(screen.getByLabelText("Min schools"));
    await user.type(screen.getByLabelText("Min schools"), "10{Enter}");
    await user.selectOptions(screen.getByLabelText("Filter attribute"), "school_type");
    await user.selectOptions(screen.getByLabelText("Filter value"), "general");
    expect(doc().search).toMatchObject({
      depth: 1,
      min_n: 150,
      min_n_complement: 120,
      min_schools: 10,
      filter: { school_type: "general" },
    });

    // method preset selector and its settings
    await user.click(screen.getByRole("radio", { name: "current" }));
    await waitFor(() => expect(doc().method.name).toBe("current"));
    expect(doc().method.analysis).toMatchObject({
      standardize: "per_group",
      residualize_method: "simple",
      normalisation: "none",
    });
    expect(doc().method.emm.quality).toBe("current");
    expect(screen.getByLabelText("Standardize")).toHaveValue("per_group");
    await user.click(screen.getByRole("radio", { name: "recommended" }));
    await waitFor(() => expect(doc().method.emm.quality).toBe("ent_z"));
    await user.selectOptions(screen.getByLabelText("Normalisation"), "centered");
    await user.selectOptions(screen.getByLabelText("Comparison baseline"), "population");
    // the menu is built from emm.QUALITY_FUNCTIONS (GET /api/qualities): the entries with in_menu
    await waitFor(() => {
      const qOptions = [...(screen.getByLabelText("Quality function") as HTMLSelectElement).options].map(
        (o) => o.value,
      );
      expect(qOptions).toEqual(["current", "ent_z", "cook"]);
    });
    await user.selectOptions(screen.getByLabelText("Quality function"), "current");
    await user.selectOptions(screen.getByLabelText("Residualize method"), "simple");
    expect(doc().method.analysis.normalisation).toBe("centered");
    expect(doc().method.emm).toMatchObject({ baseline: "population", quality: "current" });
    expect(doc().method.analysis.residualize_method).toBe("simple");
    // steps (b)+(c) can be switched off (refinements stay in the list as found)
    expect(doc().method.emm.prune_refinements).toBe(true);
    await user.click(
      screen.getByRole("checkbox", { name: "Drop A AND B when it does not beat A or B alone" }),
    );
    expect(doc().method.emm.prune_refinements).toBe(false);
  });
});
