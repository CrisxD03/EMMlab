import { screen, waitFor, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { World } from "../src/api/types";
import { unfoldedEquation } from "../src/components/Equation";
import { loadWorld } from "../src/features/commands";
import { useDoc } from "../src/state/doc";
import { useUI } from "../src/state/ui";
import { openPreset, renderApp } from "./utils";

const PATH = "feedback -> achievement";

/** Two independent student attributes whose twists add on one arrow: the conjunction beats each single condition. */
const TWO_ATTR_WORLD = {
  name: "ca",
  schools: { n_schools: 160, students_per_school: { dist: "fixed", n: 40 } },
  attributes: {
    a: { level: "student", values: { x: 0.5, y: 0.5 } },
    b: { level: "student", values: { u: 0.5, v: 0.5 } },
  },
  nodes: { feedback: { root: { mean: 0, sd: 1 } }, achievement: { recipe: { noise_sd: 1.0 } } },
  arrows: [{ from: "feedback", to: "achievement", slope: 0.1 }],
  shifts: [],
  twists: [
    { arrow: PATH, when: { a: "y" }, delta: 0.7 },
    { arrow: PATH, when: { b: "u" }, delta: 0.7 },
  ],
} as unknown as World;

describe("U14 contrastive analysis of a found conjunction", () => {
  it("decomposes the clicked finding into flipped slices with intervals and a verdict, and follows the selection", async () => {
    const { user } = renderApp();
    await loadWorld(TWO_ATTR_WORLD, "two attributes");
    useDoc.getState().apply("quality", "other", (d) => {
      d.method.emm.quality = "ent_dbeta"; // makes the conjunction beat both single conditions
    });
    await user.click(await screen.findByRole("button", { name: "Run" }));
    await screen.findByTestId("scorecard", {}, { timeout: 30000 });
    const run = useUI.getState().run!;
    const conj = run.emm.findings.find((f) => f.conditions.length === 2);
    expect(conj).toBeDefined();
    await user.click(screen.getByRole("button", { name: conj!.description }));

    const card = await screen.findByTestId("contrast", {}, { timeout: 20000 });
    await within(card).findByTestId("contrast-table", {}, { timeout: 20000 });
    const heads = within(card)
      .getAllByRole("columnheader")
      .map((h) => h.textContent ?? "");
    for (const part of ["NOT", "EMM baseline", "finding"]) expect(heads.join("|")).toContain(part);
    // the finding, both single flips, the double negation and the complement
    expect(card.querySelectorAll("[data-slice]")).toHaveLength(5);
    const verdict = within(card).getByTestId("contrast-verdicts").querySelector("[data-verdict]")!;
    expect(verdict).toHaveAttribute("data-verdict", "conjunction_specific");
    // each flip deviates from the finding on the focal path
    const row = card.querySelector(`tr[data-path="${PATH}"]`)!;
    expect(row.querySelectorAll('td[data-deviating="true"]').length).toBeGreaterThanOrEqual(2);
    // each condition on its own
    const singles = within(card).getByTestId("contrast-singles");
    for (const cond of conj!.description.split(" AND ")) expect(singles.textContent).toContain(cond);
    // the contrast picture: a 2 x 2 grid with all six comparisons and the interaction line
    const pic = await screen.findByTestId(`contrast-plot-${PATH}`, {}, { timeout: 20000 });
    expect(pic.querySelectorAll("[data-cell]")).toHaveLength(4);
    expect(pic.querySelectorAll("[data-pair]")).toHaveLength(6);
    expect(within(pic).getByTestId("interaction")).toHaveTextContent("Interaction");
    // the card follows the selection
    const other = run.emm.findings.find((f) => f.rank !== conj!.rank)!;
    await user.click(screen.getByRole("button", { name: other.description }));
    await waitFor(() => expect(screen.getByTestId("contrast")).toHaveTextContent(other.description), {
      timeout: 20000,
    });
  });
});

describe("U15 generating equations: paths, compact and primitives, in the World and Results tabs", () => {
  it("switches the World tab's equations between the three views", async () => {
    const { user } = renderApp();
    await openPreset(user, "minimal");
    useUI.getState().select({ type: "node", id: "achievement" });
    const view = (name: string) => screen.getByRole("radio", { name });
    const eqText = () => screen.getByTestId("equation-achievement").getAttribute("data-equation") ?? "";
    expect(view("Paths")).toBeChecked();
    await waitFor(() => expect(eqText()).toContain("level shifts by ses"), { timeout: 15000 });
    expect(eqText()).toContain("(0.2 - 0.4·[ses=rich])·feedback"); // the twist stays: it is a slope
    expect(eqText()).not.toContain("0.5·[ses=rich]"); // the level shift is only named
    await user.click(view("Primitives"));
    await waitFor(() => expect(eqText()).toContain("(0.275 - 0.4·[ses=rich])·e_feedback"), {
      timeout: 15000,
    });
    expect(eqText()).toContain("0.425·[ses=rich]");
    expect(eqText()).not.toMatch(/·feedback\b(?!_)/); // no parent variable is left
    await user.click(view("Compact"));
    await waitFor(() => expect(eqText()).toContain("0.5·[ses=rich]"));
    expect(useUI.getState().equationView).toBe("compact");
  });

  it("shows generating next to fitted equations in Results, and unfolds the fits too", async () => {
    const { user } = renderApp();
    await openPreset(user, "minimal");
    await user.click(screen.getByRole("button", { name: "Run" }));
    await screen.findByTestId("scorecard", {}, { timeout: 30000 });
    const card = await screen.findByTestId("equations-card");
    const gen = () =>
      within(card).getByTestId("generating-eq-achievement").getAttribute("data-equation") ?? "";
    const fitted = () =>
      within(card).getByTestId("fitted-eq-achievement").getAttribute("data-equation") ?? "";
    expect(gen()).toContain("(0.2 - 0.4·[ses=rich])·feedback");
    expect(fitted()).toMatch(/·feedback/);
    await user.click(within(card).getByRole("radio", { name: "Primitives" }));
    await waitFor(() => expect(gen()).toContain("e_feedback"));
    expect(within(card).getByTestId("primitives")).toHaveTextContent("e_feedback");
    // the fitted model unfolded into its own starting latent and residuals
    expect(fitted()).toMatch(/·feedback/);
    expect(fitted()).toContain("e_selfeff");
    expect(fitted()).not.toContain("·selfeff ");
  });
});

describe("U15 unfolded equation text", () => {
  it("lists a node's noise and school intercept once, as one factor, and writes each factor once", () => {
    const eq = unfoldedEquation({
      node: "y",
      form: "roots",
      const: [{ coef: -0.3158, when: [["gender", "f"]] }],
      vars: [],
      noise: [
        {
          symbol: "e:x",
          kind: "e",
          node: "x",
          sd: 0.9,
          poly: [
            { coef: 0.097, when: [] },
            { coef: 0.1, when: [["gender", "f"]] },
          ],
        },
        {
          symbol: "u:x",
          kind: "u",
          node: "x",
          sd: 0.3,
          poly: [
            { coef: 0.097, when: [] },
            { coef: 0.1, when: [["gender", "f"]] },
          ],
        },
        { symbol: "e:y", kind: "e", node: "y", sd: 1, poly: [{ coef: 1, when: [] }] },
      ],
    });
    expect(eq.plain).toBe("y = -0.316·[gender=f] + (0.097 + 0.1·[gender=f])·(e_x + u_x) + e_y");
  });
});

describe("U17 a twist on two conditions can be planted from the twist form", () => {
  it("adds and edits a second condition on a twist", async () => {
    const { user } = renderApp();
    await loadWorld({ ...TWO_ATTR_WORLD, twists: [] } as World, "two attributes, no twist");
    useUI.getState().select({ type: "arrow", id: PATH });
    const form = await screen.findByTestId("add-twist", {}, { timeout: 15000 });
    await user.selectOptions(within(form).getByLabelText("Twist attribute"), "a");
    await user.selectOptions(within(form).getByLabelText("Twist value"), "y");
    await user.selectOptions(within(form).getByLabelText("and (optional)"), "b");
    await user.selectOptions(within(form).getByLabelText("and value"), "u");
    await user.type(within(form).getByLabelText("Delta of new twist"), "0.5{Enter}");
    await waitFor(() => expect(useDoc.getState().doc.world.twists).toHaveLength(1));
    expect(useDoc.getState().doc.world.twists[0].when).toEqual({ a: "y", b: "u" });
    // editing the first condition keeps the second
    const row = await screen.findByTestId("twist-row-0");
    await user.selectOptions(within(row).getByLabelText("Value"), "x");
    await waitFor(() => expect(useDoc.getState().doc.world.twists[0].when).toEqual({ a: "x", b: "u" }));
    // and it can be removed again
    await user.selectOptions(within(row).getByLabelText("and (optional)"), "");
    await waitFor(() => expect(useDoc.getState().doc.world.twists[0].when).toEqual({ a: "x" }));
  });
});
