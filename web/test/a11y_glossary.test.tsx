import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { act, screen, waitFor, within } from "@testing-library/react";
import axe from "axe-core";
import { describe, expect, it } from "vitest";
import type { Glossary } from "../src/api/types";
import { useDoc } from "../src/state/doc";
import { useUI } from "../src/state/ui";
import { bridge, openPreset, renderApp } from "./utils";

async function seriousViolations(where: string) {
  const res = await axe.run(document.body, {
    // contrast needs real layout and colors: checked in a real browser (e2e/ux.spec.ts)
    rules: { "color-contrast": { enabled: false } },
    resultTypes: ["violations"],
  });
  const bad = res.violations.filter((v) => v.impact === "serious" || v.impact === "critical");
  if (bad.length)
    console.error(
      `axe (${where}):\n${bad
        .map(
          (v) =>
            `  ${v.id} (${v.impact}): ${v.help}\n    ${v.nodes
              .map((n) => n.target.join(" "))
              .slice(0, 4)
              .join("\n    ")}`,
        )
        .join("\n")}`,
    );
  return bad.map((v) => `${where}: ${v.id}`);
}

describe("UX3 accessibility: no serious/critical axe violations; keyboard-operable inspector; visible focus", () => {
  it("has no serious or critical axe violations on every view", async () => {
    const { user } = renderApp();
    const all: string[] = [];
    await screen.findByTestId("preset-minimal", {}, { timeout: 20000 });
    all.push(...(await seriousViolations("gallery")));
    await openPreset(user, "minimal");
    all.push(...(await seriousViolations("world")));
    await user.click(screen.getByRole("button", { name: "Arrow feedback -> achievement" }));
    all.push(...(await seriousViolations("world: arrow inspector")));
    await user.click(screen.getByRole("button", { name: "Attribute ses" }));
    all.push(...(await seriousViolations("world: attribute inspector")));
    await user.click(screen.getByRole("button", { name: "YAML" }));
    await screen.findByTestId("yaml-editor");
    all.push(...(await seriousViolations("world: YAML split view")));
    await user.click(screen.getByRole("tab", { name: "Researcher's Model" }));
    await user.click(await screen.findByRole("button", { name: "Copy from World" }));
    await screen.findByTestId("eq-achievement", {}, { timeout: 10000 });
    all.push(...(await seriousViolations("model")));
    await user.click(screen.getByRole("tab", { name: "Search & method" }));
    await screen.findByTestId("descriptor-ses", {}, { timeout: 20000 });
    all.push(...(await seriousViolations("search")));
    await user.click(screen.getByRole("button", { name: "Run" }));
    await screen.findByTestId("scorecard", {}, { timeout: 30000 });
    await waitFor(() => expect(screen.getByTestId("contrast-plot").querySelector("svg")).not.toBeNull(), {
      timeout: 15000,
    });
    all.push(...(await seriousViolations("results")));
    await user.click(screen.getByRole("tab", { name: "Experiments" }));
    await screen.findByTestId("launcher");
    all.push(...(await seriousViolations("experiments")));
    await user.click(screen.getByRole("button", { name: "Glossary" }));
    await screen.findByTestId("glossary");
    all.push(...(await seriousViolations("glossary")));
    expect(all).toEqual([]);
  });

  it("operates the inspector with the keyboard only", async () => {
    const { user } = renderApp();
    await openPreset(user, "minimal");
    act(() => (document.activeElement as HTMLElement | null)?.blur());
    // Tab to the outline entry of an arrow, activate it with Enter
    let found = false;
    for (let i = 0; i < 80 && !found; i++) {
      await user.tab();
      found = document.activeElement?.getAttribute("aria-label") === "Arrow feedback -> selfeff";
    }
    expect(found).toBe(true);
    await user.keyboard("{Enter}");
    expect(useUI.getState().selection).toEqual({ type: "arrow", id: "feedback -> selfeff" });
    // Tab into the inspector's slope field, type a value, Enter commits
    found = false;
    for (let i = 0; i < 200 && !found; i++) {
      await user.tab();
      found =
        document.activeElement?.closest("[data-testid=inspector]") !== null &&
        (document.activeElement as HTMLInputElement).id !== "" &&
        document.querySelector(`label[for="${(document.activeElement as HTMLElement).id}"]`)?.textContent ===
          "Slope";
    }
    expect(found).toBe(true);
    await user.keyboard("{Control>}a{/Control}0.41{Enter}");
    expect(useDoc.getState().doc.world.arrows.find((a) => a.to === "selfeff")?.slope).toBe(0.41);
    // selects are operable with the keyboard too
    const sel = within(screen.getByTestId("inspector")).getByLabelText("Specified as");
    act(() => sel.focus());
    await user.selectOptions(sel, "std");
    expect(useDoc.getState().doc.world.arrows.find((a) => a.to === "selfeff")?.std).toBeDefined();
    // Escape returns to World settings; Ctrl+Z undoes the last edit
    act(() => (document.activeElement as HTMLElement | null)?.blur());
    await user.keyboard("{Escape}");
    expect(useUI.getState().selection).toEqual({ type: "world" });
    await user.keyboard("{Control>}z{/Control}");
    expect(useDoc.getState().doc.world.arrows.find((a) => a.to === "selfeff")?.slope).toBe(0.41);
  });

  it("defines a visible focus style for every focusable control", () => {
    const css = readFileSync(resolve(process.cwd(), "src/styles.css"), "utf8");
    expect(css).toMatch(/:focus-visible\s*{\s*outline:\s*2px solid var\(--focus\)/);
    expect(css).toMatch(/\.cm-editor\.cm-focused\s*{\s*outline:/);
  });
});

describe("UX5 every glossary term is an in-app tooltip with the same definition", () => {
  it("shows each term's definition (+ amendment) as its tooltip", async () => {
    const { user } = renderApp({ firstLaunch: false });
    const glossary = await bridge.json<Glossary>("GET", "/api/glossary");
    expect(Object.keys(glossary).length).toBeGreaterThan(30);
    await user.click(screen.getByRole("button", { name: "Glossary" }));
    const panel = await screen.findByTestId("glossary");
    await waitFor(() =>
      expect(panel.querySelectorAll("[data-glossary]").length).toBe(Object.keys(glossary).length),
    );
    for (const [term, e] of Object.entries(glossary)) {
      const trigger = panel.querySelector(`[data-glossary="${CSS.escape(term)}"]`) as HTMLElement;
      expect(trigger, term).not.toBeNull();
      act(() => trigger.focus());
      const tip = await screen.findByRole("tooltip");
      const want = e.amendment ? `${e.definition} ${e.amendment}` : e.definition;
      expect(tip.textContent, term).toBe(want);
      act(() => trigger.blur());
      await waitFor(() => expect(screen.queryByRole("tooltip")).toBeNull());
    }
  });

  it("uses glossary tooltips inline across the app (e.g. Twist, Level Shift, Researcher's Model)", async () => {
    const { user } = renderApp();
    await openPreset(user, "minimal");
    for (const t of ["World", "Latent", "Attribute", "Arrow", "Level Shift", "Twist"])
      expect(document.querySelector(`[data-glossary="${t}"]`), t).not.toBeNull();
    await user.click(screen.getByRole("tab", { name: "Researcher's Model" }));
    expect(
      await screen.findByText("Researcher's Model", { selector: "[data-glossary]" }),
    ).toBeInTheDocument();
  });
});
