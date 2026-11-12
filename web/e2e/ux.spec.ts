import { expect, test } from "@playwright/test";
import { actionCounter, firstLaunch, inspector, openPreset, outline, seriousAxe } from "./helpers";

test.describe("UX1 task script A", () => {
  test("build World A from blank in ≤ 40 actions without YAML, read total effect 0.275 / −0.125", async ({
    page,
  }) => {
    await firstLaunch(page);
    await page.getByRole("button", { name: "Start blank" }).click();
    const a = actionCounter();
    const o = outline(page);
    const ins = inspector(page);
    for (const n of ["feedback", "selfeff", "achievement"]) {
      await a.click(o.getByRole("button", { name: "+ Latent" }), "+ Latent");
      await a.fill(ins.getByLabel("Name", { exact: true }), n);
    }
    await a.click(o.getByRole("button", { name: "+ Categorical attribute" }), "+ Categorical");
    await a.fill(ins.getByLabel("Name", { exact: true }), "ses");
    await a.select(ins.getByLabel("Level", { exact: true }), "school");
    await a.fill(ins.getByLabel("Value 1 name"), "poor");
    await a.fill(ins.getByLabel("Value 2 name"), "rich");
    await a.click(o.getByRole("button", { name: "Latent feedback" }), "feedback");
    await a.select(ins.getByLabel("Add arrow from feedback to"), "selfeff");
    await a.fill(ins.getByLabel("Slope of new arrow from feedback"), "0.3");
    await a.select(ins.getByLabel("Add arrow from feedback to"), "achievement");
    await a.fill(ins.getByLabel("Slope of new arrow from feedback"), "0.2");
    await a.click(o.getByRole("button", { name: "Latent selfeff" }), "selfeff");
    await a.select(ins.getByLabel("Add arrow from selfeff to"), "achievement");
    await a.fill(ins.getByLabel("Slope of new arrow from selfeff"), "0.25");
    await a.click(o.getByRole("button", { name: "Attribute ses" }), "ses");
    await a.select(ins.getByLabel("Shift node"), "feedback");
    await a.fill(ins.getByLabel("Delta of new level shift"), "0.6");
    await a.select(ins.getByLabel("Shift node"), "achievement");
    await a.fill(ins.getByLabel("Delta of new level shift"), "0.5");
    await a.click(o.getByRole("button", { name: "Arrow feedback -> achievement" }), "arrow");
    await a.fill(ins.getByLabel("Delta of new twist"), "-0.4");
    await expect(page.getByTestId("total-feedback -> achievement")).toHaveText("0.075");
    await expect(page.getByTestId("total-feedback -> achievement-ses=poor")).toHaveText("0.275");
    await expect(page.getByTestId("total-feedback -> achievement-ses=rich")).toHaveText("−0.125");
    console.log(`UX1: ${a.count} actions`);
    expect(a.count).toBeLessThanOrEqual(40);
    await expect(page.getByTestId("yaml-panel")).toHaveCount(0);
  });
});

test.describe("UX2 task script B", () => {
  test("pisa_like → Run → Repeat ×200 → hit rate ± MCSE in ≤ 12 actions", async ({ page }) => {
    await firstLaunch(page);
    const a = actionCounter();
    await a.click(page.getByRole("button", { name: "Open pisa_like", exact: true }), "open");
    await a.click(page.getByRole("button", { name: "Run", exact: true }), "run");
    await expect(page.getByTestId("scorecard")).toBeVisible({ timeout: 60_000 });
    await a.click(page.getByTestId("results").getByRole("button", { name: "Repeat ×200" }), "repeat");
    await expect(page.getByTestId("headline-hit-rate")).toHaveText(/^\d\.\d{3} ± \d\.\d{3}$/, {
      timeout: 120_000,
    });
    expect(a.count).toBeLessThanOrEqual(12);
    await page.getByRole("button", { name: "Cancel" }).click();
  });
});

const VIEWS: { tab: string; ready: string }[] = [
  { tab: "World", ready: "world-canvas" },
  { tab: "Researcher's Model", ready: "model-canvas" },
  { tab: "Search & method", ready: "descriptors" },
  { tab: "Results", ready: "results" },
  { tab: "Experiments", ready: "launcher" },
];

test.describe("UX3 accessibility", () => {
  test("no serious/critical axe violations on every route", async ({ page }) => {
    await firstLaunch(page);
    expect(await seriousAxe(page)).toEqual([]);
    await page.getByRole("button", { name: "Open minimal", exact: true }).click();
    await page.getByRole("button", { name: "Run", exact: true }).click();
    await expect(page.getByTestId("scorecard")).toBeVisible({ timeout: 60_000 });
    for (const v of VIEWS) {
      await page.getByRole("tab", { name: v.tab }).click();
      await expect(page.getByTestId(v.ready)).toBeVisible();
      expect(await seriousAxe(page), v.tab).toEqual([]);
    }
  });

  test("inspector is keyboard-operable with a visible focus ring", async ({ page }) => {
    await openPreset(page, "minimal");
    await page.locator("body").press("Tab");
    let ok = false;
    for (let i = 0; i < 80 && !ok; i++) {
      ok =
        (await page.evaluate(() => document.activeElement?.getAttribute("aria-label"))) ===
        "Arrow feedback -> selfeff";
      if (!ok) await page.keyboard.press("Tab");
    }
    expect(ok).toBe(true);
    const ring = await page.evaluate(() => getComputedStyle(document.activeElement as Element).outlineStyle);
    expect(ring).not.toBe("none");
    await page.keyboard.press("Enter");
    const slope = inspector(page).getByLabel("Slope", { exact: true });
    for (let i = 0; i < 120 && !(await slope.evaluate((e) => e === document.activeElement)); i++)
      await page.keyboard.press("Tab");
    await expect(slope).toBeFocused();
    await page.keyboard.press("ControlOrMeta+a");
    await page.keyboard.type("0.41");
    await page.keyboard.press("Enter");
    await expect(outline(page).getByRole("button", { name: "Arrow feedback -> selfeff" })).toContainText(
      "0.41",
    );
  });
});

test.describe("UX4 layout and contrast", () => {
  for (const width of [1280, 1440, 1920]) {
    for (const theme of ["light", "dark"] as const) {
      test(`${width}px, ${theme}: no horizontal scroll, AA contrast`, async ({ page }) => {
        await page.setViewportSize({ width, height: 900 });
        await page.emulateMedia({ colorScheme: theme });
        await openPreset(page, "pisa_like");
        await page.getByLabel("Theme").selectOption(theme);
        await page.getByRole("button", { name: "Run", exact: true }).click();
        await expect(page.getByTestId("scorecard")).toBeVisible({ timeout: 60_000 });
        for (const v of VIEWS) {
          await page.getByRole("tab", { name: v.tab }).click();
          await expect(page.getByTestId(v.ready)).toBeVisible();
          const overflow = await page.evaluate(() => {
            const vw = window.innerWidth;
            const page = document.documentElement.scrollWidth > vw;
            const header = document.querySelector("header")!;
            const clipped = [...header.querySelectorAll("button, select, [role=tab]")].filter(
              (e) => e.getBoundingClientRect().right > vw + 0.5,
            ).length;
            return { page, clipped, headerOverflow: header.scrollWidth > header.clientWidth };
          });
          expect(overflow, `${v.tab}`).toEqual({ page: false, clipped: 0, headerOverflow: false });
          expect(await seriousAxe(page, { contrastOnly: true }), `${v.tab} contrast`).toEqual([]);
        }
      });
    }
  }
});

test.describe("UX5 glossary tooltips", () => {
  test("every term's tooltip equals its glossary definition", async ({ page, request }) => {
    const glossary: Record<string, { definition: string; amendment?: string }> = await (
      await request.get("/api/glossary")
    ).json();
    await openPreset(page, "minimal");
    await page.getByRole("button", { name: "Glossary" }).click();
    const panel = page.getByTestId("glossary");
    for (const [term, e] of Object.entries(glossary)) {
      const trigger = panel.locator(`[data-glossary="${term.replace(/"/g, '\\"')}"]`);
      await trigger.focus();
      const tip = page.getByRole("tooltip");
      await expect(tip).toHaveText(e.amendment ? `${e.definition} ${e.amendment}` : e.definition);
      await trigger.blur();
    }
  });
});

test.describe("UX6 actionable errors", () => {
  test("an infeasible target shows t_min inline and in the issues panel with show-on-canvas", async ({
    page,
  }) => {
    await openPreset(page, "minimal");
    await outline(page).getByRole("button", { name: "Latent achievement" }).click();
    await inspector(page).getByLabel("Target SD").fill("0.2");
    await inspector(page).getByLabel("Target SD").press("Enter");
    await expect(page.getByTestId("inline-issues")).toContainText("t_min = 0.376");
    const issue = page.getByTestId("issue").first();
    await expect(issue.getByTestId("issue-element")).toHaveText("node “achievement”");
    await page.getByRole("button", { name: "← World settings" }).click();
    await issue.getByRole("button", { name: "Show on canvas" }).click();
    await expect(inspector(page).getByLabel("Target SD")).toBeFocused();
  });
});
