import { readFileSync } from "node:fs";
import { expect, test } from "@playwright/test";
import { firstLaunch, openPreset } from "./helpers";

test.describe("U12 preset gallery and one-click Run", () => {
  test("first launch shows the gallery; minimal + Run gives a scorecard", async ({ page }) => {
    await firstLaunch(page);
    for (const p of ["minimal", "pisa_like", "simpson", "level_shift_only", "variance_decoy"])
      await expect(page.getByTestId(`preset-${p}`)).toBeVisible();
    await page.getByRole("button", { name: "Open minimal", exact: true }).click();
    await page.getByRole("button", { name: "Run", exact: true }).click();
    await expect(page.getByTestId("scorecard")).toContainText("one draw — not evidence", { timeout: 60_000 });
  });
});

test.describe("U7 Researcher's Model", () => {
  test("click a node to residualize: triangle keeps R²", async ({ page }) => {
    await openPreset(page, "minimal");
    await page.getByRole("tab", { name: "Researcher's Model" }).click();
    await page.getByRole("button", { name: "Copy from World" }).click();
    await page.getByTestId("model-node-feedback").click();
    await expect(page.getByTestId("case-achievement")).toHaveText("triangle");
    await page.getByRole("button", { name: "Remove model arrow feedback -> achievement" }).click();
    await expect(page.getByTestId("case-achievement")).toHaveText("chain");
  });
});

test.describe("U8 Search Space", () => {
  test("histogram with cut points for a numeric attribute", async ({ page }) => {
    await openPreset(page, "pisa_like");
    await page.getByRole("tab", { name: "Search & method" }).click();
    await expect(page.getByTestId("hist-school_size").locator("svg")).toBeVisible({ timeout: 30_000 });
    await expect(page.getByTestId("cuts-school_size")).toContainText("cut points");
  });
});

test.describe("U9 Results", () => {
  test("changed arrow highlighted, coefficient hover shows truths and provenance, scatter per group", async ({
    page,
  }) => {
    await openPreset(page, "minimal");
    await page.getByRole("button", { name: "Run", exact: true }).click();
    await expect(page.getByTestId("scorecard")).toBeVisible({ timeout: 60_000 });
    const edge = page.getByTestId("diagram-edge-feedback -> achievement");
    await expect(edge).toHaveAttribute("data-changed", "true");
    await edge.getByRole("button").first().hover();
    const tip = page.getByRole("tooltip");
    for (const s of ["T1 structural", "T2 causal target", "T3 step estimand", "provenance"])
      await expect(tip).toContainText(s);
    await expect(page.getByTestId("scatter-plot").locator("circle").first()).toBeVisible({ timeout: 20_000 });
  });
});

test.describe("U10 Experiments", () => {
  test("launch a sweep, watch progress, see MCSE bands", async ({ page }) => {
    await openPreset(page, "minimal");
    await page.getByRole("tab", { name: "Experiments" }).click();
    const l = page.getByTestId("launcher");
    await l.getByRole("button", { name: "+ Factor (sweep)" }).click();
    await l.getByLabel("Levels of factor 1 (comma-separated)").fill("40, 80");
    await l.getByLabel("Levels of factor 1 (comma-separated)").press("Enter");
    await l.getByLabel("Replications (n_sim)").fill("4");
    await l.getByLabel("Replications (n_sim)").press("Enter");
    await l.getByRole("button", { name: "Start experiment" }).click();
    await expect(page.getByTestId("job-status")).toHaveText("done", { timeout: 120_000 });
    // the plot svg has a viewBox; legend swatches are small svgs without one
    await expect(page.getByTestId("sweep-plot").locator("svg[viewBox]").first()).toBeVisible();
    await expect(page.getByTestId("hit-rate").first()).toContainText("±");
  });
});

test.describe("U11 self-contained reports", () => {
  test("downloaded run report opens offline with everything inline", async ({ page, browser }) => {
    await openPreset(page, "minimal");
    await page.getByRole("button", { name: "Run", exact: true }).click();
    await expect(page.getByTestId("scorecard")).toBeVisible({ timeout: 60_000 });
    const [download] = await Promise.all([
      page.waitForEvent("download"),
      page.getByRole("button", { name: "Download report (.html)" }).click(),
    ]);
    const path = await download.path();
    const html = readFileSync(path, "utf8");
    expect(html).not.toMatch(/(src|href)=["']?https?:/i);
    const ctx = await browser.newContext({ offline: true });
    const p = await ctx.newPage();
    const failed: string[] = [];
    p.on("requestfailed", (r) => failed.push(r.url()));
    await p.goto(`file://${path}`);
    await expect(p.getByText("one draw — not evidence")).toBeVisible();
    expect(failed).toEqual([]);
    await ctx.close();
  });
});

test("reload keeps the document (no gallery on second launch)", async ({ page }) => {
  await firstLaunch(page);
  await page.getByRole("button", { name: "Open simpson", exact: true }).click();
  await page.waitForTimeout(600);
  await page.reload();
  await expect(page.getByTestId("gallery")).toBeHidden();
  await expect(page.getByRole("heading", { level: 1 })).toContainText("simpson");
});
