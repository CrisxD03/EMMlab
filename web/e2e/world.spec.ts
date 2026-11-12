import { expect, test } from "@playwright/test";
import { dragConnect, editToRender, firstLaunch, inspector, openPreset, outline } from "./helpers";

test.describe("U1 canvas editing in a real browser", () => {
  test("draws arrows and level shifts by dragging between handles", async ({ page }) => {
    await firstLaunch(page);
    await page.getByRole("button", { name: "Start blank" }).click();
    for (const n of ["a", "b"]) {
      await outline(page).getByRole("button", { name: "+ Latent" }).click();
      await inspector(page).getByLabel("Name", { exact: true }).fill(n);
      await inspector(page).getByLabel("Name", { exact: true }).press("Enter");
    }
    await outline(page).getByRole("button", { name: "+ Categorical attribute" }).click();
    await inspector(page).getByLabel("Name", { exact: true }).fill("g");
    await inspector(page).getByLabel("Name", { exact: true }).press("Enter");
    await page.getByRole("button", { name: "Tidy layout" }).click();
    await page.waitForTimeout(400);
    // latent -> latent = arrow
    await dragConnect(
      page,
      page.locator('[data-testid="node-a"] .react-flow__handle[data-handleid=out]'),
      page.locator('[data-testid="node-b"] .react-flow__handle[data-handleid=in]'),
    );
    await expect(page.getByTestId("edge-a -> b")).toBeVisible();
    // attribute -> latent = level shift
    await dragConnect(
      page,
      page.locator('[data-testid="node-g"] .react-flow__handle[data-handleid=out]'),
      page.locator('[data-testid="node-b"] .react-flow__handle[data-handleid=shift]'),
    );
    await expect(page.getByTestId("shift-0")).toBeVisible();
    // the new shift is selected with its delta field focused
    await expect(inspector(page).getByLabel("Delta (added to the intercept)")).toBeFocused();
  });
});

test.describe("U4 cycles are refused while dragging", () => {
  test("a connection that would close a cycle is not created", async ({ page }) => {
    await openPreset(page, "minimal");
    await page.getByRole("button", { name: "Tidy layout" }).click();
    await page.waitForTimeout(400);
    const edges = await page.locator(".react-flow__edge").count();
    await dragConnect(
      page,
      page.locator('[data-testid="node-achievement"] .react-flow__handle[data-handleid=out]'),
      page.locator('[data-testid="node-feedback"] .react-flow__handle[data-handleid=in]'),
    );
    await expect(page.locator(".react-flow__edge")).toHaveCount(edges);
    await expect(page.getByTestId("edge-achievement -> feedback")).toHaveCount(0);
  });
});

test.describe("U3 ripple kinds are visible as CSS state", () => {
  test("slope edit animates the edge; noise edit marks standardized coefficients only", async ({ page }) => {
    await openPreset(page, "minimal");
    await outline(page).getByRole("button", { name: "Arrow feedback -> selfeff" }).click();
    await inspector(page).getByLabel("Slope", { exact: true }).fill("0.45");
    await inspector(page).getByLabel("Slope", { exact: true }).press("Enter");
    await expect(page.getByTestId("edge-feedback -> selfeff")).toHaveAttribute("data-ripple", "slope");
    await outline(page).getByRole("button", { name: "Latent selfeff" }).click();
    await inspector(page).getByRole("textbox", { name: "Noise SD" }).fill("1.3");
    await inspector(page).getByRole("textbox", { name: "Noise SD" }).press("Enter");
    await expect(page.locator('[data-ripple="spread"]').first()).toBeVisible();
    await expect(page.locator('[data-ripple="slope"]')).toHaveCount(0);
  });
});

test.describe("U2 live ripple timing", () => {
  test("edit-to-render p95 < 150 ms on pisa_like", async ({ page }) => {
    await openPreset(page, "pisa_like");
    await outline(page).getByRole("button", { name: "Arrow feedback -> enjoyment" }).click();
    await expect(page.getByTestId("total-feedback -> enjoyment")).toBeVisible();
    const times: number[] = [];
    for (let i = 0; i < 20; i++) {
      times.push(
        await editToRender(page, "Slope", (0.2 + i * 0.013).toFixed(3), "total-feedback -> enjoyment"),
      );
    }
    times.sort((a, b) => a - b);
    const p95 = times[Math.ceil(0.95 * times.length) - 1];
    test.info().annotations.push({ type: "U2 p95 ms", description: p95.toFixed(1) });
    console.log(
      `U2 edit-to-render p95 = ${p95.toFixed(1)} ms (${times.map((t) => t.toFixed(0)).join(", ")})`,
    );
    expect(p95).toBeLessThan(150);
  });
});

test.describe("U5/U6 YAML sync and undo in a real browser", () => {
  test("YAML edits apply to the canvas; Ctrl/Cmd+Z undoes", async ({ page }) => {
    await openPreset(page, "minimal");
    await page.getByRole("button", { name: "YAML", exact: true }).click();
    const editor = page.locator('[data-testid="yaml-editor"] .cm-content');
    await expect(editor).toContainText("feedback -> achievement");
    await outline(page).getByRole("button", { name: "Arrow selfeff -> achievement" }).click();
    await inspector(page).getByLabel("Slope", { exact: true }).fill("0.35");
    await inspector(page).getByLabel("Slope", { exact: true }).press("Enter");
    await expect(editor).toContainText("slope: 0.35");
    await page.locator("body").click({ position: { x: 5, y: 300 } });
    await page.keyboard.press("ControlOrMeta+z");
    await expect(outline(page).getByRole("button", { name: "Arrow selfeff -> achievement" })).toContainText(
      "0.25",
    );
  });
});

test.describe("U13 KaTeX equations", () => {
  test("renders node equations matching the server", async ({ page, request }) => {
    await openPreset(page, "minimal");
    const presets = await (await request.get("/api/presets")).json();
    const world = presets.find((p: { name: string }) => p.name === "minimal").world;
    const truth = await (await request.post("/api/world/truth", { data: { world } })).json();
    const shown = await page
      .locator("[data-testid=world-equations] [data-equation]")
      .evaluateAll((els) => els.map((e) => e.getAttribute("data-equation")));
    expect(shown.sort()).toEqual([...truth.equations].sort());
    await expect(page.locator("[data-testid=world-equations] .katex").first()).toBeVisible();
  });
});
