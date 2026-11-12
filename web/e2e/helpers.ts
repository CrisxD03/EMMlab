import AxeBuilder from "@axe-core/playwright";
import { expect, type Locator, type Page } from "@playwright/test";

/** First launch: fresh storage, gallery open. */
export async function firstLaunch(page: Page) {
  await page.goto("/");
  await page.evaluate(() => localStorage.clear());
  await page.reload();
  await expect(page.getByTestId("gallery")).toBeVisible();
}

export async function openPreset(page: Page, name: string) {
  await firstLaunch(page);
  await page.getByRole("button", { name: `Open ${name}`, exact: true }).click();
  await expect(page.getByTestId("gallery")).toBeHidden();
  await expect(page.getByTestId("world-hash")).toHaveText(/spec [0-9a-f]{16}/);
}

export const outline = (page: Page) => page.getByRole("navigation", { name: "World outline" });
export const inspector = (page: Page) => page.getByTestId("inspector");

/** One action = one click, one option chosen, or one field entry (fill + Enter). */
export function actionCounter() {
  let n = 0;
  const log: string[] = [];
  return {
    get count() {
      return n;
    },
    log,
    async click(l: Locator, what = "") {
      n++;
      log.push(`click ${what}`);
      await l.click();
    },
    async select(l: Locator, value: string) {
      n++;
      log.push(`select ${value}`);
      await l.selectOption(value);
    },
    async fill(l: Locator, text: string) {
      n++;
      log.push(`fill ${text}`);
      await l.fill(text);
      await l.press("Enter");
    },
  };
}

export async function seriousAxe(page: Page, opts: { contrastOnly?: boolean } = {}) {
  let b = new AxeBuilder({ page });
  b = opts.contrastOnly
    ? b.withRules(["color-contrast"])
    : b.withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"]);
  const r = await b.analyze();
  return r.violations
    .filter((v) => opts.contrastOnly || v.impact === "serious" || v.impact === "critical")
    .map(
      (v) =>
        `${v.id}: ${v.nodes
          .map((n) => n.target.join(" "))
          .slice(0, 3)
          .join(" | ")}`,
    );
}

/** Drag from one canvas handle to another with intermediate mouse moves (React Flow needs them). */
export async function dragConnect(page: Page, from: Locator, to: Locator) {
  const a = await from.boundingBox();
  const b = await to.boundingBox();
  if (!a || !b) throw new Error("handle not visible");
  await page.mouse.move(a.x + a.width / 2, a.y + a.height / 2);
  await page.mouse.down();
  await page.mouse.move((a.x + b.x) / 2, (a.y + b.y) / 2, { steps: 8 });
  await page.mouse.move(b.x + b.width / 2, b.y + b.height / 2, { steps: 8 });
  await page.mouse.up();
}

/** Time from pressing Enter on an inspector field to the given derived value changing on screen. */
export async function editToRender(
  page: Page,
  fieldLabel: string,
  value: string,
  watchTestId: string,
): Promise<number> {
  return page.evaluate(
    ({ fieldLabel, value, watchTestId }) =>
      new Promise<number>((resolve, reject) => {
        const label = [...document.querySelectorAll("[data-testid=inspector] label")].find(
          (l) => l.textContent === fieldLabel,
        ) as HTMLLabelElement | undefined;
        const input = label?.control as HTMLInputElement | null;
        if (!input) return reject(new Error(`no field ${fieldLabel}`));
        const read = () =>
          document.querySelector(`[data-testid="${watchTestId}"]`)?.getAttribute("data-value");
        const before = read();
        const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!;
        setter.call(input, value);
        input.dispatchEvent(new Event("input", { bubbles: true }));
        requestAnimationFrame(() => {
          const t0 = performance.now();
          const obs = new MutationObserver(() => {
            if (read() !== before) {
              obs.disconnect();
              requestAnimationFrame(() => resolve(performance.now() - t0));
            }
          });
          obs.observe(document.body, {
            subtree: true,
            childList: true,
            attributes: true,
            characterData: true,
          });
          input.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
          setTimeout(() => {
            obs.disconnect();
            reject(new Error("no update within 10 s"));
          }, 10_000);
        });
      }),
    { fieldLabel, value, watchTestId },
  );
}
