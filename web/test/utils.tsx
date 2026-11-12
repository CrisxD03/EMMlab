import { QueryClientProvider } from "@tanstack/react-query";
import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect } from "vitest";
import { App } from "../src/App";
import { queryClient } from "../src/queryClient";
import { blankDoc, useDoc } from "../src/state/doc";
import { useUI } from "../src/state/ui";
import type { Bridge } from "./bridge";

export const bridge = (globalThis as unknown as { __bridge: Bridge }).__bridge;

const initialUI = { ...useUI.getState() };

/** Fresh app as on first launch: empty storage, preset gallery open. */
export function renderApp({ firstLaunch = true } = {}) {
  try {
    localStorage.clear();
  } catch {
    /* ignore */
  }
  queryClient.clear();
  useDoc.getState().reset(blankDoc());
  useUI.setState({
    ...initialUI,
    galleryOpen: firstLaunch,
    positions: {},
    routes: {},
    ripple: { seq: 0, editKind: "load", kinds: {} },
  });
  const user = userEvent.setup({ delay: null });
  const utils = render(
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>,
  );
  return { user, ...utils };
}

/**
 * Counts UI actions for the task scripts (UX1, UX2): one click, one option chosen in a select, or one
 * field entry (type a value and press Enter — Playwright's fill + Enter) is one action.
 */
export function actionCounter(user: ReturnType<typeof userEvent.setup>) {
  let n = 0;
  const log: string[] = [];
  return {
    get count() {
      return n;
    },
    log,
    async click(el: Element, what = "") {
      n++;
      log.push(`click ${what}`);
      await user.click(el);
    },
    async select(el: Element, value: string) {
      n++;
      log.push(`select ${value}`);
      await user.selectOptions(el, value);
    },
    async fill(el: Element, text: string) {
      n++;
      log.push(`fill ${text}`);
      await user.clear(el);
      await user.type(el, `${text}{Enter}`);
    },
  };
}

/** Wait until the live truth for the current World has arrived (ripple applied). */
export async function settleTruth() {
  const want = JSON.stringify(useDoc.getState().doc.world);
  await waitFor(
    () => {
      const data = queryClient.getQueryData(["truth", want]);
      expect(data).toBeTruthy();
    },
    { timeout: 15000 },
  );
  await act(async () => {
    await new Promise((r) => setTimeout(r, 30));
  });
}

export async function openPreset(
  user: ReturnType<typeof userEvent.setup>,
  name: string,
  variant: "planted" | "null" = "planted",
) {
  const gallery = await screen.findByTestId("gallery", {}, { timeout: 20000 });
  const card = await within(gallery).findByTestId(`preset-${name}`, {}, { timeout: 20000 });
  await user.click(
    within(card).getByRole("button", {
      name: variant === "null" ? `Open ${name} null variant` : `Open ${name}`,
    }),
  );
  await waitFor(() => expect(screen.queryByTestId("gallery")).toBeNull(), { timeout: 20000 });
  await waitFor(() => expect(useDoc.getState().doc.world.name).toContain(name), { timeout: 20000 });
  await settleTruth();
}

/** Put text into the CodeMirror YAML editor as a user would (replace the whole document). */
export async function typeYaml(text: string) {
  const { EditorView } = await import("@codemirror/view");
  const el = await screen.findByTestId("yaml-editor");
  const cm = el.querySelector(".cm-editor") as HTMLElement;
  const view = EditorView.findFromDOM(cm)!;
  act(() => {
    view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: text } });
  });
}

export function yamlEditorText(): string | null {
  const el = screen.queryByTestId("yaml-editor");
  if (!el) return null;
  return (el.querySelector(".cm-content") as HTMLElement | null)?.innerText ?? el.textContent;
}

export async function yamlDocText(): Promise<string> {
  const { EditorView } = await import("@codemirror/view");
  const el = await screen.findByTestId("yaml-editor");
  const view = EditorView.findFromDOM(el.querySelector(".cm-editor") as HTMLElement)!;
  return view.state.doc.toString();
}

export function outline() {
  return within(screen.getByRole("navigation", { name: "World outline" }));
}

export function inspector() {
  return within(screen.getByTestId("inspector"));
}
