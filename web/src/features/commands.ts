// App-level commands shared by the top bar, keyboard shortcuts and views.

import { ApiError, api } from "../api/client";
import type { ExperimentSpec, World } from "../api/types";
import { queryClient } from "../queryClient";
import { DEFAULT_SEARCH, RECOMMENDED, runSpecOf, useDoc } from "../state/doc";
import { useUI } from "../state/ui";
import { yamlText } from "../state/yamlText";

/** Replace the document with a World (preset, null variant, import) and lay it out. */
export async function loadWorld(world: World, label: string, yaml = "") {
  yamlText.current = yaml;
  let model = { name: `${world.name}_model`, latents: [] as string[], arrows: [] as string[] };
  try {
    model = { ...(await api.modelFromWorld(world)) } as typeof model;
  } catch {
    /* invalid World: keep an empty model, the issues panel explains */
  }
  useDoc.getState().load(
    {
      world,
      model,
      method: structuredClone(RECOMMENDED),
      search: structuredClone(DEFAULT_SEARCH),
      seed: 0,
    },
    label,
  );
  const ui = useUI.getState();
  ui.set({
    positions: {},
    routes: {},
    layoutSeq: ui.layoutSeq + 1,
    run: null,
    runIssues: [],
    galleryOpen: false,
    tab: "world",
    selection: { type: "world" },
    highlight: [],
  });
}

export async function ensureModel() {
  const { doc, apply } = useDoc.getState();
  if (doc.model.latents.length) return;
  const model = await api.modelFromWorld(doc.world);
  apply("copy model from World", "other", (d) => {
    d.model = model;
  });
}

let running = false;

/** "Run": one seed, one draw — not evidence. */
export async function runOnce() {
  if (running) return;
  running = true;
  const ui = useUI.getState();
  ui.set({ runIssues: [] });
  useUI.getState().set({ running: true });
  try {
    await ensureModel();
    const r = await api.run(runSpecOf(useDoc.getState().doc));
    useUI
      .getState()
      .set({ run: r, tab: "results", selectedFinding: r.emm.findings.length ? 1 : 0, runIssues: [] });
  } catch (e) {
    const issues =
      e instanceof ApiError && e.issues.length
        ? e.issues
        : [{ element: "run", message: String((e as Error).message ?? e) }];
    useUI.getState().set({ runIssues: issues });
  } finally {
    running = false;
    useUI.getState().set({ running: false });
  }
}

export function experimentSpec(over: Partial<ExperimentSpec> = {}): ExperimentSpec {
  const { doc } = useDoc.getState();
  return {
    name: `${doc.world.name} experiment`,
    base: runSpecOf(doc),
    presets: [doc.method.name === "current" ? "current" : "recommended"],
    factors: [],
    n_sim: 200,
    include_null_world: true,
    ...over,
  };
}

/** "Repeat ×200": the same Run replicated 200 times (with its null World) as an Experiment. */
export async function repeat200() {
  await ensureModel();
  const spec = experimentSpec({ name: `${useDoc.getState().doc.world.name} · repeat ×200`, n_sim: 200 });
  const r = await api.createExperiment(spec);
  useUI.getState().set({ experimentId: r.id, tab: "experiments" });
  await queryClient.invalidateQueries({ queryKey: ["experiments"] });
  return r.id;
}
