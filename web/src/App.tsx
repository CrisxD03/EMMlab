import * as Tabs from "@radix-ui/react-tabs";
import * as Tooltip from "@radix-ui/react-tooltip";
import { lazy, Suspense, useEffect } from "react";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { repeat200, runOnce } from "./features/commands";
import { Gallery } from "./features/gallery/Gallery";
import { GlossaryPanel } from "./features/glossary/GlossaryPanel";
import { deleteElement, removeArrow, removeShift, removeTwist } from "./features/world/actions";
import { WorldView } from "./features/world/WorldView";
import { useDoc } from "./state/doc";
import { TruthProvider } from "./state/truth";
import { applyTheme, type Tab, type Theme, useUI } from "./state/ui";

const ExperimentsView = lazy(() =>
  import("./features/experiments/ExperimentsView").then((m) => ({ default: m.ExperimentsView })),
);
const ModelView = lazy(() => import("./features/model/ModelView").then((m) => ({ default: m.ModelView })));
const ResultsView = lazy(() =>
  import("./features/results/ResultsView").then((m) => ({ default: m.ResultsView })),
);
const SearchView = lazy(() =>
  import("./features/search/SearchView").then((m) => ({ default: m.SearchView })),
);
const TABS: { value: Tab; label: string }[] = [
  { value: "world", label: "World" },
  { value: "model", label: "Researcher's Model" },
  { value: "search", label: "Search & method" },
  { value: "results", label: "Results" },
  { value: "experiments", label: "Experiments" },
];

function isTextTarget(t: EventTarget | null) {
  const el = t as HTMLElement | null;
  if (!el) return false;
  return (
    el.tagName === "INPUT" ||
    el.tagName === "TEXTAREA" ||
    el.tagName === "SELECT" ||
    el.isContentEditable ||
    !!el.closest?.(".cm-editor")
  );
}

function useShortcuts() {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const mod = e.metaKey || e.ctrlKey;
      const key = e.key.toLowerCase();
      if (mod && key === "enter") {
        e.preventDefault();
        void runOnce();
        return;
      }
      if (isTextTarget(e.target)) return;
      if (mod && key === "z") {
        e.preventDefault();
        if (e.shiftKey) useDoc.getState().redo();
        else useDoc.getState().undo();
        return;
      }
      if (mod && key === "y") {
        e.preventDefault();
        useDoc.getState().redo();
        return;
      }
      const ui = useUI.getState();
      if ((e.key === "Delete" || e.key === "Backspace") && ui.tab === "world") {
        const s = ui.selection;
        if (s.type === "node") deleteElement("node", s.id);
        else if (s.type === "attribute") deleteElement("attribute", s.id);
        else if (s.type === "arrow") removeArrow(s.id);
        else if (s.type === "shift") removeShift(s.index);
        else if (s.type === "twist") removeTwist(s.index);
        else return;
        e.preventDefault();
        return;
      }
      if (e.key === "Escape" && ui.tab === "world" && ui.selection.type !== "world")
        ui.select({ type: "world" });
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
}

function TopBar() {
  const theme = useUI((s) => s.theme);
  const running = useUI((s) => s.running);
  const set = useUI((s) => s.set);
  const canUndo = useDoc((s) => s.past.length > 0);
  const canRedo = useDoc((s) => s.future.length > 0);
  const lastLabel = useDoc((s) => s.lastEdit.label);
  return (
    <div className="flex items-center gap-1.5">
      <button
        type="button"
        className="btn btn-sm btn-ghost"
        disabled={!canUndo}
        onClick={() => useDoc.getState().undo()}
        aria-label="Undo"
        title={`Undo ${lastLabel} (⌘Z / Ctrl+Z)`}
      >
        ↶
      </button>
      <button
        type="button"
        className="btn btn-sm btn-ghost"
        disabled={!canRedo}
        onClick={() => useDoc.getState().redo()}
        aria-label="Redo"
        title="Redo (⇧⌘Z / Ctrl+Y)"
      >
        ↷
      </button>
      <span className="mx-1 h-5 w-px bg-line" aria-hidden />
      <button type="button" className="btn btn-sm btn-ghost" onClick={() => set({ galleryOpen: true })}>
        Presets
      </button>
      <button type="button" className="btn btn-sm btn-ghost" onClick={() => set({ glossaryOpen: true })}>
        Glossary
      </button>
      <label className="sr-only" htmlFor="theme">
        Theme
      </label>
      <select
        id="theme"
        className="field"
        style={{ width: 92, height: 26 }}
        value={theme}
        onChange={(e) => {
          const t = e.target.value as Theme;
          set({ theme: t });
          applyTheme(t);
        }}
      >
        <option value="system">System</option>
        <option value="light">Light</option>
        <option value="dark">Dark</option>
      </select>
      <span className="mx-1 h-5 w-px bg-line" aria-hidden />
      <button
        type="button"
        className="btn btn-primary"
        onClick={() => runOnce()}
        disabled={running}
        title="One seed: one draw, not evidence (⌘↵)"
      >
        {running ? "Running…" : "Run"}
      </button>
      <button
        type="button"
        className="btn"
        onClick={() => repeat200()}
        title="Replicate the run 200 times with its null World"
      >
        Repeat ×200
      </button>
    </div>
  );
}

export function App() {
  const tab = useUI((s) => s.tab);
  const set = useUI((s) => s.set);
  const theme = useUI((s) => s.theme);
  // While a dialog is open the app behind it is inert (not focusable, hidden from assistive tech).
  const dialogOpen = useUI((s) => s.galleryOpen || s.glossaryOpen);
  useShortcuts();
  useEffect(() => applyTheme(theme), [theme]);
  return (
    <Tooltip.Provider delayDuration={250} skipDelayDuration={100}>
      <TruthProvider>
        <Tabs.Root
          value={tab}
          onValueChange={(v) => set({ tab: v as Tab })}
          className="flex h-full flex-col"
          inert={dialogOpen}
        >
          <header className="flex h-12 shrink-0 items-center gap-4 border-line border-b bg-panel px-3">
            <div className="flex items-baseline gap-1.5">
              <span className="font-bold text-[15px] tracking-tight">emmlab</span>
              <span className="hidden text-[11.5px] text-faint 2xl:inline">ground-truth lab for EMM</span>
            </div>
            <Tabs.List className="flex h-full items-stretch gap-0.5" aria-label="Views">
              {TABS.map((t) => (
                <Tabs.Trigger
                  key={t.value}
                  value={t.value}
                  className="border-transparent border-b-2 px-2.5 font-medium text-[13px] text-muted hover:text-fg data-[state=active]:border-accent data-[state=active]:text-fg"
                >
                  {t.label}
                </Tabs.Trigger>
              ))}
            </Tabs.List>
            <span className="flex-1" />
            <TopBar />
          </header>
          <main className="min-h-0 flex-1">
            {TABS.map((t) => (
              <Tabs.Content key={t.value} value={t.value} className="h-full outline-none" tabIndex={-1}>
                <ErrorBoundary label={t.label}>
                  <Suspense fallback={<div className="p-6 text-muted">Loading…</div>}>
                    {t.value === "world" ? <WorldView /> : null}
                    {t.value === "model" ? <ModelView /> : null}
                    {t.value === "search" ? <SearchView /> : null}
                    {t.value === "results" ? <ResultsView /> : null}
                    {t.value === "experiments" ? <ExperimentsView /> : null}
                  </Suspense>
                </ErrorBoundary>
              </Tabs.Content>
            ))}
          </main>
        </Tabs.Root>
        <Gallery />
        <GlossaryPanel />
      </TruthProvider>
    </Tooltip.Provider>
  );
}
