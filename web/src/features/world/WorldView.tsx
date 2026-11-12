import { ReactFlowProvider } from "@xyflow/react";
import { lazy, Suspense, useRef, useState } from "react";
import { api, downloadText } from "../../api/client";
import type { Issue } from "../../api/types";
import { EquationViewSwitch } from "../../components/NodeEquationView";
import { Term } from "../../components/ui";
import { useDoc } from "../../state/doc";
import { useTruth } from "../../state/truth";
import { shiftEdgesShown, useUI } from "../../state/ui";
import { yamlText } from "../../state/yamlText";
import { loadWorld } from "../commands";
import { Inspector } from "./Inspector";
import { IssuesPanel } from "./IssuesPanel";
import { Outline } from "./Outline";
import { WorldCanvas } from "./WorldCanvas";

const YamlPanel = lazy(() => import("./YamlPanel").then((m) => ({ default: m.YamlPanel })));

export function WorldView() {
  const yamlOpen = useUI((s) => s.yamlOpen);
  const set = useUI((s) => s.set);
  const world = useDoc((s) => s.doc.world);
  const shiftsShown = useUI((s) => shiftEdgesShown(s.shiftEdges, world.shifts.length));
  const { issues, hash, pending, data } = useTruth();
  const fileRef = useRef<HTMLInputElement>(null);
  const [importIssues, setImportIssues] = useState<Issue[]>([]);

  const exportYaml = async () => {
    const r = await api.yaml(world, yamlText.current);
    yamlText.current = r.text;
    downloadText(r.text, `${world.name || "world"}.yaml`);
  };
  const importYaml = async (file: File) => {
    const text = await file.text();
    const r = await api.parse(text);
    if (!r.ok || !r.world) {
      setImportIssues(r.issues ?? [{ element: "yaml", message: "could not read the file" }]);
      return;
    }
    setImportIssues([]);
    await loadWorld(r.world, `import ${file.name}`, text);
  };

  const warnings = data?.warnings ?? [];
  return (
    <div
      className="grid h-full min-h-0"
      style={{
        gridTemplateColumns: yamlOpen
          ? "196px minmax(0,1fr) minmax(300px,30%) 340px"
          : "196px minmax(0,1fr) 340px",
      }}
    >
      <Outline />
      <section className="flex min-h-0 min-w-0 flex-col" aria-label="World canvas area">
        <div className="flex h-10 shrink-0 items-center gap-2 border-line border-b bg-panel px-3">
          <h1 className="m-0 truncate font-semibold text-[14px]">
            <Term term="World" />: {world.name}
          </h1>
          <span className="text-[11.5px] text-faint" data-testid="world-hash" data-hash={hash ?? ""}>
            {pending ? "updating…" : hash ? `spec ${hash}` : issues.length ? "invalid" : ""}
          </span>
          {warnings.length ? (
            <span className="badge badge-warn" title={warnings.join("\n")}>
              {warnings.length} warning{warnings.length > 1 ? "s" : ""}
            </span>
          ) : null}
          <span className="flex-1" />
          <EquationViewSwitch />
          {world.shifts.length ? (
            <button
              type="button"
              className="btn btn-sm"
              aria-pressed={shiftsShown}
              title="Draw level shifts as edges from the attribute lane, or as badges on the shifted nodes"
              onClick={() => {
                set({ shiftEdges: shiftsShown ? "hide" : "show", keepLayoutDir: true });
                setTimeout(() => set({ layoutSeq: useUI.getState().layoutSeq + 1 }), 50);
              }}
            >
              {shiftsShown ? "Shifts as badges" : "Shifts as edges"}
            </button>
          ) : null}
          <button
            type="button"
            className="btn btn-sm"
            aria-pressed={yamlOpen}
            onClick={() => set({ yamlOpen: !yamlOpen })}
          >
            {yamlOpen ? "Hide YAML" : "YAML"}
          </button>
          <button type="button" className="btn btn-sm" onClick={() => fileRef.current?.click()}>
            Import…
          </button>
          <input
            ref={fileRef}
            type="file"
            accept=".yaml,.yml,text/yaml"
            className="sr-only"
            aria-label="Import World YAML file"
            data-testid="import-yaml"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) void importYaml(f);
              e.target.value = "";
            }}
          />
          <button type="button" className="btn btn-sm" onClick={exportYaml}>
            Export YAML
          </button>
        </div>
        <div className="relative min-h-0 flex-1">
          <ReactFlowProvider>
            <WorldCanvas />
          </ReactFlowProvider>
          <IssuesPanel issues={[...importIssues, ...issues]} />
        </div>
      </section>
      {yamlOpen ? (
        <Suspense
          fallback={<div className="border-line border-l bg-panel p-4 text-muted">Loading editor…</div>}
        >
          <YamlPanel />
        </Suspense>
      ) : null}
      <Inspector />
    </div>
  );
}
