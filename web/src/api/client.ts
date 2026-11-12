import type {
  AnalysisConfig,
  ContrastResult,
  Descriptor,
  EstimandsResponse,
  ExperimentSpec,
  Glossary,
  Issue,
  Job,
  JobSummary,
  MethodPreset,
  Preset,
  QualityInfo,
  ResearcherModel,
  RunResult,
  RunSpec,
  ScatterResponse,
  SearchSpace,
  TruthResponse,
  World,
} from "./types";

export class ApiError extends Error {
  status: number;
  issues: Issue[];
  constructor(status: number, detail: unknown) {
    const issues = extractIssues(detail);
    super(issues.length ? issues.map((i) => `${i.element}: ${i.message}`).join("; ") : `HTTP ${status}`);
    this.status = status;
    this.issues = issues;
  }
}

function extractIssues(detail: unknown): Issue[] {
  const d = (detail as { detail?: unknown })?.detail ?? detail;
  if (d && typeof d === "object" && "issues" in d && Array.isArray((d as { issues: unknown }).issues)) {
    return (d as { issues: Issue[] }).issues;
  }
  if (typeof d === "string") return [{ element: "server", message: d }];
  if (Array.isArray(d)) {
    return d.map((e: { loc?: unknown[]; msg?: string }) => ({
      element: (e.loc ?? []).join("."),
      message: e.msg ?? String(e),
    }));
  }
  return [];
}

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const r = await fetch(path, {
    method,
    headers: body === undefined ? undefined : { "content-type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const ct = r.headers.get("content-type") ?? "";
  const data = ct.includes("json") ? await r.json() : await r.text();
  if (!r.ok) throw new ApiError(r.status, data);
  return data as T;
}

const get = <T>(path: string) => request<T>("GET", path);
const post = <T>(path: string, body: unknown = {}) => request<T>("POST", path, body);

export const api = {
  presets: () => get<Preset[]>("/api/presets"),
  glossary: () => get<Glossary>("/api/glossary"),
  qualities: () => get<QualityInfo[]>("/api/qualities"),
  truth: (world: World) => post<TruthResponse>("/api/world/truth", { world }),
  yaml: (world: World, text: string) =>
    post<{ text: string; hash: string }>("/api/world/yaml", { world, text }),
  parse: (text: string) =>
    post<{ ok: boolean; world?: World; hash?: string; issues?: Issue[] }>("/api/world/parse", { text }),
  nullWorld: (world: World) => post<{ world: World }>("/api/world/null", { world }),
  modelFromWorld: (world: World) => post<ResearcherModel>("/api/model/from_world", { world }),
  estimands: (world: World, model: ResearcherModel, analysis: AnalysisConfig) =>
    post<EstimandsResponse>("/api/model/estimands", { world, model, analysis }),
  descriptors: (world: World, model: ResearcherModel, search: SearchSpace) =>
    post<Record<string, Descriptor>>("/api/search/descriptors", { world, model, analysis: { search } }),
  methodPreset: (name: "current" | "recommended") => get<MethodPreset>(`/api/preset_method/${name}`),
  run: (spec: RunSpec) => post<RunResult>("/api/run", { spec }),
  runReportUrl: (key: string) => `/api/run/${key}/report`,
  scatter: (key: string, path: string, rank: number) =>
    get<ScatterResponse>(`/api/run/${key}/scatter?path=${encodeURIComponent(path)}&rank=${rank}`),
  contrast: (key: string, rank: number) => get<ContrastResult>(`/api/run/${key}/contrast?rank=${rank}`),
  createExperiment: (spec: ExperimentSpec, workers?: number) =>
    post<{ id: string; total: number }>("/api/experiments", { spec, workers: workers ?? null }),
  experiments: () => get<JobSummary[]>("/api/experiments"),
  experiment: (id: string) => get<Job>(`/api/experiments/${id}`),
  cancelExperiment: (id: string) => post<{ ok: boolean }>(`/api/experiments/${id}/cancel`),
  resumeExperiment: (id: string) => post<{ ok: boolean; status: string }>(`/api/experiments/${id}/resume`),
  experimentEventsUrl: (id: string) => `/api/experiments/${id}/events`,
  experimentReportUrl: (id: string) => `/api/experiments/${id}/report`,
  text: (path: string) => request<string>("GET", path),
};

/** Fetch a self-contained HTML report and save it as a file (U11). */
export async function downloadReport(url: string, filename: string): Promise<string> {
  const html = await api.text(url);
  const blob = new Blob([html], { type: "text/html" });
  const href = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = href;
  a.download = filename;
  a.rel = "noopener";
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(href), 2000);
  return html;
}

export function downloadText(text: string, filename: string, type = "text/yaml") {
  const blob = new Blob([text], { type });
  const href = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = href;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(href), 2000);
}
