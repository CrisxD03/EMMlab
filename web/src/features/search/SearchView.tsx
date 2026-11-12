import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { api } from "../../api/client";
import type { Binning, Descriptor, MethodPreset, QualityName } from "../../api/types";
import { Plot, PlotFigure } from "../../components/PlotFigure";
import { Checkbox, Empty, NumberField, Section, SelectField, Term } from "../../components/ui";
import { DEFAULT_SEARCH, type Doc, useDoc } from "../../state/doc";
import { useTruth } from "../../state/truth";
import { fmt } from "../../state/worldOps";
import { useQualities } from "../results/quality";

const edit = (label: string, fn: (d: Doc) => void) =>
  useDoc.getState().apply(label, "other", (d) => fn(d as Doc));

export async function choosePreset(name: "current" | "recommended") {
  const m: MethodPreset = await api.methodPreset(name);
  edit(`method preset ${name}`, (d) => {
    const res = d.method.analysis.residualize_on;
    d.method = { ...m, analysis: { ...m.analysis, residualize_on: res } };
  });
}

function MethodPanel() {
  const method = useDoc((s) => s.doc.method);
  const a = method.analysis;
  const e = method.emm;
  // the quality functions emm.QUALITY_FUNCTIONS offers in the menu, plus the one in use (it may come from YAML)
  const menu = (useQualities().data ?? []).filter((q) => q.in_menu).map((q) => q.name);
  const qualityOptions = [...new Set<QualityName>([...menu, e.quality])];
  return (
    <Section title={<Term term="Method preset" />} testId="method-panel">
      <div className="mb-3 flex gap-1.5" role="radiogroup" aria-label="Method preset">
        {(["recommended", "current"] as const).map((p) => (
          // biome-ignore lint/a11y/useSemanticElements: segmented control styled as buttons
          <button
            key={p}
            type="button"
            role="radio"
            aria-checked={method.name === p}
            className={`btn ${method.name === p ? "btn-primary" : ""}`}
            onClick={() => choosePreset(p)}
          >
            {p}
          </button>
        ))}
      </div>
      <p className="mt-0 mb-2 text-[12px] text-muted">
        {method.name === "current"
          ? "Mirrors the researcher's pipeline: per-group standardization, simple residuals, entropy × number of disjoint paths."
          : "Standardize once (common metric), total-effect residualization, entropy × Wald z."}
      </p>
      <div className="grid grid-cols-2 gap-2">
        <SelectField
          label="Normalisation"
          value={a.normalisation}
          options={[
            { value: "none", label: "none (pooled)" },
            { value: "centered", label: "school-mean centered" },
            // kept in the code, offered only when a loaded spec already uses it
            ...(a.normalisation === "within_std"
              ? [{ value: "within_std" as const, label: "within-school standardized" }]
              : []),
          ]}
          onChange={(v) =>
            edit("normalisation", (d) => {
              d.method.analysis.normalisation = v;
            })
          }
        />
        <SelectField
          label="Standardize"
          value={a.standardize}
          options={[
            { value: "global_once", label: "once, globally" },
            { value: "per_group", label: "per group" },
          ]}
          onChange={(v) =>
            edit("standardize", (d) => {
              d.method.analysis.standardize = v;
            })
          }
        />
        <SelectField
          label="Residualize method"
          value={a.residualize_method}
          options={[
            { value: "total_effect", label: "total effect" },
            { value: "simple", label: "simple residual" },
          ]}
          onChange={(v) =>
            edit("residualize method", (d) => {
              d.method.analysis.residualize_method = v;
            })
          }
        />
        <SelectField
          label="Standard errors"
          value={a.se}
          options={[
            { value: "cluster", label: "cluster-robust" },
            { value: "naive", label: "naive" },
          ]}
          onChange={(v) =>
            edit("se", (d) => {
              d.method.analysis.se = v;
            })
          }
        />
        <SelectField
          label="Quality function"
          value={e.quality}
          options={qualityOptions}
          onChange={(v) =>
            edit("quality", (d) => {
              d.method.emm.quality = v;
            })
          }
        />
        <SelectField
          label="Interval standard errors"
          value={e.interval_se}
          options={[
            { value: "analysis", label: "as the analysis (cluster-robust)" },
            { value: "naive", label: "student-level (as plspm)" },
          ]}
          onChange={(v) =>
            edit("interval standard errors", (d) => {
              d.method.emm.interval_se = v;
            })
          }
        />
        <SelectField
          label="Comparison baseline"
          value={e.baseline}
          options={["complement", "population"]}
          onChange={(v) =>
            edit("baseline", (d) => {
              d.method.emm.baseline = v;
            })
          }
        />
        <NumberField
          label="Top k findings"
          value={e.top_k}
          min={1}
          onCommit={(v) =>
            edit("top k", (d) => {
              d.method.emm.top_k = Math.round(v ?? 10);
            })
          }
        />
        <NumberField
          label="Beam width"
          value={e.beam_width}
          min={1}
          onCommit={(v) =>
            edit("beam", (d) => {
              d.method.emm.beam_width = Math.round(v ?? 20);
            })
          }
        />
        <div className="flex items-end pb-1">
          <Checkbox
            label="Drop A AND B when it does not beat A or B alone"
            checked={e.prune_refinements ?? true}
            onChange={(v) =>
              edit("prune refinements", (d) => {
                d.method.emm.prune_refinements = v;
              })
            }
          />
        </div>
      </div>
      <p className="mt-2 mb-0 text-[11.5px] text-faint">
        <Term term="Quality Function" /> · <Term term="Comparison Baseline" /> · <Term term="Normalisation" />
      </p>
    </Section>
  );
}

function Histogram({
  d,
  name,
  cuts,
}: {
  d: Extract<Descriptor, { type: "numeric" }>;
  name: string;
  cuts: number[];
}) {
  const bins = d.hist.map((c, i) => ({ x1: d.edges[i], x2: d.edges[i + 1], c }));
  return (
    <PlotFigure
      label={`Distribution of ${name} with cut points`}
      testId={`hist-${name}`}
      deps={[JSON.stringify(d), cuts.join(",")]}
      make={() => ({
        width: 340,
        height: 120,
        marginLeft: 36,
        marginBottom: 26,
        x: { label: name, nice: true },
        y: { label: "students", grid: true },
        marks: [
          Plot.rectY(bins, {
            x1: "x1",
            x2: "x2",
            y: "c",
            fill: "var(--accent)",
            fillOpacity: 0.35,
            inset: 0.5,
          }),
          Plot.ruleX(cuts, { stroke: "var(--bad)", strokeWidth: 1.5, strokeDasharray: "3 2" }),
          Plot.ruleY([0]),
        ],
      })}
    />
  );
}

function DescriptorCard({ name, d }: { name: string; d: Descriptor }) {
  const search = useDoc((s) => s.doc.search);
  const selected = !search.attributes || search.attributes.includes(name);
  const toggle = (v: boolean) =>
    edit(`search ${name}`, (doc) => {
      const names = Object.keys(descriptorsCache.current ?? {});
      const cur = doc.search.attributes ?? names;
      const next = v ? [...new Set([...cur, name])] : cur.filter((x) => x !== name);
      if (next.length === names.length && names.every((n) => next.includes(n))) delete doc.search.attributes;
      else doc.search.attributes = names.filter((n) => next.includes(n));
    });
  const b: Binning = search.binning[name] ?? search.default_binning;
  const setBin = (patch: Partial<Binning>) =>
    edit(`binning ${name}`, (doc) => {
      doc.search.binning[name] = { ...(doc.search.binning[name] ?? doc.search.default_binning), ...patch };
    });
  return (
    <li className="card p-3" data-testid={`descriptor-${name}`}>
      <div className="mb-1.5 flex items-center gap-2">
        <Checkbox label={`Search ${name}`} checked={selected} onChange={toggle}>
          <span className="font-semibold">{name}</span>
        </Checkbox>
        <span className="badge">{d.level}</span>
        <span className="badge">{d.type}</span>
        {d.role === "decoy" ? <span className="badge badge-warn">decoy</span> : null}
      </div>
      {d.type === "categorical" ? (
        <ul
          className="m-0 flex list-none flex-col gap-0.5 p-0 text-[12px]"
          aria-label={`Value shares of ${name}`}
        >
          {Object.entries(d.values).map(([v, p]) => (
            <li key={v} className="grid grid-cols-[90px_1fr_44px] items-center gap-2">
              <span className="truncate">{v}</span>
              <span className="h-2 rounded bg-panel-2">
                <span
                  className="block h-2 rounded bg-accent"
                  style={{ width: `${Math.round(p * 100)}%`, opacity: 0.55 }}
                />
              </span>
              <span className="num text-right text-muted">{fmt(p, 2)}</span>
            </li>
          ))}
        </ul>
      ) : (
        <>
          <div className="mb-1 grid grid-cols-3 gap-2">
            <NumberField
              label="Bins"
              value={b.n_bins}
              min={2}
              max={20}
              onCommit={(v) => setBin({ n_bins: Math.round(v ?? 5) })}
            />
            <SelectField
              label="Weighting"
              value={b.weighting}
              options={["students", "schools"]}
              onChange={(v) => setBin({ weighting: v })}
            />
            <div className="flex items-end pb-1">
              <Checkbox label="One-sided" checked={b.one_sided} onChange={(v) => setBin({ one_sided: v })} />
            </div>
          </div>
          <Histogram d={d} name={name} cuts={d.cuts} />
          <div className="text-[11.5px] text-muted" data-testid={`cuts-${name}`}>
            cut points: {d.cuts.map((c) => fmt(c, 2)).join(", ")}
          </div>
        </>
      )}
    </li>
  );
}

const descriptorsCache: { current: Record<string, Descriptor> | null } = { current: null };

export function SearchView() {
  const { world, model, search, seed } = useDoc((s) => s.doc);
  const { issues } = useTruth();
  const q = useQuery({
    queryKey: ["descriptors", JSON.stringify([world, model.latents, search.binning, search.default_binning])],
    queryFn: () => api.descriptors(world, model, search),
    enabled: issues.length === 0,
    placeholderData: keepPreviousData,
  });
  if (q.data) descriptorsCache.current = q.data;
  const attrNames = Object.keys(q.data ?? {});
  const cats = Object.entries(world.attributes);
  const [fAttr, fVal] = Object.entries(search.filter ?? {})[0] ?? ["", ""];
  return (
    <div className="grid h-full min-h-0" style={{ gridTemplateColumns: "minmax(0,1fr) 380px" }}>
      <section className="scroll-y min-w-0 p-4" aria-label="Search space">
        <h1 className="mt-0 mb-1 font-semibold text-[16px]">
          <Term term="Search Space" />
        </h1>
        <p className="mt-0 mb-3 text-[12.5px] text-muted">
          Tick the attributes EMM may use. Numeric attributes are cut into bins; the red lines are the cut
          points of one generated dataset.
        </p>
        {q.isError ? <p className="text-bad">{String((q.error as Error).message)}</p> : null}
        {issues.length ? <Empty>Fix the World's issues to see descriptor distributions.</Empty> : null}
        <ul className="m-0 grid list-none grid-cols-1 gap-3 p-0 xl:grid-cols-2" data-testid="descriptors">
          {Object.entries(q.data ?? {}).map(([name, d]) => (
            <DescriptorCard key={name} name={name} d={d} />
          ))}
        </ul>
        {q.isFetching && !q.data ? <Empty>Generating one dataset to show distributions…</Empty> : null}
        {!q.isFetching && q.data && !attrNames.length ? (
          <Empty>The World has no attributes to search.</Empty>
        ) : null}
      </section>
      <aside className="scroll-y h-full border-line border-l bg-panel" aria-label="Search settings">
        <Section title="Search">
          <div className="grid grid-cols-2 gap-2">
            <SelectField
              label="Depth (conditions per subgroup)"
              value={String(search.depth) as "1" | "2" | "3"}
              options={["1", "2", "3"]}
              onChange={(v) =>
                edit("depth", (d) => {
                  d.search.depth = Number(v);
                })
              }
            />
            <NumberField
              label="Seed"
              value={seed}
              min={0}
              onCommit={(v) =>
                edit("seed", (d) => {
                  d.seed = Math.round(v ?? 0);
                })
              }
            />
            <NumberField
              label="Min students (n)"
              value={search.min_n}
              min={1}
              onCommit={(v) =>
                edit("min n", (d) => {
                  d.search.min_n = Math.round(v ?? 100);
                })
              }
            />
            <NumberField
              label="Min complement n"
              value={search.min_n_complement}
              min={1}
              onCommit={(v) =>
                edit("min n complement", (d) => {
                  d.search.min_n_complement = Math.round(v ?? 100);
                })
              }
            />
            <NumberField
              label="Min schools"
              value={search.min_schools}
              min={0}
              onCommit={(v) =>
                edit("min schools", (d) => {
                  d.search.min_schools = Math.round(v ?? 0);
                })
              }
            />
          </div>
        </Section>
        <Section title="Filter (analyse only one subpopulation)">
          <div className="grid grid-cols-2 gap-2">
            <SelectField
              label="Filter attribute"
              value={fAttr}
              options={[{ value: "", label: "no filter" }, ...cats.map(([k]) => ({ value: k, label: k }))]}
              onChange={(v) =>
                edit("filter", (d) => {
                  if (!v) delete d.search.filter;
                  else d.search.filter = { [v]: Object.keys(d.world.attributes[v].values)[0] };
                })
              }
            />
            <SelectField
              label="Filter value"
              value={fVal}
              disabled={!fAttr}
              options={fAttr ? Object.keys(world.attributes[fAttr]?.values ?? {}) : [""]}
              onChange={(v) =>
                edit("filter value", (d) => {
                  d.search.filter = { [fAttr]: v };
                })
              }
            />
          </div>
        </Section>
        <Section title="Default binning (numeric attributes)">
          <div className="grid grid-cols-2 gap-2">
            <NumberField
              label="Bins"
              value={search.default_binning.n_bins}
              min={2}
              max={20}
              onCommit={(v) =>
                edit("default bins", (d) => {
                  d.search.default_binning.n_bins = Math.round(v ?? 5);
                })
              }
            />
            <SelectField
              label="Weighting"
              value={search.default_binning.weighting}
              options={["students", "schools"]}
              onChange={(v) =>
                edit("default weighting", (d) => {
                  d.search.default_binning.weighting = v;
                })
              }
            />
            <Checkbox
              label="One-sided cuts"
              checked={search.default_binning.one_sided}
              onChange={(v) =>
                edit("default one-sided", (d) => {
                  d.search.default_binning.one_sided = v;
                })
              }
            />
          </div>
          <button
            type="button"
            className="btn btn-sm mt-2"
            onClick={() =>
              edit("reset search", (d) => {
                d.search = structuredClone(DEFAULT_SEARCH);
              })
            }
          >
            Reset search space
          </button>
        </Section>
        <MethodPanel />
      </aside>
    </div>
  );
}
