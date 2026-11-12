import { useEffect, useRef, useState } from "react";
import type { Issue, World } from "../../api/types";
import { nodeEquation } from "../../components/Equation";
import { NodeEquationView, PrimitivesLegend } from "../../components/NodeEquationView";
import {
  Checkbox,
  DerivedValue,
  Empty,
  NumberField,
  Section,
  SelectField,
  Term,
  TextField,
} from "../../components/ui";
import { useDoc } from "../../state/doc";
import { useTruth } from "../../state/truth";
import { useUI } from "../../state/ui";
import {
  arrowKey,
  categoricalNames,
  connectionProblem,
  defaultValue,
  fmt,
  fmtTyped,
  nameTaken,
  normKey,
  parseArrowKey,
  parseNumber,
  renameAttrValue,
  syncRootRecipe,
  validName,
} from "../../state/worldOps";
import {
  addArrow,
  addShift,
  addTwist,
  deleteElement,
  editWorld,
  removeArrow,
  removeShift,
  removeTwist,
  renameElement,
  setArrowSlope,
} from "./actions";
import { issueFix, issuesFor, showIssue } from "./issues";

function useFocusField(): [string | null, () => void] {
  const f = useUI((s) => s.focusField);
  return [f, () => useUI.getState().set({ focusField: null })];
}

/** Focus a field after mount when "show on canvas"/creation asked for it. */
function useAutoFocus<T extends HTMLElement>(want: boolean) {
  const ref = useRef<T>(null);
  useEffect(() => {
    if (want && ref.current) {
      ref.current.focus();
      if (ref.current instanceof HTMLInputElement) ref.current.select();
      useUI.getState().set({ focusField: null });
    }
  }, [want]);
  return ref;
}

function nameProblem(w: World, current: string) {
  return (v: string) => {
    if (!validName(v)) return "use letters, digits and _ (start with a letter)";
    if (v !== current && nameTaken(w, v)) return `“${v}” is already used`;
    return null;
  };
}

function InlineIssues({ issues }: { issues: Issue[] }) {
  const w = useDoc((s) => s.doc.world);
  if (!issues.length) return null;
  return (
    <div className="mx-4 my-2 flex flex-col gap-1.5" data-testid="inline-issues">
      {issues.map((i, k) => {
        const fix = issueFix(i, w);
        return (
          <div
            key={k}
            role="alert"
            className="rounded-md border border-bad bg-bad-soft px-2 py-1.5 text-[12.5px] text-bad"
          >
            <div>
              <code className="text-[11.5px]">{i.element}</code>: {i.message}
            </div>
            {fix ? (
              <button type="button" className="btn btn-sm mt-1" onClick={fix.run}>
                {fix.label}
              </button>
            ) : null}
          </div>
        );
      })}
    </div>
  );
}

function fieldIssue(issues: Issue[], field: string): string | null {
  const hit = issues.find(
    (i) => i.element.endsWith(`.${field}`) || (field.startsWith("target") && i.element.endsWith(".target")),
  );
  return hit ? hit.message : null;
}

// ----------------------------------------------------------------------------- continuous node

function NodeInspector({ name }: { name: string }) {
  const w = useDoc((s) => s.doc.world);
  const { derived, issues } = useTruth();
  const [focus] = useFocusField();
  const n = w.nodes[name];
  const nameRef = useAutoFocus<HTMLInputElement>(focus === "name");
  const targetSdRef = useAutoFocus<HTMLInputElement>(focus === "target.sd" || focus === "target");
  const targetR2Ref = useAutoFocus<HTMLInputElement>(focus === "target.r2");
  const noiseRef = useAutoFocus<HTMLInputElement>(focus === "recipe.noise_sd" || focus === "recipe");
  if (!n) return <Empty>Node “{name}” no longer exists.</Empty>;
  const d = derived?.nodes[name];
  const mine = issuesFor(issues, "node", name, w);
  const set = (label: string, kind: "spread" | "other" | "structure" | "slope", fn: (w: World) => void) =>
    editWorld(`${label} ${name}`, kind, fn);
  const node = (w: World) => w.nodes[name];
  const eq = nodeEquation(w, name, derived);
  const outs = w.arrows.filter((a) => a.from === name);
  const ins = w.arrows.filter((a) => a.to === name);
  const shiftsIn = w.shifts.map((s, i) => ({ s, i })).filter(({ s }) => s.target === name);
  const isLatent = (n.kind ?? "latent") === "latent";
  return (
    <div data-testid="node-inspector">
      <Section title={isLatent ? <Term term="Latent" /> : <Term term="Attribute">Continuous attribute</Term>}>
        <div className="grid grid-cols-2 gap-2">
          <div className="col-span-2">
            <TextField
              label="Name"
              value={name}
              inputRef={nameRef}
              autoSelect
              validate={nameProblem(w, name)}
              onCommit={(v) => renameElement("node", name, v)}
            />
          </div>
          <SelectField
            label="Kind"
            value={n.kind ?? "latent"}
            options={[
              { value: "latent", label: "latent" },
              { value: "attribute", label: "continuous attribute" },
            ]}
            onChange={(v) =>
              set("kind", "other", (w) => {
                node(w).kind = v;
              })
            }
          />
          <SelectField
            label="Level"
            value={n.level ?? "student"}
            options={["student", "school"]}
            onChange={(v) =>
              set("level", "other", (w) => {
                node(w).level = v;
                if (v === "school") {
                  delete node(w).school_sd;
                  if (node(w).target) delete node(w).target!.icc;
                }
              })
            }
          />
          <SelectField
            label="Role"
            value={n.role ?? "normal"}
            options={[
              { value: "normal", label: "normal" },
              { value: "decoy", label: "decoy (affects nothing)" },
            ]}
            onChange={(v) =>
              set("role", "other", (w) => {
                node(w).role = v;
              })
            }
          />
          <div className="flex items-end pb-1">
            <Checkbox
              label="Observed"
              checked={n.observed !== false}
              onChange={(v) =>
                set("observed", "other", (w) => {
                  node(w).observed = v;
                })
              }
            />
          </div>
        </div>
      </Section>
      <InlineIssues issues={mine} />
      {n.root ? (
        <Section title={<Term term="Distribution">Root distribution</Term>}>
          <div className="grid grid-cols-2 gap-2">
            <NumberField
              label="Mean"
              value={n.root.mean ?? 0}
              onCommit={(v) =>
                set("mean", "slope", (w) => {
                  node(w).root!.mean = v ?? 0;
                })
              }
            />
            <NumberField
              label="SD"
              value={n.root.sd ?? 1}
              min={1e-9}
              onCommit={(v) =>
                set("sd", "spread", (w) => {
                  node(w).root!.sd = v ?? 1;
                })
              }
            />
          </div>
        </Section>
      ) : (
        <Section title={<Term term="Recipe" />}>
          <div className="grid grid-cols-2 gap-2">
            <NumberField
              label="Intercept"
              value={n.recipe?.intercept ?? 0}
              onCommit={(v) =>
                set("intercept", "slope", (w) => {
                  node(w).recipe = { ...node(w).recipe, intercept: v ?? 0 };
                })
              }
            />
            <NumberField
              label="Noise SD"
              value={n.recipe?.noise_sd}
              allowEmpty
              min={0}
              inputRef={noiseRef}
              placeholder={d?.noise_sd !== undefined ? `solved ${fmt(d.noise_sd)}` : "from target"}
              error={fieldIssue(mine, "noise_sd")}
              onCommit={(v) =>
                set("noise sd", "spread", (w) => {
                  const r = { ...node(w).recipe };
                  if (v === null) delete r.noise_sd;
                  else r.noise_sd = v;
                  node(w).recipe = r;
                })
              }
            />
          </div>
        </Section>
      )}
      <Section title="Targets (solved for noise and school SD)">
        <div className="grid grid-cols-3 gap-2">
          <NumberField
            label="Target SD"
            value={n.target?.sd}
            allowEmpty
            min={1e-9}
            inputRef={targetSdRef}
            disabled={!n.recipe}
            error={fieldIssue(mine, "target.sd")}
            onCommit={(v) =>
              set("target sd", "spread", (w) => {
                const t = { ...node(w).target };
                if (v === null) delete t.sd;
                else {
                  t.sd = v;
                  delete t.r2;
                  if (node(w).recipe) delete node(w).recipe!.noise_sd;
                }
                node(w).target = t;
                if (!Object.keys(t).length) delete node(w).target;
              })
            }
          />
          <NumberField
            label="Target R²"
            value={n.target?.r2}
            allowEmpty
            min={0}
            max={0.999}
            inputRef={targetR2Ref}
            disabled={!n.recipe}
            onCommit={(v) =>
              set("target r2", "spread", (w) => {
                const t = { ...node(w).target };
                if (v === null) delete t.r2;
                else {
                  t.r2 = v;
                  delete t.sd;
                  if (node(w).recipe) delete node(w).recipe!.noise_sd;
                }
                node(w).target = t;
                if (!Object.keys(t).length) delete node(w).target;
              })
            }
          />
          <NumberField
            label="Target ICC"
            value={n.target?.icc}
            allowEmpty
            min={0}
            max={0.999}
            disabled={(n.level ?? "student") === "school"}
            onCommit={(v) =>
              set("target icc", "spread", (w) => {
                const t = { ...node(w).target };
                if (v === null) delete t.icc;
                else t.icc = v;
                node(w).target = t;
                if (!Object.keys(t).length) delete node(w).target;
              })
            }
          />
        </div>
        {(n.level ?? "student") === "student" ? (
          <div className="mt-2 grid grid-cols-2 gap-2">
            <NumberField
              label="School random-intercept SD"
              value={n.school_sd ?? 0}
              min={0}
              onCommit={(v) =>
                set("school sd", "spread", (w) => {
                  node(w).school_sd = v ?? 0;
                })
              }
            />
          </div>
        ) : null}
      </Section>
      <Section title={<Term term="Arrow">Arrows</Term>}>
        <ArrowRows arrows={[...ins, ...outs]} />
        <AddArrowRow from={name} />
      </Section>
      {isLatent ? (
        <Section title={<Term term="Level Shift">Level shifts into {name}</Term>}>
          {shiftsIn.length ? (
            <ShiftRows rows={shiftsIn} />
          ) : (
            <Empty>No attribute moves this node's average.</Empty>
          )}
          <AddShiftRow target={name} />
        </Section>
      ) : null}
      <Section title={<Term term="Derived">Derived</Term>}>
        <div className="grid grid-cols-4 gap-x-2 gap-y-1 text-[12px]">
          <span className="text-muted">mean</span>
          <span className="text-muted">SD</span>
          <span className="text-muted">R²</span>
          <span className="text-muted">ICC</span>
          <DerivedValue
            value={d?.mean}
            label={`Mean of ${name} (pooled over all cells)`}
            rippleKey={`node:${name}:mean`}
            testId={`derived-mean-${name}`}
          />
          <DerivedValue
            value={d?.sd}
            label={`SD of ${name} (pooled)`}
            rippleKey={`node:${name}:sd`}
            testId={`derived-sd-${name}`}
          />
          <DerivedValue
            value={d?.r2}
            label={`R² of ${name}: share explained by parents and shifts`}
            rippleKey={`node:${name}:r2`}
          />
          <DerivedValue
            value={d?.icc}
            label={`ICC of ${name}: share of variance between schools`}
            rippleKey={`node:${name}:icc`}
          />
        </div>
        {d?.noise_sd !== undefined && n.recipe ? (
          <div className="mt-1.5 text-[12px] text-muted">
            noise SD used{" "}
            <DerivedValue value={d.noise_sd} label={`Noise SD of ${name} (typed or solved from target)`} />
            {d.school_sd ? (
              <>
                {" "}
                · school SD{" "}
                <DerivedValue value={d.school_sd} label={`School random-intercept SD of ${name}`} />
              </>
            ) : null}
          </div>
        ) : null}
        {d && Object.keys(d.groups).length ? (
          <table className="data mt-2" aria-label={`Per-group means and SDs of ${name}`}>
            <thead>
              <tr>
                <th>group</th>
                <th className="r">mean</th>
                <th className="r">SD</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(d.groups).map(([g, v]) => (
                <tr key={g}>
                  <td>{g}</td>
                  <td className="r">
                    <DerivedValue value={v.mean} label={`Mean of ${name} in ${g}`} />
                  </td>
                  <td className="r">
                    <DerivedValue value={v.sd} label={`SD of ${name} in ${g}`} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : null}
      </Section>
      {eq ? (
        <Section title="Equation">
          <NodeEquationView name={name} world={w} derived={derived} testId={`equation-${name}`} />
        </Section>
      ) : null}
      <div className="px-4 py-3">
        <button type="button" className="btn btn-danger" onClick={() => deleteElement("node", name)}>
          Delete {name}
        </button>
      </div>
    </div>
  );
}

function ArrowRows({ arrows }: { arrows: { from: string; to: string; slope?: number; std?: number }[] }) {
  const select = useUI((s) => s.select);
  if (!arrows.length) return <Empty>No arrows yet.</Empty>;
  return (
    <ul className="m-0 mb-2 flex list-none flex-col gap-1 p-0">
      {arrows.map((a) => {
        const key = arrowKey(a);
        return (
          <li key={key} className="flex items-center gap-2">
            <button
              type="button"
              className="link min-w-0 flex-1 truncate text-left"
              onClick={() => select({ type: "arrow", id: key })}
            >
              {a.from} → {a.to}
            </button>
            <div className="w-20">
              <NumberField
                label={`Slope ${key}`}
                hideLabel
                value={a.slope ?? a.std}
                onCommit={(v) => v !== null && setArrowSlope(key, v, a.slope !== undefined ? "slope" : "std")}
              />
            </div>
            <button
              type="button"
              className="btn btn-ghost btn-sm"
              aria-label={`Remove arrow ${key}`}
              onClick={() => removeArrow(key)}
            >
              ×
            </button>
          </li>
        );
      })}
    </ul>
  );
}

/** "Add arrow from the selected node": pick a child, type the slope, Enter. */
function AddArrowRow({ from }: { from: string }) {
  const w = useDoc((s) => s.doc.world);
  const [to, setTo] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const selRef = useRef<HTMLSelectElement>(null);
  const slopeRef = useRef<HTMLInputElement>(null);
  const [slope, setSlope] = useState("");
  const candidates = Object.keys(w.nodes).filter((n) => n !== from);
  const add = () => {
    if (!to) return setErr("choose a child node");
    const v = slope.trim() === "" ? 0.3 : parseNumber(slope);
    if (v === null) return setErr("slope is not a number");
    const problem = connectionProblem(w, from, to);
    if (problem) return setErr(problem);
    addArrow(from, to, v);
    setErr(null);
    setTo("");
    setSlope("");
    selRef.current?.focus();
  };
  return (
    <div className="flex flex-col gap-1" data-testid="add-arrow">
      <div className="flex items-end gap-1.5">
        <div className="min-w-0 flex-1">
          <SelectField
            label={`Add arrow from ${from} to`}
            value={to}
            selectRef={selRef}
            options={[
              { value: "", label: "choose child…" },
              ...candidates.map((c) => {
                const p = connectionProblem(w, from, c);
                return { value: c, label: p ? `${c} (${p})` : c };
              }),
            ]}
            onChange={(v) => {
              setTo(v);
              setErr(v ? connectionProblem(w, from, v) : null);
              if (v) slopeRef.current?.focus();
            }}
          />
        </div>
        <div className="w-20">
          <label className="label" htmlFor={`add-slope-${from}`}>
            slope
          </label>
          <input
            id={`add-slope-${from}`}
            ref={slopeRef}
            className="field num-field"
            inputMode="decimal"
            placeholder="0.3"
            aria-label={`Slope of new arrow from ${from}`}
            value={slope}
            onChange={(e) => setSlope(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                add();
              }
            }}
          />
        </div>
        <button type="button" className="btn" onClick={add}>
          Add
        </button>
      </div>
      {err ? (
        <div role="alert" className="text-[11.5px] text-bad">
          {err}
        </div>
      ) : null}
    </div>
  );
}

function ShiftRows({ rows }: { rows: { s: World["shifts"][number]; i: number }[] }) {
  const select = useUI((s) => s.select);
  return (
    <ul className="m-0 mb-2 flex list-none flex-col gap-1 p-0">
      {rows.map(({ s, i }) => {
        const cond = Object.entries(s.when)
          .map(([k, v]) => `${k}=${v}`)
          .join(" & ");
        return (
          <li key={i} className="flex items-center gap-2">
            <button
              type="button"
              className="link min-w-0 flex-1 truncate text-left"
              onClick={() => select({ type: "shift", index: i })}
            >
              {cond} → {s.target}
            </button>
            <div className="w-20">
              <NumberField
                label={`Shift ${cond} on ${s.target}`}
                hideLabel
                value={s.delta}
                onCommit={(v) =>
                  v !== null &&
                  editWorld("shift delta", "slope", (w) => {
                    w.shifts[i].delta = v;
                  })
                }
              />
            </div>
            <button
              type="button"
              className="btn btn-ghost btn-sm"
              aria-label={`Remove level shift ${cond} on ${s.target}`}
              onClick={() => removeShift(i)}
            >
              ×
            </button>
          </li>
        );
      })}
    </ul>
  );
}

/** Add a level shift. Either the attribute or the target is fixed by the context. */
function AddShiftRow({ attr: fixedAttr, target: fixedTarget }: { attr?: string; target?: string }) {
  const w = useDoc((s) => s.doc.world);
  const attrs = categoricalNames(w);
  const [attr, setAttr] = useState(fixedAttr ?? attrs[0] ?? "");
  const vals = Object.keys(w.attributes[attr]?.values ?? {});
  const [value, setValue] = useState(defaultValue(w.attributes[attr]?.values ?? {}));
  const [target, setTarget] = useState(fixedTarget ?? "");
  const [delta, setDelta] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const deltaRef = useRef<HTMLInputElement>(null);
  const targetRef = useRef<HTMLSelectElement>(null);
  useEffect(() => {
    if (!vals.includes(value)) setValue(defaultValue(w.attributes[attr]?.values ?? {}));
  }, [attr, vals, value, w.attributes]);
  if (!attrs.length) return <Empty>Add a categorical attribute to create level shifts.</Empty>;
  const targets = Object.keys(w.nodes);
  const add = () => {
    if (!attr) return setErr("choose an attribute");
    if (!target) return setErr("choose the node to shift");
    const v = delta.trim() === "" ? 0.5 : parseNumber(delta);
    if (v === null) return setErr("delta is not a number");
    const problem = connectionProblem(w, attr, target);
    if (problem) return setErr(problem);
    addShift(attr, value, target, v);
    setErr(null);
    setDelta("");
    if (!fixedTarget) {
      setTarget("");
      targetRef.current?.focus();
    }
  };
  return (
    <div className="flex flex-col gap-1" data-testid="add-shift">
      <div className="grid grid-cols-[1fr_1fr_1fr_64px_auto] items-end gap-1.5">
        {fixedAttr ? (
          <div className="pb-1 text-[12px] text-muted">{fixedAttr} =</div>
        ) : (
          <SelectField label="Attribute" value={attr} options={attrs} onChange={(v) => setAttr(v)} />
        )}
        <SelectField label="Value" value={value} options={vals} onChange={(v) => setValue(v)} />
        {fixedTarget ? (
          <div className="pb-1 text-[12px] text-muted">→ {fixedTarget}</div>
        ) : (
          <SelectField
            label="Shift node"
            value={target}
            selectRef={targetRef}
            options={[{ value: "", label: "choose…" }, ...targets.map((t) => ({ value: t, label: t }))]}
            onChange={(v) => {
              setTarget(v);
              if (v) deltaRef.current?.focus();
            }}
          />
        )}
        <div>
          <label className="label" htmlFor={`add-delta-${fixedAttr ?? fixedTarget}`}>
            delta
          </label>
          <input
            id={`add-delta-${fixedAttr ?? fixedTarget}`}
            ref={deltaRef}
            className="field num-field"
            inputMode="decimal"
            placeholder="0.5"
            aria-label="Delta of new level shift"
            value={delta}
            onChange={(e) => setDelta(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                add();
              }
            }}
          />
        </div>
        <button type="button" className="btn" onClick={add}>
          Add
        </button>
      </div>
      {err ? (
        <div role="alert" className="text-[11.5px] text-bad">
          {err}
        </div>
      ) : null}
    </div>
  );
}

// ----------------------------------------------------------------------------- categorical attribute

function AttributeInspector({ name }: { name: string }) {
  const w = useDoc((s) => s.doc.world);
  const { issues } = useTruth();
  const [focus] = useFocusField();
  const nameRef = useAutoFocus<HTMLInputElement>(focus === "name");
  const a = w.attributes[name];
  const [newValue, setNewValue] = useState("");
  if (!a) return <Empty>Attribute “{name}” no longer exists.</Empty>;
  const mine = issuesFor(issues, "attribute", name, w);
  const total = Object.values(a.values).reduce((x, y) => x + y, 0);
  const shifts = w.shifts.map((s, i) => ({ s, i })).filter(({ s }) => name in s.when);
  const twists = w.twists.map((t, i) => ({ t, i })).filter(({ t }) => name in t.when);
  const others = categoricalNames(w).filter((x) => x !== name);
  const at = (w: World) => w.attributes[name];
  return (
    <div data-testid="attribute-inspector">
      <Section title={<Term term="Attribute">Categorical attribute</Term>}>
        <div className="grid grid-cols-2 gap-2">
          <div className="col-span-2">
            <TextField
              label="Name"
              value={name}
              inputRef={nameRef}
              autoSelect
              validate={nameProblem(w, name)}
              onCommit={(v) => renameElement("attribute", name, v)}
            />
          </div>
          <SelectField
            label="Level"
            value={a.level ?? "student"}
            options={["student", "school"]}
            onChange={(v) =>
              editWorld("level", "other", (w) => {
                at(w).level = v;
              })
            }
          />
          <SelectField
            label="Role"
            value={a.role ?? "normal"}
            options={[
              { value: "normal", label: "normal" },
              { value: "decoy", label: "decoy (affects nothing)" },
            ]}
            onChange={(v) =>
              editWorld("role", "other", (w) => {
                at(w).role = v;
              })
            }
          />
          <div className="col-span-2">
            <Checkbox
              label="Ordinal (values are ordered)"
              checked={!!a.ordinal}
              onChange={(v) =>
                editWorld("ordinal", "other", (w) => {
                  at(w).ordinal = v;
                })
              }
            />
          </div>
        </div>
      </Section>
      <InlineIssues issues={mine} />
      <Section
        title={<Term term="Distribution">Values and proportions</Term>}
        actions={
          <span className={`num text-[12px] ${Math.abs(total - 1) > 1e-6 ? "text-bad" : "text-muted"}`}>
            Σ {fmtTyped(total)}
          </span>
        }
      >
        <ul className="m-0 flex list-none flex-col gap-1 p-0">
          {Object.entries(a.values).map(([v, p], i) => (
            <li key={`${i}`} className="grid grid-cols-[1fr_76px_auto] items-end gap-1.5">
              <TextField
                label={`Value ${i + 1} name`}
                hideLabel
                value={v}
                validate={(nv) => (!nv ? "empty" : nv !== v && nv in a.values ? "duplicate value" : null)}
                onCommit={(nv) =>
                  editWorld("rename value", "structure", (w) => renameAttrValue(w, name, v, nv))
                }
              />
              <NumberField
                label={`Proportion of ${v}`}
                hideLabel
                value={p}
                min={0}
                onCommit={(np) =>
                  editWorld("proportion", "other", (w) => {
                    at(w).values[v] = np ?? 0;
                  })
                }
              />
              <button
                type="button"
                className="btn btn-ghost btn-sm"
                aria-label={`Remove value ${v}`}
                disabled={Object.keys(a.values).length <= 2}
                onClick={() =>
                  editWorld("remove value", "structure", (w) => {
                    delete at(w).values[v];
                    w.shifts = w.shifts.filter((s) => s.when[name] !== v);
                    w.twists = w.twists.filter((t) => t.when[name] !== v);
                    syncRootRecipe(w, Object.keys(w.nodes));
                  })
                }
              >
                ×
              </button>
            </li>
          ))}
        </ul>
        <div className="mt-1.5 flex items-end gap-1.5">
          <div className="flex-1">
            <label className="sr-only" htmlFor={`new-value-${name}`}>
              New value name
            </label>
            <input
              id={`new-value-${name}`}
              className="field"
              placeholder="new value…"
              value={newValue}
              onChange={(e) => setNewValue(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && newValue.trim() && !(newValue.trim() in a.values)) {
                  const v = newValue.trim();
                  editWorld("add value", "structure", (w) => {
                    at(w).values[v] = 0;
                  });
                  setNewValue("");
                }
              }}
            />
          </div>
          {Math.abs(total - 1) > 1e-6 ? (
            <button
              type="button"
              className="btn btn-sm"
              onClick={() =>
                editWorld("normalize", "other", (w) => {
                  const s = total || 1;
                  for (const k of Object.keys(at(w).values))
                    at(w).values[k] = Math.round((at(w).values[k] / s) * 1e6) / 1e6;
                })
              }
            >
              Normalize
            </button>
          ) : null}
        </div>
        <div className="mt-2">
          <SelectField
            label="Depends on (conditional proportions)"
            value={a.depends_on ?? ""}
            options={[{ value: "", label: "independent" }, ...others.map((o) => ({ value: o, label: o }))]}
            onChange={(v) =>
              editWorld("depends on", "other", (w) => {
                if (!v) {
                  delete at(w).depends_on;
                  delete at(w).table;
                } else {
                  at(w).depends_on = v;
                  const rows: Record<string, Record<string, number>> = {};
                  for (const pv of Object.keys(w.attributes[v].values)) rows[pv] = { ...at(w).values };
                  at(w).table = rows;
                }
              })
            }
          />
        </div>
        {a.depends_on && a.table ? (
          <table className="data mt-2" aria-label={`Proportions of ${name} given ${a.depends_on}`}>
            <thead>
              <tr>
                <th>{a.depends_on}</th>
                {Object.keys(a.values).map((v) => (
                  <th key={v} className="r">
                    {v}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Object.entries(a.table).map(([pv, row]) => (
                <tr key={pv}>
                  <td>{pv}</td>
                  {Object.keys(a.values).map((v) => (
                    <td key={v}>
                      <NumberField
                        label={`P(${name}=${v} | ${a.depends_on}=${pv})`}
                        hideLabel
                        value={row[v]}
                        min={0}
                        onCommit={(np) =>
                          editWorld("table", "other", (w) => {
                            at(w).table![pv][v] = np ?? 0;
                          })
                        }
                      />
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        ) : null}
      </Section>
      <Section title={<Term term="Level Shift">Level shifts from {name}</Term>}>
        {shifts.length ? (
          <ShiftRows rows={shifts} />
        ) : (
          <Empty>None yet: a level shift moves a latent's average.</Empty>
        )}
        <AddShiftRow attr={name} />
      </Section>
      <Section title={<Term term="Twist">Twists conditioned on {name}</Term>}>
        {twists.length ? (
          <ul className="m-0 flex list-none flex-col gap-1 p-0">
            {twists.map(({ t, i }) => (
              <li key={i}>
                <button
                  type="button"
                  className="link"
                  onClick={() => useUI.getState().select({ type: "twist", index: i })}
                >
                  {t.arrow}: Δ {fmtTyped(t.delta)} when{" "}
                  {Object.entries(t.when)
                    .map(([k, v]) => `${k}=${v}`)
                    .join(" & ")}
                </button>
              </li>
            ))}
          </ul>
        ) : (
          <Empty>Select an arrow to plant a twist on it.</Empty>
        )}
      </Section>
      <div className="px-4 py-3">
        <button type="button" className="btn btn-danger" onClick={() => deleteElement("attribute", name)}>
          Delete {name}
        </button>
      </div>
    </div>
  );
}

// ----------------------------------------------------------------------------- arrow (+ twists)

function ArrowInspector({ arrowId, twistIndex }: { arrowId: string; twistIndex?: number }) {
  const w = useDoc((s) => s.doc.world);
  const { derived, issues } = useTruth();
  const [focus] = useFocusField();
  const slopeRef = useAutoFocus<HTMLInputElement>(focus === "slope");
  const key = normKey(arrowId);
  const a = w.arrows.find((x) => arrowKey(x) === key);
  const [from, to] = parseArrowKey(key);
  const twists = w.twists.map((t, i) => ({ t, i })).filter(({ t }) => normKey(t.arrow) === key);
  const mine = [
    ...issuesFor(issues, "arrow", key, w),
    ...twists.flatMap(({ i }) => issuesFor(issues, "twist", i, w)),
  ];
  if (!a) {
    return (
      <div className="p-4">
        <Empty>
          There is no arrow {key} in the World.
          {twistIndex !== undefined ? " A twist refers to it: twists can only sit on existing arrows." : ""}
        </Empty>
        <InlineIssues issues={twistIndex !== undefined ? issuesFor(issues, "twist", twistIndex, w) : []} />
        {twistIndex !== undefined ? (
          <button type="button" className="btn btn-danger mt-2" onClick={() => removeTwist(twistIndex)}>
            Remove twist
          </button>
        ) : null}
      </div>
    );
  }
  const d = derived?.arrows[key];
  const mode = a.slope !== undefined ? "slope" : "std";
  const eq = nodeEquation(w, to, derived);
  return (
    <div data-testid="arrow-inspector">
      <Section title={<Term term="Arrow" />}>
        <div className="mb-2 text-[14px] font-semibold">
          {from} → {to}
        </div>
        <div className="grid grid-cols-2 gap-2">
          <SelectField
            label="Specified as"
            value={mode}
            options={[
              { value: "slope", label: "raw slope" },
              { value: "std", label: "standardized (std)" },
            ]}
            onChange={(v) =>
              setArrowSlope(
                key,
                (v === "slope" ? (d?.baseline_slope ?? a.std) : (d?.std ?? a.slope)) ?? 0.3,
                v,
              )
            }
          />
          <NumberField
            label={mode === "slope" ? "Slope" : "Standardized coefficient"}
            value={mode === "slope" ? a.slope : a.std}
            inputRef={slopeRef}
            onCommit={(v) => v !== null && setArrowSlope(key, v, mode)}
          />
        </div>
      </Section>
      <InlineIssues issues={mine} />
      <Section title={<Term term="Twist">Twists on this arrow</Term>}>
        {twists.length ? (
          <ul className="m-0 mb-2 flex list-none flex-col gap-1.5 p-0">
            {twists.map(({ i }) => (
              <TwistRow key={i} index={i} highlighted={i === twistIndex} />
            ))}
          </ul>
        ) : (
          <Empty>No twist: the slope is the same for everyone.</Empty>
        )}
        <AddTwistRow arrow={key} />
      </Section>
      <Section title={<Term term="Derived">Derived effects</Term>}>
        <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-[12.5px]" data-testid="arrow-derived">
          <span className="text-muted">baseline slope</span>
          <DerivedValue
            value={d?.baseline_slope}
            label={`Baseline slope of ${key} (raw units)`}
            rippleKey={`arrow:${key}:raw`}
          />
          <span className="text-muted">average direct effect</span>
          <DerivedValue
            value={d?.avg_slope}
            label={`Direct effect ${key}, averaged over cells (T1/T2, raw units)`}
            rippleKey={`arrow:${key}:raw`}
          />
          <span className="text-muted">
            <Term term="True Quantities">total effect</Term> (avg)
          </span>
          <DerivedValue
            value={d?.avg_total}
            label={`Total effect ${key}, averaged over cells (raw units)`}
            rippleKey={`arrow:${key}:raw`}
            testId={`total-${key}`}
          />
          <span className="text-muted">standardized β (pooled)</span>
          <DerivedValue
            value={d?.std}
            label={`Standardized coefficient ${key} (pooled)`}
            rippleKey={`arrow:${key}:std`}
            testId={`std-${key}`}
          />
        </div>
        {d && Object.keys(d.groups).length ? (
          <table className="data mt-2" aria-label={`Per-group effects of ${key}`} data-testid="arrow-groups">
            <thead>
              <tr>
                <th>group</th>
                <th className="r">slope</th>
                <th className="r">total</th>
                <th className="r">β</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(d.groups).map(([g, v]) => (
                <tr key={g} data-group={g}>
                  <td>{g}</td>
                  <td className="r">
                    <DerivedValue
                      value={v.slope}
                      label={`Direct slope ${key} in ${g}`}
                      rippleKey={`arrow:${key}:raw`}
                    />
                  </td>
                  <td className="r">
                    <DerivedValue
                      value={v.total}
                      label={`Total effect ${key} in ${g}`}
                      rippleKey={`arrow:${key}:raw`}
                      testId={`total-${key}-${g}`}
                    />
                  </td>
                  <td className="r">
                    <DerivedValue
                      value={v.std}
                      label={`Standardized coefficient ${key} in ${g}`}
                      rippleKey={`arrow:${key}:std`}
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : null}
      </Section>
      {eq ? (
        <Section title={`Equation of ${to}`}>
          <NodeEquationView name={to} world={w} derived={derived} />
        </Section>
      ) : null}
      <div className="px-4 py-3">
        <button type="button" className="btn btn-danger" onClick={() => removeArrow(key)}>
          Delete arrow
        </button>
      </div>
    </div>
  );
}

function TwistRow({ index, highlighted }: { index: number; highlighted: boolean }) {
  const w = useDoc((s) => s.doc.world);
  const t = w.twists[index];
  const ref = useRef<HTMLLIElement>(null);
  useEffect(() => {
    if (highlighted) ref.current?.scrollIntoView?.({ block: "nearest" });
  }, [highlighted]);
  if (!t) return null;
  const [attr, value] = Object.entries(t.when)[0] ?? ["", ""];
  const second = Object.entries(t.when)[1] as [string, string] | undefined;
  const attrs = categoricalNames(w);
  const setWhen = (first: [string, string], and: [string, string] | undefined) =>
    editWorld("twist condition", "slope", (w) => {
      w.twists[index].when =
        and && and[0] !== first[0] ? { [first[0]]: first[1], [and[0]]: and[1] } : { [first[0]]: first[1] };
    });
  return (
    <li
      ref={ref}
      className={`rounded-md border p-2 ${highlighted ? "border-twist bg-twist-soft" : "border-line"}`}
      data-testid={`twist-row-${index}`}
    >
      <div className="grid grid-cols-[1fr_1fr_70px_auto] items-end gap-1.5">
        <SelectField
          label="Condition attribute"
          value={attr}
          options={attrs.includes(attr) ? attrs : [attr, ...attrs]}
          onChange={(v) => setWhen([v, defaultValue(w.attributes[v]?.values ?? {})], second)}
        />
        <SelectField
          label="Value"
          value={value}
          options={Object.keys(w.attributes[attr]?.values ?? { [value]: 0 })}
          onChange={(v) => setWhen([attr, v], second)}
        />
        <NumberField
          label="Δ slope"
          value={t.delta}
          onCommit={(v) =>
            v !== null &&
            editWorld("twist delta", "slope", (w) => {
              w.twists[index].delta = v;
            })
          }
        />
        <button
          type="button"
          className="btn btn-ghost btn-sm"
          aria-label={`Remove twist ${index + 1}`}
          onClick={() => removeTwist(index)}
        >
          ×
        </button>
      </div>
      <div className="mt-1.5 grid grid-cols-[1fr_1fr] items-end gap-1.5" data-testid={`twist-and-${index}`}>
        <SelectField
          label="and (optional)"
          value={second?.[0] ?? ""}
          options={[
            { value: "", label: "— only the first condition" },
            ...attrs.filter((a) => a !== attr).map((a) => ({ value: a, label: a })),
          ]}
          onChange={(v) =>
            setWhen([attr, value], v ? [v, defaultValue(w.attributes[v]?.values ?? {})] : undefined)
          }
        />
        {second ? (
          <SelectField
            label="and value"
            value={second[1]}
            options={Object.keys(w.attributes[second[0]]?.values ?? { [second[1]]: 0 })}
            onChange={(v) => setWhen([attr, value], [second[0], v])}
          />
        ) : (
          <span />
        )}
      </div>
      <div className="mt-1.5">
        <TextField
          label="Label"
          value={t.label ?? ""}
          placeholder="e.g. feedback helps girls more"
          onCommit={(v) =>
            editWorld("twist label", "other", (w) => {
              if (v) w.twists[index].label = v;
              else delete w.twists[index].label;
            })
          }
        />
      </div>
    </li>
  );
}

/** Twist UI only offers existing arrows (it lives on one) and categorical attributes (U4). */
function AddTwistRow({ arrow }: { arrow: string }) {
  const w = useDoc((s) => s.doc.world);
  const attrs = categoricalNames(w);
  const [attr, setAttr] = useState(attrs[0] ?? "");
  const vals = Object.keys(w.attributes[attr]?.values ?? {});
  const [value, setValue] = useState(defaultValue(w.attributes[attr]?.values ?? {}));
  const [delta, setDelta] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const [andAttr, setAndAttr] = useState("");
  const andVals = Object.keys(w.attributes[andAttr]?.values ?? {});
  const [andValue, setAndValue] = useState("");
  useEffect(() => {
    if (!attrs.includes(attr) && attrs[0]) setAttr(attrs[0]);
  }, [attrs, attr]);
  useEffect(() => {
    if (andAttr && (andAttr === attr || !attrs.includes(andAttr))) setAndAttr("");
    if (andAttr && !andVals.includes(andValue))
      setAndValue(defaultValue(w.attributes[andAttr]?.values ?? {}));
  }, [andAttr, attr, attrs, andVals, andValue, w.attributes]);
  useEffect(() => {
    if (!vals.includes(value)) setValue(defaultValue(w.attributes[attr]?.values ?? {}));
  }, [attr, vals, value, w.attributes]);
  if (!attrs.length) return <Empty>Twists are conditioned on a categorical attribute: add one first.</Empty>;
  const add = () => {
    const v = parseNumber(delta);
    if (v === null) return setErr("type the slope change Δ");
    if (addTwist(arrow, attr, value, v, andAttr ? [andAttr, andValue] : undefined)) {
      setDelta("");
      setErr(null);
    }
  };
  return (
    <div className="flex flex-col gap-1" data-testid="add-twist">
      <div className="grid grid-cols-[1fr_1fr_70px_auto] items-end gap-1.5">
        <SelectField
          label="Twist attribute"
          value={attr}
          options={attrs}
          onChange={setAttr}
          testId="twist-attr"
        />
        <SelectField label="Twist value" value={value} options={vals} onChange={setValue} />
        <div>
          <label className="label" htmlFor={`twist-delta-${arrow}`}>
            Δ slope
          </label>
          <input
            id={`twist-delta-${arrow}`}
            className="field num-field"
            inputMode="decimal"
            placeholder="−0.2"
            aria-label="Delta of new twist"
            value={delta}
            onChange={(e) => setDelta(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                add();
              }
            }}
          />
        </div>
        <button type="button" className="btn" onClick={add}>
          Add twist
        </button>
      </div>
      {attrs.length > 1 ? (
        <div className="grid grid-cols-[1fr_1fr] items-end gap-1.5">
          <SelectField
            label="and (optional)"
            value={andAttr}
            options={[
              { value: "", label: "— one condition" },
              ...attrs.filter((a) => a !== attr).map((a) => ({ value: a, label: a })),
            ]}
            onChange={setAndAttr}
          />
          {andAttr ? (
            <SelectField label="and value" value={andValue} options={andVals} onChange={setAndValue} />
          ) : (
            <span />
          )}
        </div>
      ) : null}
      {err ? (
        <div role="alert" className="text-[11.5px] text-bad">
          {err}
        </div>
      ) : null}
    </div>
  );
}

// ----------------------------------------------------------------------------- level shift

function ShiftInspector({ index }: { index: number }) {
  const w = useDoc((s) => s.doc.world);
  const { issues } = useTruth();
  const [focus] = useFocusField();
  const deltaRef = useAutoFocus<HTMLInputElement>(focus === "delta");
  const s = w.shifts[index];
  if (!s) return <Empty>This level shift no longer exists.</Empty>;
  const [attr, value] = Object.entries(s.when)[0] ?? ["", ""];
  const attrs = categoricalNames(w);
  return (
    <div data-testid="shift-inspector">
      <Section title={<Term term="Level Shift" />}>
        <div className="grid grid-cols-2 gap-2">
          <SelectField
            label="Attribute"
            value={attr}
            options={attrs.includes(attr) ? attrs : [attr, ...attrs]}
            onChange={(v) =>
              editWorld("shift attribute", "structure", (w) => {
                w.shifts[index].when = { [v]: defaultValue(w.attributes[v]?.values ?? {}) };
              })
            }
          />
          <SelectField
            label="Value"
            value={value}
            options={Object.keys(w.attributes[attr]?.values ?? { [value]: 0 })}
            onChange={(v) =>
              editWorld("shift value", "structure", (w) => {
                w.shifts[index].when = { [attr]: v };
              })
            }
          />
          <SelectField
            label="Shifted node"
            value={s.target}
            options={
              Object.keys(w.nodes).includes(s.target)
                ? Object.keys(w.nodes)
                : [s.target, ...Object.keys(w.nodes)]
            }
            onChange={(v) =>
              editWorld("shift target", "structure", (w) => {
                const old = w.shifts[index].target;
                w.shifts[index].target = v;
                syncRootRecipe(w, [old, v]);
              })
            }
          />
          <NumberField
            label="Delta (added to the intercept)"
            value={s.delta}
            inputRef={deltaRef}
            onCommit={(v) =>
              v !== null &&
              editWorld("shift delta", "slope", (w) => {
                w.shifts[index].delta = v;
              })
            }
          />
        </div>
      </Section>
      <InlineIssues issues={issuesFor(issues, "shift", index, w)} />
      <div className="px-4 py-3">
        <button type="button" className="btn btn-danger" onClick={() => removeShift(index)}>
          Delete level shift
        </button>
      </div>
    </div>
  );
}

// ----------------------------------------------------------------------------- world

function WorldInspector() {
  const w = useDoc((s) => s.doc.world);
  const { derived, unfolded } = useTruth();
  const unfoldOn = useUI((s) => s.equationView === "primitives");
  const sps = w.schools?.students_per_school ?? {};
  const setSchools = (fn: (s: NonNullable<World["schools"]>) => void, kind: "other" | "spread" = "other") =>
    editWorld("schools", kind, (w) => {
      w.schools = w.schools ?? {};
      fn(w.schools);
    });
  const select = useUI((s) => s.select);
  return (
    <div data-testid="world-inspector">
      <Section title={<Term term="World" />}>
        <div className="grid grid-cols-2 gap-2">
          <div className="col-span-2">
            <TextField
              label="World name"
              value={w.name}
              onCommit={(v) =>
                editWorld("rename world", "other", (w) => {
                  w.name = v || "untitled";
                })
              }
            />
          </div>
          <div className="col-span-2 flex flex-col gap-0.5">
            <label className="label" htmlFor="world-description">
              Description
            </label>
            <textarea
              id="world-description"
              className="field"
              rows={2}
              defaultValue={w.description ?? ""}
              key={w.description}
              onBlur={(e) => {
                const v = e.target.value;
                if (v !== (w.description ?? ""))
                  editWorld("description", "other", (w) => {
                    w.description = v;
                  });
              }}
            />
          </div>
          <NumberField
            label="Schools"
            value={w.schools?.n_schools ?? 200}
            min={2}
            onCommit={(v) =>
              setSchools((s) => {
                s.n_schools = Math.round(v ?? 200);
              })
            }
          />
          <NumberField
            label="Students per school"
            value={sps.n ?? 35}
            min={2}
            onCommit={(v) =>
              setSchools((s) => {
                s.students_per_school = { ...s.students_per_school, n: Math.round(v ?? 35) };
              })
            }
          />
          <SelectField
            label="School sizes"
            value={sps.dist ?? "fixed"}
            options={["fixed", "uniform", "poisson"]}
            onChange={(v) =>
              setSchools((s) => {
                s.students_per_school = { ...s.students_per_school, dist: v };
              })
            }
          />
          <SelectField
            label="School allocation"
            value={w.schools?.allocation ?? "stratified"}
            options={["stratified", "random"]}
            onChange={(v) =>
              setSchools((s) => {
                s.allocation = v;
              })
            }
          />
          <div className="col-span-2">
            <SelectField
              label="Generation"
              value={w.generation ?? "naive"}
              options={["naive", "internally_standardized"]}
              onChange={(v) =>
                editWorld("generation", "other", (w) => {
                  w.generation = v;
                })
              }
            />
          </div>
        </div>
      </Section>
      <Section title={<Term term="True Quantities">Total effects (avg over cells)</Term>}>
        {derived && Object.keys(derived.total_effects).length ? (
          <table className="data" aria-label="Total effects" data-testid="total-effects">
            <tbody>
              {Object.entries(derived.total_effects).map(([k, v]) => (
                <tr key={k}>
                  <td>
                    {w.arrows.some((a) => arrowKey(a) === k) ? (
                      <button type="button" className="link" onClick={() => select({ type: "arrow", id: k })}>
                        {k.replace(" -> ", " → ")}
                      </button>
                    ) : (
                      k.replace(" -> ", " → ")
                    )}
                  </td>
                  <td className="r">
                    <DerivedValue
                      value={v}
                      label={`Total effect ${k} (π-weighted average over cells, raw units)`}
                      rippleKey={`total:${k}`}
                      testId={`total-effect-${k}`}
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <Empty>Draw arrows between latents to see total effects.</Empty>
        )}
      </Section>
      <Section title="Equations">
        {Object.keys(w.nodes).length ? (
          <div className="flex flex-col gap-1.5" data-testid="world-equations">
            {Object.keys(w.nodes).map((n) => (
              <NodeEquationView key={n} name={n} world={w} derived={derived} testId={`equation-${n}`} />
            ))}
            {unfoldOn && unfolded ? <PrimitivesLegend unfolded={unfolded} /> : null}
          </div>
        ) : (
          <Empty>No nodes yet.</Empty>
        )}
      </Section>
    </div>
  );
}

// ----------------------------------------------------------------------------- dispatcher

export function Inspector() {
  const sel = useUI((s) => s.selection);
  const w = useDoc((s) => s.doc.world);
  let body: React.ReactNode;
  let key = sel.type;
  switch (sel.type) {
    case "node":
      body = <NodeInspector name={sel.id} />;
      key = `node:${sel.id}` as typeof key;
      break;
    case "attribute":
      body = <AttributeInspector name={sel.id} />;
      key = `attr:${sel.id}` as typeof key;
      break;
    case "arrow":
      body = <ArrowInspector arrowId={sel.id} />;
      key = `arrow:${sel.id}` as typeof key;
      break;
    case "twist": {
      const t = w.twists[sel.index];
      body = t ? (
        <ArrowInspector arrowId={t.arrow} twistIndex={sel.index} />
      ) : (
        <Empty>This twist no longer exists.</Empty>
      );
      key = `twist:${sel.index}` as typeof key;
      break;
    }
    case "shift":
      body = <ShiftInspector index={sel.index} />;
      key = `shift:${sel.index}` as typeof key;
      break;
    default:
      body = <WorldInspector />;
  }
  return (
    <aside
      className="scroll-y h-full border-line border-l bg-panel"
      aria-label="Inspector"
      data-testid="inspector"
    >
      {sel.type !== "world" ? (
        <div className="flex items-center justify-between border-line border-b px-4 py-2">
          <button
            type="button"
            className="link text-[12.5px]"
            onClick={() => useUI.getState().select({ type: "world" })}
          >
            ← World settings
          </button>
          <span className="text-[11.5px] text-faint">Enter commits · Esc reverts</span>
        </div>
      ) : null}
      <div key={key}>{body}</div>
    </aside>
  );
}

export { showIssue };
