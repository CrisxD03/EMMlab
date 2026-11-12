import katex from "katex";
import { useMemo } from "react";
import type { Derived, EqTerm, FindingPath, UnfoldedEq, World } from "../api/types";
import { parseArrowKey } from "../state/worldOps";

/** Python's format(x, 'g') with a precision (used by the server's equation strings). */
export function formatG(x: number, precision = 6, sign = false): string {
  if (!Number.isFinite(x)) return String(x);
  let s: string;
  if (x === 0) s = "0";
  else {
    const exp = Math.floor(Math.log10(Math.abs(Number(x.toPrecision(precision)))));
    if (exp < -4 || exp >= precision) {
      const [m, e] = x.toExponential(precision - 1).split("e");
      const mm = m.includes(".") ? m.replace(/0+$/, "").replace(/\.$/, "") : m;
      const en = Number(e);
      s = `${mm}e${en < 0 ? "-" : "+"}${String(Math.abs(en)).padStart(2, "0")}`;
    } else {
      s = String(Number(x.toPrecision(precision)));
      if (s.includes("e")) s = x.toFixed(Math.max(0, precision - 1 - exp));
    }
  }
  if (sign && !s.startsWith("-")) s = `+${s}`;
  return s;
}

const tx = (s: string) => `\\text{${s.replace(/[_&%$#{}]/g, (c) => `\\${c}`)}}`;
const texNum = (x: number, sign = false) => {
  const s = formatG(x, 3, sign);
  return s
    .replace(/^\+/, "+ ")
    .replace(/^-/, "- ")
    .replace(/e([+-])0*(\d+)/, "\\times 10^{$1$2}");
};
const cond = (when: Record<string, string>) =>
  Object.entries(when)
    .map(([k, v]) => `${k}=${v}`)
    .join(" & ");
const condTex = (when: Record<string, string>) =>
  `[${Object.entries(when)
    .map(([k, v]) => `${tx(k)}{=}${tx(v)}`)
    .join("\\,\\&\\,")}]`;

export interface NodeEquation {
  plain: string;
  tex: string;
}

/** The node's equation in the server's plain format (report.world_equations) and as TeX. */
export function nodeEquation(world: World, name: string, derived?: Derived): NodeEquation | null {
  const node = world.nodes[name];
  if (!node) return null;
  if (node.root) {
    const m = node.root.mean ?? 0;
    const sd = node.root.sd ?? 1;
    return {
      plain: `${name} ~ Normal(${formatG(m)}, ${formatG(sd)})`,
      tex: `${tx(name)} \\sim \\mathcal{N}\\!\\left(${texNum(m)},\\ ${texNum(sd)}\\right)`,
    };
  }
  const slopeOf = (from: string) => {
    const key = `${from} -> ${name}`;
    const a = world.arrows.find((x) => x.from === from && x.to === name);
    return derived?.arrows[key]?.baseline_slope ?? a?.slope ?? 0;
  };
  const plain: string[] = [];
  const tex: string[] = [];
  for (const a of world.arrows.filter((x) => x.to === name)) {
    const b = slopeOf(a.from);
    plain.push(`${formatG(b, 3, true)}·${a.from}`);
    tex.push(`${texNum(b, true)}\\,${tx(a.from)}`);
  }
  for (const s of world.shifts.filter((x) => x.target === name)) {
    plain.push(`${formatG(s.delta, 3, true)}·[${cond(s.when)}]`);
    tex.push(`${texNum(s.delta, true)}\\,${condTex(s.when)}`);
  }
  for (const t of world.twists) {
    const [p, c] = parseArrowKey(t.arrow);
    if (c !== name) continue;
    plain.push(`${formatG(t.delta, 3, true)}·[${cond(t.when)}]·${p}`);
    tex.push(`${texNum(t.delta, true)}\\,${condTex(t.when)}\\,${tx(p)}`);
  }
  const intercept = node.recipe?.intercept ?? 0;
  const sd = derived?.nodes[name]?.noise_sd ?? node.recipe?.noise_sd ?? 0;
  return {
    plain: `${name} = ${formatG(intercept)} ${plain.join(" ")} + ε(sd ${formatG(sd, 3)})`,
    tex: `${tx(name)} = ${texNum(intercept)} ${tex.join(" ")} + \\varepsilon,\\quad \\sigma_\\varepsilon = ${texNum(sd)}`,
  };
}

const indTex = (a: string, v: string) => `[${tx(a)}{=}${tx(v)}]`;
const indPlain = (a: string, v: string) => `[${a}=${v}]`;

/**
 * One polynomial term as TeX / plain text, with its sign when `signed`. A coefficient of 1 is left out before a
 * factor; `bare` marks a term that is then empty (just ±1 with no indicator).
 */
function termText(t: EqTerm, signed: boolean, hasFactor: boolean) {
  const unit = Math.abs(Math.abs(t.coef) - 1) < 1e-9;
  const lead = unit && (t.when.length > 0 || hasFactor);
  const sign = t.coef < 0 ? "-" : "+";
  const num = lead ? "" : formatG(Math.abs(t.coef), 3);
  const numTex = lead ? "" : texNum(Math.abs(t.coef));
  const inds = t.when.map(([a, v]) => ({ tex: indTex(a, v), plain: indPlain(a, v) }));
  const tex = [numTex, ...inds.map((i) => i.tex)].filter(Boolean).join("\\,");
  const plain = [num, ...inds.map((i) => i.plain)].filter(Boolean).join("·");
  const bare = lead && t.when.length === 0;
  const show = signed || t.coef < 0;
  const sep = signed ? " " : ""; // "- 0.3 x" after another term, "-0.3 x" at the start
  return {
    tex: show ? `${sign}${sep}${tex}`.trimEnd() : tex,
    plain: show ? `${sign}${sep}${plain}`.trimEnd() : plain,
    bare,
  };
}

/** A coefficient polynomial times an optional factor symbol, as one signed chunk of an equation. */
function chunk(rows: EqTerm[], factor: { tex: string; plain: string } | null, first: boolean) {
  if (rows.length === 1) {
    const t = termText(rows[0], !first, factor !== null);
    if (!factor) return t;
    const join = t.bare ? " " : "\\,";
    return {
      tex: `${t.tex}${t.tex ? join : ""}${factor.tex}`,
      plain: `${t.plain}${t.plain ? (t.bare ? " " : "·") : ""}${factor.plain}`,
    };
  }
  const inner = rows.map((r, i) => termText(r, i > 0, false));
  const sum = { tex: inner.map((x) => x.tex).join(" "), plain: inner.map((x) => x.plain).join(" ") };
  if (!factor) return first ? sum : { tex: `+ ${sum.tex}`, plain: `+ ${sum.plain}` };
  const tex = `\\left(${sum.tex}\\right)\\,${factor.tex}`;
  const plain = `(${sum.plain})·${factor.plain}`;
  return { tex: first ? tex : `+ ${tex}`, plain: first ? plain : `+ ${plain}` };
}

const noiseTex = (kind: "e" | "u", node: string) =>
  kind === "e" ? `\\varepsilon_{${tx(node)}}` : `u_{${tx(node)}}`;

/**
 * A generating equation from the server (`direct`: parents stay as variables; `roots`: only attribute
 * indicators and noise sources). Every factor appears once, as in the supervisor's request.
 */
export function unfoldedEquation(eq: UnfoldedEq): NodeEquation {
  const parts: { tex: string; plain: string }[] = [];
  const push = (rows: EqTerm[], factor: { tex: string; plain: string } | null) => {
    if (rows.length) parts.push(chunk(rows, factor, parts.length === 0));
  };
  push(eq.const, null);
  for (const v of eq.vars) push(v.poly, { tex: tx(v.name), plain: v.name });
  // A node's own noise and its school random intercept enter every equation with the same coefficient: show
  // them as one factor (e_x + u_x) instead of listing the coefficient twice.
  const done = new Set<string>();
  for (const n of eq.noise) {
    if (done.has(n.symbol)) continue;
    const twin = eq.noise.find(
      (m) => m.node === n.node && m.kind !== n.kind && JSON.stringify(m.poly) === JSON.stringify(n.poly),
    );
    if (twin) {
      done.add(twin.symbol);
      const [e, u] = n.kind === "e" ? [n, twin] : [twin, n];
      push(n.poly, {
        tex: `\\left(${noiseTex("e", e.node)} + ${noiseTex("u", u.node)}\\right)`,
        plain: `(${e.symbol.replace(":", "_")} + ${u.symbol.replace(":", "_")})`,
      });
    } else push(n.poly, { tex: noiseTex(n.kind, n.node), plain: n.symbol.replace(":", "_") });
  }
  return {
    plain: `${eq.node} = ${parts.map((p) => p.plain).join(" ") || "0"}`,
    tex: `${tx(eq.node)} = ${parts.map((p) => p.tex).join(" ") || "0"}`,
  };
}

/**
 * The "paths" view of a generating equation: only what the Researcher's Model can see, the latent arrows (with any
 * twists, which are slope changes) and the node's noise. Level shifts are named, not written out.
 */
export function pathsEquation(eq: UnfoldedEq): NodeEquation {
  const base = unfoldedEquation({ ...eq, const: [] });
  const attrs = [...new Set(eq.const.flatMap((t) => t.when.map(([a]) => a)))];
  if (!attrs.length) return base;
  return {
    plain: `${base.plain} + level shifts by ${attrs.join(", ")}`,
    tex: `${base.tex} + \\text{\\small level shifts by ${attrs.join(", ").replace(/_/g, "\\_")}}`,
  };
}

/** Researcher's-model equation for one child inside a subgroup (b_g) or its baseline (b_b). */
export function subgroupEquation(
  child: string,
  paths: FindingPath[],
  which: "b_g" | "b_b",
): NodeEquation | null {
  const rows = paths.filter((p) => parseArrowKey(p.path)[1] === child);
  if (!rows.length) return null;
  const plain = rows.map((p) => `${formatG(p[which], 3, true)}·${parseArrowKey(p.path)[0]}`).join(" ");
  const tex = rows.map((p) => `${texNum(p[which], true)}\\,${tx(parseArrowKey(p.path)[0])}`).join(" ");
  return { plain: `${child} = ${plain} + e`, tex: `${tx(child)} = ${tex.replace(/^\+ /, "")}` };
}

export function modelEquation(child: string, regs: string[], coef: number[]): NodeEquation {
  const plain = regs.map((r, i) => `${formatG(coef[i], 3, true)}·${r}`).join(" ");
  const tex = regs.map((r, i) => `${texNum(coef[i], true)}\\,${tx(r)}`).join(" ");
  return { plain: `${child} = ${plain} + e`, tex: `${tx(child)} = ${tex.replace(/^\+ /, "")} + e` };
}

export function Equation({ eq, testId, label }: { eq: NodeEquation; testId?: string; label?: string }) {
  const html = useMemo(
    () => katex.renderToString(eq.tex, { throwOnError: false, output: "htmlAndMathml", strict: "ignore" }),
    [eq.tex],
  );
  return (
    <div
      className="equation"
      data-equation={eq.plain}
      data-testid={testId}
      role="math"
      aria-label={label ?? eq.plain}
      // biome-ignore lint/security/noDangerouslySetInnerHtml: KaTeX output from our own TeX string
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
