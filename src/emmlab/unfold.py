"""Generating equations written in terms of primitives (supervisor meeting, week of 2026-10-12).

The Researcher's Model shows the *fitted* equations. The truth is the World's generating process, and the gap
between the two is where spurious or phantom findings live. This module writes that process out in two forms:

``direct``  each node in terms of its parents (the compact equation of the spec: slopes, level shifts, twists);
``roots``   every parent substituted again and again until only primitives remain: attribute indicators
            ``[attr=value]``, one noise source ``e:<node>`` per node and a school random intercept ``u:<node>``
            where the node has an ICC.

Like terms are collected, so every factor appears once. Indicators are 0/1 variables: ``[a=x]^2 = [a=x]`` and
two values of one attribute never hold together, so such products vanish. A node's value is affine in its
parents' values with coefficients that depend on indicators only, hence every term of a ``roots`` equation is
an indicator product times at most one noise source, and in each cell the equation is ``mean + sum of
coefficient x noise``: exactly what ``truth.cell_moments`` computes, which the tests compare to the last digit.

Internally standardized Worlds (``generation: internally_standardized``) rescale every latent when compiled;
the rescaling factor is recovered from the compiled tensors so the equations are those of the generated data.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .world import CompiledWorld

Indicator = tuple[str, str]
Mono = frozenset[Indicator]
Poly = dict[Mono, float]
Symbol = tuple[str, str]  # ("one", ""), ("var", parent), ("e", node), ("u", node)
Expr = dict[Symbol, Poly]

ONE: Symbol = ("one", "")
TOL = 1e-12


class UnfoldUnavailable(Exception):
    """The generating equation of a node cannot be recovered from the compiled World."""


def _mono_mul(a: Mono, b: Mono) -> Mono | None:
    merged = dict(a)
    for attr, val in b:
        if merged.get(attr, val) != val:
            return None  # two values of one attribute never hold together
        merged[attr] = val
    return frozenset(merged.items())


def _poly_add(into: Poly, other: Poly, scale: float = 1.0) -> None:
    for m, c in other.items():
        into[m] = into.get(m, 0.0) + scale * c


def _poly_mul(p: Poly, q: Poly) -> Poly:
    out: Poly = {}
    for m1, c1 in p.items():
        for m2, c2 in q.items():
            m = _mono_mul(m1, m2)
            if m is not None:
                out[m] = out.get(m, 0.0) + c1 * c2
    return out


def _expr_add(into: Expr, sym: Symbol, poly: Poly) -> None:
    _poly_add(into.setdefault(sym, {}), poly)


def _scale(cw: CompiledWorld, v: str) -> float:
    """Factor by which compilation rescaled node ``v`` (1 for ``naive`` generation)."""
    if cw.world.generation == "naive":
        return 1.0
    j = cw.index[v]
    uj = cw.index.get(f"u:{v}")
    if uj is not None:
        return float(cw.B[0, j, uj])
    noise_sd = cw.solved.get(v, {}).get("noise_sd", 0.0)
    if noise_sd > 0:
        return math.sqrt(cw.psi[j]) / noise_sd
    raise UnfoldUnavailable(f"'{v}' has no noise to recover its internal standardization from")


def _direct(cw: CompiledWorld, v: str) -> Expr:
    """The compact equation of node ``v`` as polynomials over attribute indicators, symbols left unexpanded."""
    world = cw.world
    j = cw.index[v]
    k = _scale(cw, v)
    shifts = [s for s in world.shifts if s.target == v]
    const: Poly = {}
    cell0 = {a: val for a, val in cw.assigns[0].items()}
    const[frozenset()] = float(cw.alpha[0, j]) - sum(
        s.delta * k for s in shifts if all(cell0[a] == val for a, val in s.when.items()))
    for s in shifts:
        _poly_add(const, {frozenset(s.when.items()): s.delta * k})
    expr: Expr = {ONE: const}
    for arrow in (a for a in world.arrows if a.child == v):
        poly: Poly = {frozenset(): cw.slopes[arrow.key] * k}
        for t in world.twists:
            if t.parent == arrow.parent and t.child == v:
                _poly_add(poly, {frozenset(t.when.items()): t.delta * k})
        _expr_add(expr, ("var", arrow.parent), poly)
    expr[("e", v)] = {frozenset(): 1.0}
    uj = cw.index.get(f"u:{v}")
    if uj is not None and cw.psi[uj] > 0:
        expr[("u", v)] = {frozenset(): float(cw.B[0, j, uj])}
    return expr


def _substitute(cw: CompiledWorld, direct: dict[str, Expr]) -> dict[str, Expr]:
    out: dict[str, Expr] = {}
    for v in cw.visible:  # topological order: parents come first
        expr: Expr = {}
        for sym, poly in direct[v].items():
            if sym[0] == "var":
                for sym2, poly2 in out[sym[1]].items():
                    _expr_add(expr, sym2, _poly_mul(poly, poly2))
            else:
                _expr_add(expr, sym, poly)
        out[v] = expr
    return out


def _poly_rows(poly: Poly, order: dict[str, int]) -> list[dict[str, Any]]:
    rows = [{"coef": float(c), "when": sorted(([a, x] for a, x in m), key=lambda t: (order.get(t[0], 99), t[1]))}
            for m, c in poly.items() if abs(c) > TOL]
    return sorted(rows, key=lambda r: (len(r["when"]), [(order.get(a, 99), x) for a, x in r["when"]]))


def _equation(cw: CompiledWorld, v: str, expr: Expr, form: str) -> dict[str, Any]:
    order = {a: i for i, a in enumerate(cw.world.attributes)}
    pos = {n: i for i, n in enumerate(cw.visible)}
    const = _poly_rows(expr.get(ONE, {}), order)
    vars_ = [{"name": s[1], "poly": rows} for s in sorted((s for s in expr if s[0] == "var"), key=lambda s: pos[s[1]])
             if (rows := _poly_rows(expr[s], order))]
    noise = []
    for sym in sorted((s for s in expr if s[0] in ("e", "u")), key=lambda s: (pos[s[1]], s[0] == "u")):
        rows = _poly_rows(expr[sym], order)
        if not rows:
            continue
        idx = cw.index[sym[1]] if sym[0] == "e" else cw.index[f"u:{sym[1]}"]
        noise.append({"symbol": f"{sym[0]}:{sym[1]}", "kind": sym[0], "node": sym[1],
                      "sd": math.sqrt(float(cw.psi[idx])), "poly": rows})
    return {"node": v, "form": form, "const": const, "vars": vars_, "noise": noise}


def unfold_world(cw: CompiledWorld) -> dict[str, Any]:
    """Both forms of every node's generating equation, plus the primitives they are written in."""
    direct: dict[str, Expr] = {}
    unavailable: dict[str, str] = {}
    for v in cw.visible:
        try:
            direct[v] = _direct(cw, v)
        except UnfoldUnavailable as e:
            unavailable[v] = str(e)
    if unavailable:
        return {"equations": {"roots": {}, "direct": {}}, "primitives": {"attributes": [], "noise": []},
                "unavailable": unavailable}
    roots = _substitute(cw, direct)
    eqs = {"direct": {v: _equation(cw, v, direct[v], "direct") for v in cw.visible},
           "roots": {v: _equation(cw, v, roots[v], "roots") for v in cw.visible}}
    attrs = [{"name": n, "level": a.level, "role": a.role, "values": dict(a.values), "depends_on": a.depends_on}
             for n, a in cw.world.attributes.items()]
    noise = [{"symbol": f"{kind}:{v}", "kind": kind, "node": v, "sd": math.sqrt(float(cw.psi[cw.index[idx]]))}
             for v in cw.visible for kind, idx in (("e", v), ("u", f"u:{v}"))
             if idx in cw.index and cw.psi[cw.index[idx]] > 0]
    return {"equations": eqs, "primitives": {"attributes": attrs, "noise": noise}, "unavailable": {}}


def evaluate_cell(eq: dict[str, Any], assign: dict[str, str]) -> tuple[float, float]:
    """Mean and variance of a ``roots`` equation in the cell where each attribute takes ``assign``'s value."""
    def value(poly: list[dict[str, Any]]) -> float:
        return float(sum(t["coef"] for t in poly if all(assign[a] == x for a, x in t["when"])))

    mean = value(eq["const"])
    var = float(np.sum([value(n["poly"]) ** 2 * n["sd"] ** 2 for n in eq["noise"]]))
    return mean, var


def unfold_fitted(latents: list[str], coefs: dict[str, dict[str, float]]) -> dict[str, dict[str, Any]]:
    """A fitted Researcher's Model unfolded to its own primitives: the starting latents (no parents in the model)
    stay as variables and every other latent contributes its residual ``e:<node>``.

    Each coefficient is then the model-implied total effect of that starting latent or residual. The fitted model
    contains no attributes, so nothing here is indicator-dependent; residual SDs are not part of a fit summary.
    """
    exprs: dict[str, Expr] = {}
    for v in latents:  # the model's latents are listed parents-first
        regs = coefs.get(v)
        if not regs:
            exprs[v] = {("var", v): {frozenset(): 1.0}}
            continue
        expr: Expr = {("e", v): {frozenset(): 1.0}}
        for r, b in regs.items():
            for sym, poly in exprs[r].items():
                _expr_add(expr, sym, {m: b * c for m, c in poly.items()})
        exprs[v] = expr
    pos = {n: i for i, n in enumerate(latents)}
    out = {}
    for v, expr in exprs.items():
        vars_ = [{"name": s[1], "poly": _poly_rows(expr[s], {})}
                 for s in sorted((s for s in expr if s[0] == "var"), key=lambda s: pos[s[1]])]
        noise = [{"symbol": f"e:{s[1]}", "kind": "e", "node": s[1], "sd": None, "poly": _poly_rows(expr[s], {})}
                 for s in sorted((s for s in expr if s[0] == "e"), key=lambda s: pos[s[1]])]
        out[v] = {"node": v, "form": "fitted", "const": [], "vars": [x for x in vars_ if x["poly"]],
                  "noise": [x for x in noise if x["poly"]]}
    return out
