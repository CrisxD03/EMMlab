"""World validation and compilation into per-cell linear SCMs (research note R4 §1).

A *cell* is one value for every categorical attribute. Within a cell the World is an ordinary
linear Gaussian SCM ``v = alpha_g + B_g v + eps`` over all continuous variables ``v`` (latents,
continuous attributes and hidden school random intercepts ``u:<node>``), in topological order.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from .spec import Arrow, CategoricalAttribute, World

PROB_TOL = 1e-6


class Issue(dict):
    """A validation issue: element path + message."""

    def __init__(self, element: str, message: str):
        super().__init__(element=element, message=message)

    def __str__(self) -> str:
        return f"{self['element']}: {self['message']}"


class WorldValidationError(ValueError):
    def __init__(self, issues: list[Issue]):
        self.issues = issues
        super().__init__("; ".join(str(i) for i in issues))


class InfeasibleTargetError(WorldValidationError):
    pass


# ---------------------------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------------------------


def _incoming(world: World) -> dict[str, list[str]]:
    inc: dict[str, list[str]] = {n: [] for n in world.nodes}
    for a in world.arrows:
        if a.child in inc:
            inc[a.child].append(a.parent)
    return inc


def is_child(world: World, node: str) -> bool:
    return any(a.child == node for a in world.arrows) or any(s.target == node for s in world.shifts)


def validate_world(world: World) -> list[Issue]:
    """Return a list of issues (empty when the World is valid)."""
    issues: list[Issue] = []
    cats, nodes = world.attributes, world.nodes

    for name in set(cats) & set(nodes):
        issues.append(Issue(name, "name used both as categorical attribute and continuous node"))

    # Categorical attributes: probabilities, dependencies.
    for name, attr in cats.items():
        s = sum(attr.values.values())
        if abs(s - 1) > PROB_TOL:
            issues.append(Issue(f"attributes.{name}", f"proportions sum to {s:.4f}, not 1"))
        if attr.depends_on is not None:
            parent = cats.get(attr.depends_on)
            if parent is None:
                issues.append(Issue(f"attributes.{name}.depends_on",
                                    f"'{attr.depends_on}' is not a categorical attribute"))
                continue
            if attr.level == "school" and parent.level == "student":
                issues.append(Issue(f"attributes.{name}.depends_on",
                                    "a school-level attribute cannot depend on a student-level one"))
            if attr.table is None:
                issues.append(Issue(f"attributes.{name}.table",
                                    "depends_on requires a table of proportions per parent value"))
                continue
            for pv in parent.values:
                row = attr.table.get(pv)
                if row is None:
                    issues.append(Issue(f"attributes.{name}.table", f"missing row for {attr.depends_on}={pv}"))
                    continue
                if set(row) != set(attr.values):
                    issues.append(Issue(f"attributes.{name}.table.{pv}", "values must match the attribute's values"))
                rs = sum(row.values())
                if abs(rs - 1) > PROB_TOL:
                    issues.append(Issue(f"attributes.{name}.table.{pv}", f"proportions sum to {rs:.4f}, not 1"))
    g = nx.DiGraph()
    g.add_nodes_from(cats)
    for name, attr in cats.items():
        if attr.depends_on in cats:
            g.add_edge(attr.depends_on, name)
    if not nx.is_directed_acyclic_graph(g):
        cyc = " -> ".join(n for n, _ in nx.find_cycle(g))
        issues.append(Issue("attributes", f"attribute dependencies form a cycle: {cyc}"))

    # Arrows.
    seen: set[str] = set()
    for i, a in enumerate(world.arrows):
        el = f"arrows[{i}] ({a.key})"
        for end in (a.parent, a.child):
            if end not in nodes:
                hint = " (categorical attributes act through shifts, not arrows)" if end in cats else ""
                issues.append(Issue(el, f"'{end}' is not a continuous node{hint}"))
        if a.key in seen:
            issues.append(Issue(el, "duplicate arrow"))
        seen.add(a.key)
        if a.parent in nodes and a.child in nodes:
            pn, cn = nodes[a.parent], nodes[a.child]
            if cn.level == "school" and pn.level == "student":
                issues.append(Issue(el, "a school-level node can only have school-level parents"))
            if cn.kind == "attribute" and pn.kind == "latent":
                issues.append(Issue(el, "a continuous attribute cannot be caused by a latent"))
    cg = nx.DiGraph()
    cg.add_nodes_from(nodes)
    cg.add_edges_from((a.parent, a.child) for a in world.arrows if a.parent in nodes and a.child in nodes)
    if not nx.is_directed_acyclic_graph(cg):
        cyc = " -> ".join(n for n, _ in nx.find_cycle(cg))
        issues.append(Issue("arrows", f"arrows form a cycle: {cyc}"))

    # Roots and children.
    for name, node in nodes.items():
        el = f"nodes.{name}"
        child = is_child(world, name)
        if child and node.root is not None:
            issues.append(Issue(el, "has incoming arrows or shifts, so it is a Child: give a recipe, not a root"))
        if child and node.recipe is None:
            issues.append(Issue(el, "is a Child (has incoming arrows or shifts) but has no recipe"))
        if not child and node.recipe is not None:
            issues.append(Issue(el, "has no incoming arrows or shifts, so it is a Root: give a root distribution"))
        if not child and node.root is None:
            issues.append(Issue(el, "is a Root but has no root distribution"))
        if node.level == "school" and (node.school_sd > 0 or (node.target and node.target.icc is not None)):
            issues.append(Issue(el, "school random intercept / icc only applies to student-level nodes"))
        if child and node.recipe is not None and node.recipe.noise_sd is None:
            t = node.target
            if t is None or (t.sd is None and t.r2 is None):
                issues.append(Issue(el, "a Child needs noise_sd or a target (sd or r2) to solve it"))

    # Measurement layer and plausible values.
    for name, node in nodes.items():
        el = f"nodes.{name}"
        if (node.reliability is not None or node.reliability_when or node.plausible_values) and \
                (node.level == "school" or node.kind != "latent"):
            issues.append(Issue(el, "reliability / plausible values apply to student-level latents only"))
        if node.plausible_values is not None:
            for c in node.plausible_values.conditioning:
                if c not in cats and c not in nodes:
                    issues.append(Issue(f"{el}.plausible_values", f"unknown conditioning variable '{c}'"))
                if c == name:
                    issues.append(Issue(f"{el}.plausible_values", "a node cannot condition on itself"))
        for i, rule in enumerate(node.reliability_when):
            for attr, val in rule.when.items():
                if attr not in cats or val not in cats[attr].values:
                    issues.append(Issue(f"{el}.reliability_when[{i}]", f"'{attr}={val}' is not a categorical value"))
    # Noise shapes and observation scales.
    for name, node in nodes.items():
        el = f"nodes.{name}"
        if node.noise is not None and node.noise.skew != 0 and node.level == "school":
            issues.append(Issue(f"{el}.noise", "a noise shape applies to student-level nodes only"))
        if node.scale is not None:
            if node.level == "school" or node.kind != "latent" or not node.observed:
                issues.append(Issue(f"{el}.scale", "an observation scale applies to observed student-level latents"))
            if node.reliability is not None or node.reliability_when or node.plausible_values is not None:
                issues.append(Issue(f"{el}.scale", "an observation scale cannot be combined with reliability or "
                                                   "plausible values"))
    sd = world.schools.sampling
    if sd is not None:
        for i, rule in enumerate(sd.oversample):
            for attr, val in rule.when.items():
                if attr not in cats or cats[attr].level != "school" or val not in cats[attr].values:
                    issues.append(Issue(f"schools.sampling.oversample[{i}]",
                                        "oversampling conditions must name school-level categorical values"))

    # Conditions of shifts and twists.
    def check_when(el: str, when: dict[str, str], target_level: str | None) -> None:
        for attr, val in when.items():
            if attr in nodes:
                issues.append(Issue(el, f"condition on continuous '{attr}': conditions may only use categorical "
                                        "attributes (declare it as an ordinal categorical attribute instead)"))
                continue
            if attr not in cats:
                issues.append(Issue(el, f"unknown attribute '{attr}' in condition"))
                continue
            if val not in cats[attr].values:
                issues.append(Issue(el, f"'{val}' is not a value of '{attr}'"))
            if target_level == "school" and cats[attr].level == "student":
                issues.append(Issue(el, "a school-level node cannot depend on a student-level attribute"))
        if not when:
            issues.append(Issue(el, "condition must name at least one attribute"))

    for i, s in enumerate(world.shifts):
        el = f"shifts[{i}]"
        if s.target not in nodes:
            issues.append(Issue(el, f"target '{s.target}' is not a continuous node"))
            check_when(el, s.when, None)
        else:
            check_when(el, s.when, nodes[s.target].level)
    arrow_keys = {a.key for a in world.arrows}
    for i, t in enumerate(world.twists):
        el = f"twists[{i}] ({t.arrow})"
        try:
            key = f"{t.parent} -> {t.child}"
        except ValueError as e:
            issues.append(Issue(el, str(e)))
            continue
        if key not in arrow_keys:
            issues.append(Issue(el, "a twist can only modify an existing arrow"))
            check_when(el, t.when, None)
        else:
            check_when(el, t.when, nodes[t.child].level if t.child in nodes else None)
    return issues


def has_scales(world: World) -> bool:
    """Does the researcher see any latent through an observation scale (so truths need the oracle)?"""
    return any(n.scale is not None for n in world.nodes.values())


def without_scales(world: World) -> World:
    """The same World with every latent observed directly (the latent-scale truth)."""
    nodes = {k: v.model_copy(update={"scale": None}) for k, v in world.nodes.items()}
    return world.model_copy(update={"nodes": nodes})


def twist_coding_warnings(world: World) -> list[str]:
    """Warn when every value of an attribute twists the same arrow (baseline applies to nobody)."""
    out = []
    by_key: dict[tuple[str, str], set[str]] = {}
    for t in world.twists:
        if len(t.when) == 1:
            (attr, val), = t.when.items()
            by_key.setdefault((t.arrow.replace(" ", ""), attr), set()).add(val)
    for (arrow, attr), vals in by_key.items():
        if attr in world.attributes and vals >= set(world.attributes[attr].values):
            out.append(f"every value of '{attr}' twists {arrow}: the arrow's baseline slope applies to nobody; "
                       "consider folding one twist into the baseline")
    return out


# ---------------------------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------------------------


@dataclass
class CompiledWorld:
    world: World
    vars: list[str]  # continuous variables in topological order (incl. hidden "u:<node>")
    index: dict[str, int]
    school_level: np.ndarray  # bool (p,)
    psi: np.ndarray  # noise variances (p,)
    attr_order: list[str]  # categorical attributes in dependency order
    assigns: list[dict[str, str]]  # per cell
    pi: np.ndarray  # (G,)
    school_key: np.ndarray  # (G,) int index of the school-level part of the cell
    pi_school: np.ndarray  # (S,)
    alpha: np.ndarray  # (G, p)
    B: np.ndarray  # (G, p, p)
    solved: dict[str, dict[str, float]] = field(default_factory=dict)
    slopes: dict[str, float] = field(default_factory=dict)  # resolved raw baseline slopes
    warnings: list[str] = field(default_factory=list)
    psi_cell: np.ndarray | None = None  # (G, p) per-cell noise variances (measurement layer)
    pi_sample: np.ndarray | None = None  # (G,) cell shares in the *sample* (oversampled strata)
    pv: dict[str, dict] = field(default_factory=dict)  # plausible-value parameters per node

    @property
    def psi_g(self) -> np.ndarray:
        return self.psi_cell if self.psi_cell is not None else np.broadcast_to(self.psi, self.alpha.shape)

    def data_var(self, name: str) -> str:
        """The variable a Dataset column stands for: plausible value, observed index or the latent itself."""
        for pre in ("pv:", "obs:"):
            if f"{pre}{name}" in self.index:
                return f"{pre}{name}"
        return name

    def with_composition(self, kind: str) -> CompiledWorld:
        """Copy with cell shares of the population (default) or of the (oversampled) sample."""
        import dataclasses
        if kind == "sample" and self.pi_sample is not None:
            return dataclasses.replace(self, pi=self.pi_sample)
        return self

    @property
    def p(self) -> int:
        return len(self.vars)

    @property
    def visible(self) -> list[str]:
        return [v for v in self.vars if ":" not in v]

    def cell_mask(self, when: Mapping[str, str | list[str]] | None) -> np.ndarray:
        if not when:
            return np.ones(len(self.assigns), dtype=bool)
        out = np.ones(len(self.assigns), dtype=bool)
        for i, a in enumerate(self.assigns):
            for attr, val in when.items():
                vals = val if isinstance(val, list) else [val]
                if a[attr] not in vals:
                    out[i] = False
                    break
        return out

    def A(self) -> np.ndarray:
        eye = np.eye(self.p)
        return np.linalg.solve(eye[None] - self.B, np.broadcast_to(eye, self.B.shape))


def _attr_order(world: World) -> list[str]:
    g = nx.DiGraph()
    names = list(world.attributes)
    g.add_nodes_from(names)
    for n, a in world.attributes.items():
        if a.depends_on:
            g.add_edge(a.depends_on, n)
    rank = {n: (0 if world.attributes[n].level == "school" else 1, i) for i, n in enumerate(names)}
    return list(nx.lexicographical_topological_sort(g, key=lambda n: rank[n]))


def _cond_probs(attr: CategoricalAttribute, assign: dict[str, str]) -> dict[str, float]:
    if attr.depends_on and attr.table:
        return attr.table[assign[attr.depends_on]]
    return attr.values


def _enumerate_cells(world: World, order: list[str]):
    school_attrs = [a for a in order if world.attributes[a].level == "school"]
    student_attrs = [a for a in order if world.attributes[a].level == "student"]
    school_parts: list[tuple[dict[str, str], float]] = []
    for combo in itertools.product(*[list(world.attributes[a].values) for a in school_attrs]):
        assign = dict(zip(school_attrs, combo, strict=True))
        p = 1.0
        for a in school_attrs:
            p *= _cond_probs(world.attributes[a], assign)[assign[a]]
        school_parts.append((assign, p))
    cells, pis, skeys = [], [], []
    for si, (sassign, ps) in enumerate(school_parts):
        for combo in itertools.product(*[list(world.attributes[a].values) for a in student_attrs]):
            assign = dict(sassign)
            assign.update(zip(student_attrs, combo, strict=True))
            p = ps
            for a in student_attrs:
                p *= _cond_probs(world.attributes[a], assign)[assign[a]]
            cells.append(assign)
            pis.append(p)
            skeys.append(si)
    return cells, np.array(pis), np.array(skeys), np.array([p for _, p in school_parts])


def _var_order(world: World) -> list[str]:
    g = nx.DiGraph()
    names = list(world.nodes)
    g.add_nodes_from(names)
    g.add_edges_from((a.parent, a.child) for a in world.arrows)
    rank = {n: i for i, n in enumerate(names)}
    order = list(nx.lexicographical_topological_sort(g, key=lambda n: rank[n]))
    out: list[str] = []
    for n in order:
        node = world.nodes[n]
        if node.level == "student" and (node.school_sd > 0 or (node.target and node.target.icc is not None)
                                        or (node.root is not None and node.target and node.target.icc)):
            out.append(f"u:{n}")
        out.append(n)
    return out


def _matches(when: dict[str, str], assign: dict[str, str]) -> bool:
    return all(assign[a] == v for a, v in when.items())


def compile_world(world: World, *, check: bool = True) -> CompiledWorld:
    """Validate, enumerate cells, resolve std arrows and solve targets (in topological order)."""
    if check:
        issues = validate_world(world)
        if issues:
            raise WorldValidationError(issues)
    order = _attr_order(world)
    assigns, pi, skey, pi_s = _enumerate_cells(world, order)
    vars_ = _var_order(world)
    idx = {v: i for i, v in enumerate(vars_)}
    p, G = len(vars_), len(assigns)
    school_level = np.array([
        (world.nodes[v].level == "school") if not v.startswith("u:") else True for v in vars_
    ])
    alpha = np.zeros((G, p))
    B = np.zeros((G, p, p))
    psi = np.zeros(p)
    cw = CompiledWorld(world, vars_, idx, school_level, psi, order, assigns, pi, skey, pi_s, alpha, B)
    cw.warnings.extend(twist_coding_warnings(world))
    from . import truth  # local import: truth depends on CompiledWorld

    arrows_into: dict[str, list[Arrow]] = {}
    for a in world.arrows:
        arrows_into.setdefault(a.child, []).append(a)

    for v in vars_:
        if v.startswith("u:"):
            continue
        j = idx[v]
        node = world.nodes[v]
        t = node.target
        uj = idx.get(f"u:{v}")
        if node.root is not None:
            alpha[:, j] = node.root.mean
            for s in world.shifts:  # (roots have no shifts by definition, kept for safety)
                if s.target == v:
                    alpha[cw.cell_mask(s.when), j] += s.delta
            var = node.root.sd ** 2
            icc = t.icc if (t and t.icc is not None) else 0.0
            if uj is not None:
                psi[uj] = icc * var
                B[:, j, uj] = 1.0
                psi[j] = (1 - icc) * var
            else:
                psi[j] = var
            cw.solved[v] = {"noise_sd": math.sqrt(psi[j]), "school_sd": math.sqrt(psi[uj]) if uj is not None else 0.0}
        else:
            rec = node.recipe
            assert rec is not None
            alpha[:, j] = rec.intercept
            for s in world.shifts:
                if s.target == v:
                    alpha[cw.cell_mask(s.when), j] += s.delta
            # resolve slopes (std arrows need the child's target SD and the parents' pooled SDs)
            needs_std = any(a.std is not None for a in arrows_into.get(v, []))
            sd_now = (np.sqrt(np.clip(np.diag(truth.pooled_moments(cw)[1]), 0, None))
                      if needs_std else np.zeros(p))
            t_sd = t.sd if (t and t.sd is not None) else None
            for a in arrows_into.get(v, []):
                if a.slope is not None:
                    slope = a.slope
                else:
                    assert a.std is not None
                    child_sd = t_sd if t_sd is not None else 1.0
                    psd = sd_now[idx[a.parent]]
                    slope = a.std * child_sd / psd if psd > 0 else 0.0
                B[:, j, idx[a.parent]] = slope
                cw.slopes[a.key] = slope
            for tw in world.twists:
                if tw.child == v:
                    B[cw.cell_mask(tw.when), j, idx[tw.parent]] += tw.delta
            if uj is not None:
                B[:, j, uj] = 1.0
            if t is None:
                noise_var, school_var = (rec.noise_sd or 0.0) ** 2, node.school_sd ** 2
            else:
                psi[j] = 0.0
                if uj is not None:
                    psi[uj] = 0.0
                _, sig = truth.pooled_moments(cw)
                v_expl = float(sig[j, j])
                b_exist = 0.0
                if t.icc is not None:
                    sw = truth.within_cov(cw, n=None)
                    b_exist = float(sig[j, j] - sw[j, j])
                noise_var, school_var = _solve_node(v, node, v_expl, b_exist)
            psi[j] = noise_var
            if uj is not None:
                psi[uj] = school_var
            cw.solved[v] = {"noise_sd": math.sqrt(noise_var), "school_sd": math.sqrt(school_var)}
        if world.generation == "internally_standardized" and world.nodes[v].kind == "latent":
            mu, sig = truth.pooled_moments(cw)
            m, s = mu[j], math.sqrt(sig[j, j])
            if s > 0:
                alpha[:, j] = (alpha[:, j] - m) / s
                B[:, j, :] /= s
                psi[j] /= s**2
                if uj is not None:
                    B[:, j, uj] = 1.0 / s
    _add_measurement(cw)
    _add_composition(cw)
    idx = cw.index
    psi = cw.psi
    _, sig = truth.pooled_moments(cw)
    for v in cw.visible:
        j, uj = idx[v], idx.get(f"u:{v}")
        own = psi[j] + (psi[uj] if uj is not None else 0.0)
        cw.solved.setdefault(v, {})["explained_var"] = float(max(sig[j, j] - own, 0.0))
    return cw


def _solve_node(name: str, node, v_expl: float, b_exist: float) -> tuple[float, float]:
    t = node.target
    rec = node.recipe
    school_given = node.school_sd ** 2
    el = f"nodes.{name}.target"
    if t is None or (t.sd is None and t.r2 is None):
        noise = (rec.noise_sd or 0.0) ** 2
        if t is not None and t.icc is not None:
            school = (t.icc * (v_expl + noise) - b_exist) / (1 - t.icc)
            if school < 0:
                raise InfeasibleTargetError([Issue(el, f"icc {t.icc} is below the between-school share already "
                                                       f"implied by the parents ({b_exist / (v_expl + noise):.3f})")])
            return noise, school
        return noise, school_given
    total = t.sd ** 2 if t.sd is not None else v_expl / t.r2  # type: ignore[operator]
    if t.icc is not None:
        school = t.icc * total - b_exist
        if school < 0:
            t_min = math.sqrt(b_exist / t.icc) if t.icc > 0 else float("inf")
            raise InfeasibleTargetError([Issue(el, f"infeasible icc: between-school variance from parents is "
                                                   f"{b_exist:.4f}; need sd >= {t_min:.3f}")])
    else:
        school = school_given
    noise = total - v_expl - school
    if noise <= 0:
        t_min = math.sqrt(v_expl + school)
        what = f"sd {t.sd}" if t.sd is not None else f"r2 {t.r2}"
        raise InfeasibleTargetError([Issue(el, f"infeasible target {what}: parents already explain "
                                               f"variance {v_expl:.4f}; t_min = {t_min:.3f}")])
    return noise, school


# ---------------------------------------------------------------------------------------------
# Measurement layer, plausible values, sample composition (M6)
# ---------------------------------------------------------------------------------------------


def _extend(cw: CompiledWorld, name: str) -> int:
    """Append a student-level variable (a leaf: nothing in the World depends on it)."""
    G, p = cw.alpha.shape
    psi_cell = np.array(cw.psi_g, copy=True)
    cw.vars.append(name)
    cw.index[name] = p
    cw.school_level = np.append(cw.school_level, False)
    cw.psi = np.append(cw.psi, 0.0)
    cw.psi_cell = np.concatenate([psi_cell, np.zeros((G, 1))], axis=1)
    cw.alpha = np.concatenate([cw.alpha, np.zeros((G, 1))], axis=1)
    B = np.zeros((G, p + 1, p + 1))
    B[:, :p, :p] = cw.B
    cw.B = B
    return p


def _add_measurement(cw: CompiledWorld) -> None:
    from . import truth
    world = cw.world
    for v, node in world.nodes.items():
        pvc = node.plausible_values
        base = pvc.reliability if pvc is not None else node.reliability
        if base is None and not node.reliability_when:
            continue
        base = 1.0 if base is None else base
        _, sig = truth.pooled_moments(cw)
        var_true = float(sig[cw.index[v], cw.index[v]])
        o = _extend(cw, f"obs:{v}")
        cw.B[:, o, cw.index[v]] = 1.0
        rho = np.full(len(cw.assigns), base)
        for rule in node.reliability_when:
            rho[cw.cell_mask(rule.when)] = rule.reliability
        err = var_true * (1 - rho) / rho
        cw.psi_cell[:, o] = err  # type: ignore[index]
        cw.psi[o] = float(cw.pi @ err)
        cw.solved.setdefault(v, {})["reliability"] = float(base)
        if pvc is not None:
            _add_pv(cw, v, pvc, o, float(cw.pi @ err))


def _add_pv(cw: CompiledWorld, v: str, pvc, o: int, err_var: float) -> None:
    """PV = a*s + (1-a)*(conditioning prediction) + posterior noise (main-effects conditioning)."""
    from . import truth
    world = cw.world
    regs, cont = [], []
    for c in pvc.conditioning:
        if c in world.attributes:
            regs += [f"{c}={val}" for val in list(world.attributes[c].values)[1:]]
        elif c in world.nodes:
            dv = cw.data_var(c) if not c.startswith(("obs:", "pv:")) else c
            regs.append(dv)
            cont.append(dv)
        else:
            raise WorldValidationError([Issue(f"nodes.{v}.plausible_values.conditioning",
                                              f"unknown conditioning variable '{c}'")])
    pop = truth.population(cw, "none")
    e = truth.ols(pop, f"obs:{v}", regs)
    tau2 = e.resid_var - err_var
    if tau2 <= 0:
        raise InfeasibleTargetError([Issue(f"nodes.{v}.plausible_values", "conditioning model explains more "
                                           "than the true-score variance; lower the test reliability")])
    a = tau2 / (tau2 + err_var)
    post = tau2 * err_var / (tau2 + err_var)
    p = _extend(cw, f"pv:{v}")
    cw.B[:, p, o] = a
    cw.alpha[:, p] = (1 - a) * e.intercept
    for r in regs:
        if "=" in r:
            attr, _, val = r.partition("=")
            cw.alpha[cw.cell_mask({attr: val}), p] += (1 - a) * e.coef[r]
        else:
            cw.B[:, p, cw.index[r]] += (1 - a) * e.coef[r]
    cw.psi_cell[:, p] = post  # type: ignore[index]
    cw.psi[p] = post
    cw.pv[v] = {"n": pvc.n, "shrinkage_a": a, "posterior_var": post, "tau2": tau2, "error_var": err_var,
                "conditioning": dict(e.coef), "intercept": e.intercept}


def _add_composition(cw: CompiledWorld) -> None:
    sd = cw.world.schools.sampling
    if sd is None or not sd.oversample:
        return
    f = np.ones(len(cw.assigns))
    for rule in sd.oversample:
        f[cw.cell_mask(rule.when)] *= rule.factor
    cw.pi_sample = cw.pi * f / (cw.pi * f).sum()


def school_factors(cw: CompiledWorld) -> np.ndarray:
    """Oversampling factor per school-level cell part (indexed like ``pi_school``)."""
    out = np.ones(len(cw.pi_school))
    sd = cw.world.schools.sampling
    if sd is None:
        return out
    for s in range(len(cw.pi_school)):
        g = int(np.nonzero(cw.school_key == s)[0][0])
        for rule in sd.oversample:
            if all(cw.assigns[g].get(a) == val for a, val in rule.when.items()):
                out[s] *= rule.factor
    return out
