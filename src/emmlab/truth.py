"""Analytic truth engine (research note R4).

Three truths per step (system_design §3):
  T1 structural  — the World's per-cell slopes / total effects,
  T2 causal      — pi-weighted average effect over the analysed (sub)population,
  T3 estimand    — what a regression on a given (filtered, transformed) population converges to.

All functions are pure numpy and work on a ``CompiledWorld``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from .world import CompiledWorld

# ---------------------------------------------------------------------------------------------
# Per-cell and pooled moments
# ---------------------------------------------------------------------------------------------


def cell_moments(cw: CompiledWorld) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """mu_g = A_g alpha_g, Sigma_g = A_g Psi A_g^T for every cell. Returns (mu, Sigma, A)."""
    A = cw.A()
    mu = np.einsum("gij,gj->gi", A, cw.alpha)
    sig = np.einsum("gij,gj,gkj->gik", A, cw.psi_g, A)
    return mu, sig, A


def mixture(w: np.ndarray, mu: np.ndarray, sig: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w = w / w.sum()
    m = w @ mu
    second = np.einsum("g,gij->ij", w, sig + np.einsum("gi,gj->gij", mu, mu))
    return m, second - np.outer(m, m)


def pooled_moments(cw: CompiledWorld, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    mu, sig, _ = cell_moments(cw)
    w = cw.pi if mask is None else np.where(mask, cw.pi, 0.0)
    return mixture(w, mu, sig)


def _students_factors(cw: CompiledWorld, n: int | float | None) -> tuple[float, float, float]:
    """Student-weighted E[(1-1/n)], E[(1-1/n)^2], E[(n-1)/n^2] for the centering correction."""
    if n is None:
        return 1.0, 1.0, 0.0
    if isinstance(n, (int, float)) and n != -1:
        nn = float(n)
        return 1 - 1 / nn, (1 - 1 / nn) ** 2, (nn - 1) / nn**2
    sps = cw.world.schools.students_per_school
    if sps.dist == "fixed":
        ns, pmf = np.array([sps.n], float), np.array([1.0])
    elif sps.dist == "uniform":
        ns = np.arange(sps.min, sps.max + 1, dtype=float)
        pmf = np.ones_like(ns) / len(ns)
    else:
        ns = np.arange(sps.min, sps.max + 1, dtype=float)
        pmf = stats.poisson.pmf(ns, sps.n)
        pmf = pmf / pmf.sum()
    w = pmf * ns
    w = w / w.sum()
    return (float(w @ (1 - 1 / ns)), float(w @ (1 - 1 / ns) ** 2), float(w @ ((ns - 1) / ns**2)))


def within_cov(cw: CompiledWorld, n: int | float | None = None, mask: np.ndarray | None = None) -> np.ndarray:
    """Pooled within-school covariance Sigma_W (R4 §4), optionally restricted to cells in ``mask``."""
    pop = population(cw, "centered", n=n, frame=mask)
    m, cov = pop.moments()
    return cov


# ---------------------------------------------------------------------------------------------
# Populations: per-cell first and second moments after a (linear) transform
# ---------------------------------------------------------------------------------------------


class NotAnalytic(Exception):
    """Raised when a quantity has no closed form and needs an oracle simulation."""


@dataclass
class Population:
    vars: list[str]
    w: np.ndarray  # (G,) unnormalised weights
    mean: np.ndarray  # (G, q)
    second: np.ndarray  # (G, q, q)  E[z z^T | cell]
    assigns: list[dict[str, str]]

    def restrict(self, mask: np.ndarray) -> Population:
        return Population(self.vars, np.where(mask, self.w, 0.0), self.mean, self.second, self.assigns)

    def moments(self) -> tuple[np.ndarray, np.ndarray]:
        w = self.w / self.w.sum()
        m = w @ self.mean
        s = np.einsum("g,gij->ij", w, self.second)
        return m, s - np.outer(m, m)

    def linear(self, M: np.ndarray, c: np.ndarray | None = None, names: list[str] | None = None) -> Population:
        """Apply z' = M z + c (same map in every cell)."""
        c = np.zeros(M.shape[0]) if c is None else c
        mean = self.mean @ M.T + c
        sec = np.einsum("ij,gjk,lk->gil", M, self.second, M)
        mc = np.einsum("ij,gj->gi", M, self.mean)
        sec = sec + np.einsum("gi,j->gij", mc, c) + np.einsum("i,gj->gij", c, mc) + np.outer(c, c)[None]
        return Population(names or [f"z{i}" for i in range(M.shape[0])], self.w, mean, sec, self.assigns)

    def total_weight(self) -> float:
        return float(self.w.sum())


def population(cw: CompiledWorld, normalisation: str = "none", *, n: int | float | None = -1,
               frame: np.ndarray | None = None, visible_only: bool = False) -> Population:
    """Per-cell moments of the continuous variables after normalisation.

    ``n``: students per school used for school-mean centering (``-1`` = from the World design,
    ``None`` = infinite schools). ``frame`` restricts the population (and the school means) to cells.
    """
    mu, sig, A = cell_moments(cw)
    w = cw.pi.copy() if frame is None else np.where(frame, cw.pi, 0.0)
    if normalisation == "none":
        second = sig + np.einsum("gi,gj->gij", mu, mu)
        pop = Population(cw.vars, w, mu, second, cw.assigns)
    elif normalisation == "centered":
        c1, c2, c3 = _students_factors(cw, n)
        psi_wg = np.where(cw.school_level[None], 0.0, cw.psi_g)
        psi_bg = np.where(cw.school_level[None], cw.psi_g, 0.0)
        G, p = mu.shape
        mean = np.zeros_like(mu)
        second = np.zeros_like(sig)
        for s in np.unique(cw.school_key):
            idx = np.where(cw.school_key == s)[0]
            ws = w[idx]
            if ws.sum() <= 0:
                continue
            ph = ws / ws.sum()
            mbar = ph @ mu[idx]
            Abar = np.einsum("h,hij->ij", ph, A[idx])
            M = np.zeros((len(idx), p, p))
            for k, g in enumerate(idx):
                d = mu[g] - mbar
                dA = A[g] - Abar
                M[k] = (A[g] * psi_wg[g]) @ A[g].T + (dA * psi_bg[g]) @ dA.T + np.outer(d, d)
            Mbar = np.einsum("h,hij->ij", ph, M)
            for k, g in enumerate(idx):
                mean[g] = c1 * (mu[g] - mbar)
                second[g] = c2 * M[k] + c3 * Mbar
        pop = Population(cw.vars, w, mean, second, cw.assigns)
    else:
        raise NotAnalytic(f"normalisation '{normalisation}' has no closed form; use an oracle simulation")
    if visible_only:
        keep = [i for i, v in enumerate(cw.vars) if not v.startswith("u:")]  # obs:/pv: stay (data vars)
        pop = Population([cw.vars[i] for i in keep], pop.w, pop.mean[:, keep],
                         pop.second[:, keep][:, :, keep], pop.assigns)
    return pop


# ---------------------------------------------------------------------------------------------
# Projection (OLS estimand)
# ---------------------------------------------------------------------------------------------


@dataclass
class OLSEstimand:
    intercept: float
    coef: dict[str, float]
    r2: float
    resid_var: float
    var_y: float
    sd: dict[str, float]


def _regressor_map(pop: Population, regs: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell linear maps for regressors; supports 'var', 'attr=val' dummies, 'attr=val:var'."""
    G, q = pop.mean.shape
    k = len(regs)
    M = np.zeros((G, k, q))
    c = np.zeros((G, k))
    idx = {v: i for i, v in enumerate(pop.vars)}
    for r, name in enumerate(regs):
        if "=" in name:
            cond, _, var = name.partition(":")
            attr, _, val = cond.partition("=")
            on = np.array([a[attr] == val for a in pop.assigns], float)
            if var:
                M[:, r, idx[var]] = on
            else:
                c[:, r] = on
        else:
            M[:, r, idx[name]] = 1.0
    return M, c


def ols(pop: Population, y: str, regs: list[str]) -> OLSEstimand:
    """Population OLS of y on regs (+ intercept): beta = E[xx']^{-1} E[xy] with x = (1, regs)."""
    names = list(regs) + [y]
    M, c = _regressor_map(pop, names)
    mean = np.einsum("gkq,gq->gk", M, pop.mean) + c
    sec = np.einsum("gkq,gqr,glr->gkl", M, pop.second, M)
    mc = np.einsum("gkq,gq->gk", M, pop.mean)
    sec = sec + np.einsum("gk,gl->gkl", mc, c) + np.einsum("gk,gl->gkl", c, mc) + np.einsum("gk,gl->gkl", c, c)
    w = pop.w / pop.w.sum()
    m = w @ mean
    S = np.einsum("g,gkl->kl", w, sec)
    k = len(regs)
    # augment with intercept
    X2 = np.zeros((k + 1, k + 1))
    X2[0, 0] = 1.0
    X2[0, 1:] = m[:k]
    X2[1:, 0] = m[:k]
    X2[1:, 1:] = S[:k, :k]
    xy = np.concatenate([[m[k]], S[:k, k]])
    beta = np.linalg.solve(X2, xy)
    var_y = S[k, k] - m[k] ** 2
    cov = S - np.outer(m, m)
    resid = var_y - beta[1:] @ cov[:k, :k] @ beta[1:] if k else var_y
    sd = {n: float(math.sqrt(max(cov[i, i], 0.0))) for i, n in enumerate(names)}
    return OLSEstimand(float(beta[0]), {r: float(b) for r, b in zip(regs, beta[1:], strict=True)},
                       float(1 - resid / var_y) if var_y > 0 else 0.0, float(resid), float(var_y), sd)


# ---------------------------------------------------------------------------------------------
# Effects and derived quantities
# ---------------------------------------------------------------------------------------------


def total_effects(cw: CompiledWorld) -> np.ndarray:
    """T_g = A_g - I per cell; entry [g, i, j] = total effect of var j on var i."""
    return cw.A() - np.eye(cw.p)[None]


def group_average(cw: CompiledWorld, arr: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    w = cw.pi if mask is None else np.where(mask, cw.pi, 0.0)
    return np.einsum("g,g...->...", w / w.sum(), arr)


def node_r2(cw: CompiledWorld) -> dict[str, float]:
    _, sig = pooled_moments(cw)
    out = {}
    for v in cw.visible:
        j = cw.index[v]
        uj = cw.index.get(f"u:{v}")
        own = cw.psi[j] + (cw.psi[uj] if uj is not None else 0.0)
        out[v] = float(1 - own / sig[j, j]) if sig[j, j] > 0 else 0.0
    return out


def node_icc(cw: CompiledWorld) -> dict[str, float]:
    _, sig = pooled_moments(cw)
    sw = within_cov(cw, n=None)
    return {v: float(1 - sw[cw.index[v], cw.index[v]] / sig[cw.index[v], cw.index[v]])
            if sig[cw.index[v], cw.index[v]] > 0 else 0.0 for v in cw.visible}


def varsortability(cw: CompiledWorld) -> float:
    """Fraction of ancestor->descendant pairs ordered by increasing marginal variance (Reisach 2021)."""
    _, sig = pooled_moments(cw)
    T = group_average(cw, total_effects(cw))
    lat = [v for v in cw.visible if cw.world.nodes[v].kind == "latent"]
    num = den = 0.0
    for a in lat:
        for d in lat:
            if a != d and abs(T[cw.index[d], cw.index[a]]) > 1e-12:
                va, vd = sig[cw.index[a], cw.index[a]], sig[cw.index[d], cw.index[d]]
                num += 1.0 if va < vd else (0.5 if math.isclose(va, vd) else 0.0)
                den += 1
    return num / den if den else float("nan")


def derived(cw: CompiledWorld) -> dict:
    """Everything the canvas shows as Derived: means, SDs, R2, ICC, per-group values, effects.

    Per-cell moments are computed once; pooled and per-group values are mixtures of them.
    """
    mu_g, sig_g, A = cell_moments(cw)
    mu, sig = mixture(cw.pi, mu_g, sig_g)
    T = A - np.eye(cw.p)[None]
    Tbar = group_average(cw, T)
    r2 = {}
    for v in cw.visible:
        j, uj = cw.index[v], cw.index.get(f"u:{v}")
        own = cw.psi[j] + (cw.psi[uj] if uj is not None else 0.0)
        r2[v] = float(1 - own / sig[j, j]) if sig[j, j] > 0 else 0.0
    sw = within_cov(cw, n=None)
    icc = {v: float(1 - sw[cw.index[v], cw.index[v]] / sig[cw.index[v], cw.index[v]])
           if sig[cw.index[v], cw.index[v]] > 0 else 0.0 for v in cw.visible}
    groups_m: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for attr, a in cw.world.attributes.items():
        for val in a.values:
            mask = cw.cell_mask({attr: val})
            w = np.where(mask, cw.pi, 0.0)
            if w.sum() <= 0:
                continue
            gm, gs = mixture(w, mu_g, sig_g)
            groups_m[f"{attr}={val}"] = (gm, gs, mask)
    nodes = {}
    for v in cw.visible:
        j = cw.index[v]
        groups = {k: {"mean": float(gm[j]), "sd": float(math.sqrt(max(gs[j, j], 0.0)))}
                  for k, (gm, gs, _) in groups_m.items()}
        nodes[v] = {"mean": float(mu[j]), "sd": float(math.sqrt(sig[j, j])), "r2": r2[v], "icc": icc[v],
                    "groups": groups, **{k: float(x) for k, x in cw.solved.get(v, {}).items()}}
    arrows = {}
    for a in cw.world.arrows:
        i, j = cw.index[a.child], cw.index[a.parent]
        per_group = {}
        for k, (_, gs, mask) in groups_m.items():
            slope = float(group_average(cw, cw.B[:, i, j], mask))
            tot = float(group_average(cw, T[:, i, j], mask))
            per_group[k] = {"slope": slope, "total": tot,
                            "std": slope * math.sqrt(gs[j, j]) / math.sqrt(gs[i, i]) if gs[i, i] > 0 else 0.0}
        avg_slope = float(group_average(cw, cw.B[:, i, j]))
        arrows[a.key] = {
            "baseline_slope": cw.slopes.get(a.key, float("nan")),
            "avg_slope": avg_slope,
            "avg_total": float(Tbar[i, j]),
            "std": avg_slope * math.sqrt(sig[j, j]) / math.sqrt(sig[i, i]) if sig[i, i] > 0 else 0.0,
            "groups": per_group,
        }
    lat = list(cw.visible)
    total = {f"{s} -> {d}": float(Tbar[cw.index[d], cw.index[s]]) for s in lat for d in lat
             if s != d and abs(Tbar[cw.index[d], cw.index[s]]) > 1e-12}
    lat_only = [v for v in cw.visible if cw.world.nodes[v].kind == "latent"]
    num = den = 0.0
    for s_ in lat_only:
        for d in lat_only:
            if s_ != d and abs(Tbar[cw.index[d], cw.index[s_]]) > 1e-12:
                va, vd = sig[cw.index[s_], cw.index[s_]], sig[cw.index[d], cw.index[d]]
                num += 1.0 if va < vd else (0.5 if math.isclose(va, vd) else 0.0)
                den += 1
    return {"nodes": nodes, "arrows": arrows, "total_effects": total,
            "varsortability": num / den if den else float("nan"), "n_cells": len(cw.assigns),
            "warnings": cw.warnings}


# ---------------------------------------------------------------------------------------------
# Power (R2 §4.2)
# ---------------------------------------------------------------------------------------------


def power_slope_difference(n_g: np.ndarray | float, p: float, delta: float, sigma2: float = 0.91,
                           alpha: float = 0.05, var_x: float = 1.0) -> np.ndarray:
    """Analytic power to detect a subgroup slope difference (two-sided z-test)."""
    n_g = np.asarray(n_g, float)
    se = np.sqrt(sigma2 / (n_g * (1 - p) * var_x))
    z = stats.norm.ppf(1 - alpha / 2)
    return stats.norm.cdf(abs(delta) / se - z) + stats.norm.cdf(-abs(delta) / se - z)


def required_n_g(p: float, delta: float, sigma2: float = 0.91, alpha: float = 0.05, power: float = 0.8) -> float:
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return float(z**2 * sigma2 / (delta**2 * (1 - p)))
