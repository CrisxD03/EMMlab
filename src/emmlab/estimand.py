"""T3 engine: the analytic mirror of ``stages.prepare`` + ``Prepared.fit``.

For a World, a Researcher's Model and analysis settings, return what each regression converges
to on any categorical subgroup — the step estimand (system_design §3). Normalisations without a
closed form (within-school standardization) and threshold-defined subgroups use an oracle
simulation of the identical data pipeline.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from . import truth
from .spec import AnalysisConfig, ResearcherModel
from .stages import apply_residualization, residualization_coefs
from .world import CompiledWorld, has_scales


@dataclass
class EqEstimand:
    child: str
    regs: list[str]
    coef: np.ndarray
    intercept: float
    r2: float
    sd: dict[str, float]

    def as_dict(self) -> dict:
        return {"child": self.child, "coef": dict(zip(self.regs, self.coef.tolist(), strict=True)),
                "r2": self.r2}


class EstimandEngine:
    def __init__(self, cw: CompiledWorld, model: ResearcherModel, analysis: AnalysisConfig,
                 frame: dict[str, str] | None = None, n: int | float | None = -1, scale: bool = True,
                 true_scores: bool = False):
        if not true_scores and has_scales(cw.world):
            raise truth.NotAnalytic("observation scales (censored / count / ordinal) have no closed form; "
                                    "use an oracle simulation")
        # weighted analyses target the population; unweighted ones the (oversampled) sample composition
        cw = cw.with_composition("population" if analysis.weights == "final" else "sample")
        if analysis.pv_mode == "mean" and cw.pv and not true_scores:
            import dataclasses
            psi_cell = np.array(cw.psi_g, copy=True)
            for v, info in cw.pv.items():
                psi_cell[:, cw.index[f"pv:{v}"]] /= info["n"]
            cw = dataclasses.replace(cw, psi_cell=psi_cell)
        self.cw, self.model, self.analysis = cw, model, analysis
        self.frame_cells = cw.cell_mask(frame) if frame else None
        pop = truth.population(cw, analysis.normalisation, n=n, frame=self.frame_cells, visible_only=True)
        lat = model.latents
        idx = [pop.vars.index(v if true_scores else cw.data_var(v)) for v in lat]
        pop = truth.Population(lat, pop.w, pop.mean[:, idx], pop.second[:, idx][:, :, idx], pop.assigns)
        m, cov = pop.moments()
        sd = np.sqrt(np.clip(np.diag(cov), 0, None))
        sd_safe = np.where(sd > 0, sd, 1.0) if scale else np.ones_like(sd)
        shift = -m / sd_safe if scale else np.zeros_like(m)
        self.pop = pop.linear(np.diag(1 / sd_safe), shift, names=lat)  # global standardization
        self.li = {v: i for i, v in enumerate(lat)}
        self.equations = model.equations()
        full0 = {}
        for c, regs in self.equations.items():
            e = truth.ols(self.pop, c, regs)
            full0[c] = e.coef
        _, cov_std = self.pop.moments()
        self.coefs = residualization_coefs(model, analysis, full0, None, self.li, cov=cov_std)
        self._maps: dict[str, np.ndarray] = {}
        for c, regs in self.equations.items():
            rows = []
            for r in regs:
                vec = np.zeros(len(lat))
                vec[self.li[r]] = 1.0
                rows.append(apply_residualization(vec, r, model, analysis, self.coefs, self.li))
            yv = np.zeros(len(lat))
            yv[self.li[c]] = 1.0
            rows.append(yv)
            self._maps[c] = np.array(rows)

    def fit(self, cell_mask: np.ndarray | None = None, *, per_group: bool | None = None) -> dict[str, EqEstimand]:
        pg = (self.analysis.standardize == "per_group") if per_group is None else per_group
        pop = self.pop if cell_mask is None else self.pop.restrict(cell_mask)
        if pop.w.sum() <= 0:
            raise ValueError("empty population")
        out = {}
        for c, regs in self.equations.items():
            names = [f"x{i}" for i in range(len(regs))] + ["y"]
            sub = pop.linear(self._maps[c], names=names)
            e = truth.ols(sub, "y", names[:-1])
            coef = np.array([e.coef[n] for n in names[:-1]])
            if pg:
                sx = np.array([e.sd[n] for n in names[:-1]])
                coef = coef * sx / e.sd["y"] if e.sd["y"] > 0 else coef * 0
            out[c] = EqEstimand(c, regs, coef, e.intercept, e.r2, e.sd)
        return out

    @cached_property
    def full(self) -> dict[str, EqEstimand]:
        return self.fit()

    def mask(self, when: Mapping[str, str | list[str]] | None) -> np.ndarray:
        m = self.cw.cell_mask(when)
        if self.frame_cells is not None:
            m &= self.frame_cells
        return m


def total_effect_truth(cw: CompiledWorld, src: str, dst: str, when: dict | None = None) -> float:
    """T2: pi-weighted average total effect of src on dst over the cells matching ``when``."""
    T = truth.total_effects(cw)
    return float(truth.group_average(cw, T[:, cw.index[dst], cw.index[src]], cw.cell_mask(when)))
