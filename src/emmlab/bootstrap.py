"""Stratified school bootstrap with everything recomputed inside each replicate (D7, R3 §2).

Schools are resampled with replacement within strata of school-level attributes, so no stratum
vanishes. Inside each replicate: school means, global standardization, residualization, numeric
cut points (by quantile), subgroup membership, fits and contrasts are all recomputed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .emm import Condition, Description
from .spec import AnalysisConfig, BootstrapConfig, ResearcherModel
from .stages import prepare


def strata_of(df: pd.DataFrame, school_attrs: list[str]) -> pd.Series:
    first = df.groupby("school_id", sort=True).first()
    if not school_attrs:
        return pd.Series("all", index=first.index)
    return first[school_attrs].astype(str).agg("|".join, axis=1)


def resample_schools(df: pd.DataFrame, strata: pd.Series, rng: np.random.Generator) -> pd.DataFrame:
    by_school = {sid: idx for sid, idx in df.groupby("school_id", sort=True).indices.items()}
    picks: list[int] = []
    for _, schools in strata.groupby(strata):
        ids = schools.index.to_numpy()
        picks.extend(rng.choice(ids, size=len(ids), replace=True).tolist())
    parts, new_ids = [], []
    for k, sid in enumerate(picks):
        rows = by_school[sid]
        parts.append(rows)
        new_ids.append(np.full(len(rows), k))
    idx = np.concatenate(parts)
    out = df.iloc[idx].reset_index(drop=True)
    out["school_id"] = np.concatenate(new_ids)
    return out


def recompute_description(desc: Description, df: pd.DataFrame) -> Description:
    """Re-derive numeric cut points from their quantiles in the replicate."""
    conds = []
    for c in desc.conditions:
        if c.quantile is not None and c.op in ("<", ">="):
            v = float(np.quantile(df[c.attr].to_numpy(float), c.quantile))
            conds.append(Condition(c.attr, c.op, v, quantile=c.quantile))
        elif c.op == "in[)" and c.quantile is not None and c.upper_quantile is not None:
            x = df[c.attr].to_numpy(float)
            conds.append(Condition(c.attr, c.op, float(np.quantile(x, c.quantile)),
                                   float(np.quantile(x, c.upper_quantile)), c.quantile, c.upper_quantile))
        else:
            conds.append(c)
    return Description(tuple(conds))


@dataclass
class BootResult:
    full: dict[str, dict]  # path -> {"se", "ci", "reps"}
    findings: list[dict]  # per finding: path -> {"b_g": ci, "b_b": ci, "diff": ci}
    reps: int
    failed: int

    def to_dict(self) -> dict:
        return {"full": self.full, "findings": self.findings, "reps": self.reps, "failed": self.failed}


def school_bootstrap(df: pd.DataFrame, model: ResearcherModel, analysis: AnalysisConfig, filt: dict | None,
                     school_attrs: list[str], descriptions: list[Description], baseline: str,
                     cfg: BootstrapConfig, seed: np.random.SeedSequence | int, per_group: bool | None = None,
                     ) -> BootResult:
    rng = np.random.default_rng(seed)
    strata = strata_of(df, school_attrs if cfg.stratify else [])
    full_draws: dict[str, list[float]] = {}
    find_draws: list[dict[str, dict[str, list[float]]]] = [{} for _ in descriptions]
    failed = 0
    for _ in range(cfg.reps):
        bdf = resample_schools(df, strata, rng)
        try:
            prep = prepare(bdf, model, analysis, filt)
        except Exception:  # degenerate replicate (e.g. zero variance)
            failed += 1
            continue
        fits = prep.fit(per_group=per_group, cluster=False)
        for c, f in fits.items():
            for i, r in enumerate(f.regs):
                full_draws.setdefault(f"{r} -> {c}", []).append(float(f.coef[i]))
        frame = bdf.loc[prep.frame].reset_index(drop=True)
        for k, d in enumerate(descriptions):
            dd = recompute_description(d, frame)
            m = dd.mask(frame)
            base = ~m if baseline == "complement" else np.ones_like(m)
            if m.sum() < 10 or base.sum() < 10:
                continue
            g = prep.fit(m, per_group=per_group, cluster=False)
            b = prep.fit(base, per_group=per_group, cluster=False)
            for c in g:
                for i, r in enumerate(g[c].regs):
                    rec = find_draws[k].setdefault(f"{r} -> {c}", {"b_g": [], "b_b": [], "diff": []})
                    rec["b_g"].append(float(g[c].coef[i]))
                    rec["b_b"].append(float(b[c].coef[i]))
                    rec["diff"].append(float(g[c].coef[i] - b[c].coef[i]))
    lo, hi = (1 - cfg.level) / 2, 1 - (1 - cfg.level) / 2

    def summ(x: list[float]) -> dict:
        a = np.asarray(x, float)
        a = a[np.isfinite(a)]
        if len(a) < 2:
            return {"se": None, "ci": [None, None], "n": int(len(a))}
        return {"se": float(a.std(ddof=1)), "ci": [float(np.quantile(a, lo)), float(np.quantile(a, hi))],
                "n": int(len(a))}

    full = {p: summ(v) for p, v in full_draws.items()}
    finds = [{p: {k: summ(v) for k, v in rec.items()} for p, rec in fd.items()} for fd in find_draws]
    return BootResult(full, finds, cfg.reps, failed)
