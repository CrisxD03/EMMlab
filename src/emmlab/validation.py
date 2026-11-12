"""Selection-aware validation for real data (design T3, research notes R1 §2 and R3 §2.2).

- Permutation null: shuffle descriptors at their own level (school-level descriptors across schools,
  student-level descriptors within schools), keep the latent block intact, rerun the whole search and
  record the top quality. Its 95th percentile is the search-level threshold.
- Holdout: split schools (stratified) into discovery and confirmation halves; discover on one half,
  test each finding's pre-registered path on the other, Benjamini-Hochberg across findings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .emm import Description, run_emm
from .spec import EMMConfig, SearchSpace
from .stages import AnyPrepared, ols_fit, prepare


def descriptor_levels(df: pd.DataFrame, cols: list[str]) -> tuple[list[str], list[str]]:
    """Split descriptors into school-level (constant within every school) and student-level."""
    school, student = [], []
    for c in cols:
        (school if (df.groupby("school_id", observed=True)[c].nunique() <= 1).all() else student).append(c)
    return school, student


def permute_descriptors(df: pd.DataFrame, cols: list[str], rng: np.random.Generator) -> pd.DataFrame:
    school_cols, student_cols = descriptor_levels(df, cols)
    out = df.copy()
    if school_cols:
        first = df.groupby("school_id", sort=True)[school_cols].first()
        perm = first.iloc[rng.permutation(len(first))].reset_index(drop=True)
        perm.index = first.index
        for c in school_cols:
            out[c] = df["school_id"].map(perm[c]).to_numpy()
            if isinstance(df[c].dtype, pd.CategoricalDtype):
                out[c] = pd.Categorical(out[c], categories=df[c].cat.categories, ordered=df[c].cat.ordered)
    if student_cols:
        idx = np.arange(len(df))
        new = idx.copy()
        for _, rows in df.groupby("school_id", sort=True).indices.items():
            new[rows] = rows[rng.permutation(len(rows))]
        for c in student_cols:
            out[c] = df[c].to_numpy()[new]
            if isinstance(df[c].dtype, pd.CategoricalDtype):
                out[c] = pd.Categorical(out[c], categories=df[c].cat.categories, ordered=df[c].cat.ordered)
    return out


@dataclass
class PermutationNull:
    qualities: list[float]
    threshold: float
    observed_top: float
    p_value: float

    def to_dict(self) -> dict:
        return self.__dict__


def permutation_null(df: pd.DataFrame, prep: AnyPrepared, emm: EMMConfig, search: SearchSpace, reps: int,
                     seed: int | np.random.SeedSequence, observed_top: float | None = None) -> PermutationNull:
    rng = np.random.default_rng(seed)
    from .emm import descriptor_columns
    cols = descriptor_columns(df.loc[prep.frame], prep.model.latents, search)
    # the latent block is untouched, so the prepared arrays are reused: only descriptors move
    qs = []
    for _ in range(reps):
        pdf = permute_descriptors(df, cols, rng)
        res = run_emm(pdf, prep, emm, search, canonical=False)
        qs.append(max((f.quality for f in res.raw), default=0.0))
    if observed_top is None:
        res = run_emm(df, prep, emm, search, canonical=False)
        observed_top = max((f.quality for f in res.raw), default=0.0)
    q = np.array(qs)
    return PermutationNull(qs, float(np.quantile(q, 0.95)), float(observed_top),
                           float((1 + (q >= observed_top).sum()) / (len(q) + 1)))


def split_schools(df: pd.DataFrame, rng: np.random.Generator, strata: list[str]) -> tuple[pd.DataFrame,
                                                                                          pd.DataFrame]:
    first = df.groupby("school_id", sort=True).first()
    key = first[strata].astype(str).agg("|".join, axis=1) if strata else pd.Series("all", index=first.index)
    disc: list = []
    for _, ids in key.groupby(key):
        sch = ids.index.to_numpy()
        sch = sch[rng.permutation(len(sch))]
        disc.extend(sch[: len(sch) // 2].tolist())
    d = set(disc)
    m = df["school_id"].isin(d).to_numpy()
    return df.loc[m].reset_index(drop=True), df.loc[~m].reset_index(drop=True)


def benjamini_hochberg(p: np.ndarray, q: float) -> np.ndarray:
    p = np.asarray(p, float)
    n = len(p)
    if n == 0:
        return np.zeros(0, bool)
    order = np.argsort(p)
    thresh = q * (np.arange(1, n + 1) / n)
    passed = p[order] <= thresh
    k = np.max(np.nonzero(passed)[0]) + 1 if passed.any() else 0
    out = np.zeros(n, bool)
    out[order[:k]] = True
    return out


def holdout(df: pd.DataFrame, model, analysis, emm: EMMConfig, search: SearchSpace, school_strata: list[str],
            seed: int | np.random.SeedSequence, q: float = 0.05) -> dict:
    rng = np.random.default_rng(seed)
    d_df, c_df = split_schools(df, rng, school_strata)
    d_prep = prepare(d_df, model, analysis, search.filter)
    c_prep = prepare(c_df, model, analysis, search.filter)
    res = run_emm(d_df, d_prep, emm, search)
    c_frame = c_df.loc[c_prep.frame].reset_index(drop=True)
    per_group = analysis.standardize == "per_group"
    rows, ps = [], []
    for f in res.findings:
        best = f.evaluation.best
        if best is None:
            continue
        m = Description(f.description.conditions).mask(c_frame)
        base = ~m if emm.baseline == "complement" else np.ones_like(m)
        if m.sum() < 10 or base.sum() < 10:
            ps.append(1.0)
            rows.append({"description": str(f.description), "path": best.path, "p": 1.0, "n_confirm": int(m.sum())})
            continue
        i = c_prep.equations[best.child].index(best.reg)
        bg, sg = c_prep.fit_eq(best.child, m, per_group, True)
        bb, sb = c_prep.fit_eq(best.child, base, per_group, True)
        z = (bg[i] - bb[i]) / np.sqrt(sg[i] ** 2 + sb[i] ** 2)
        p = float(2 * stats.norm.sf(abs(z)))
        ps.append(p)
        rows.append({"description": str(f.description), "path": best.path, "diff_discovery": best.diff,
                     "diff_confirm": float(bg[i] - bb[i]), "z": float(z), "p": p, "n_confirm": int(m.sum())})
    keep = benjamini_hochberg(np.array(ps), q)
    for r, k in zip(rows, keep, strict=True):
        r["confirmed"] = bool(k)
    return {"findings": rows, "q": q, "n_discovery_schools": int(d_df["school_id"].nunique()),
            "n_confirm_schools": int(c_df["school_id"].nunique())}


__all__ = ["permutation_null", "holdout", "benjamini_hochberg", "permute_descriptors", "ols_fit"]
