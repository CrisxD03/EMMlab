"""Analysis stages on data: frame -> normalise -> standardize (once) -> residualize -> OLS.

The same sequence is mirrored analytically in ``estimand.py`` so every stage has a T3 to check.
Sampling weights (``W_FSTUWT``), Fay-BRR replicate weights (``W_FSTURWT1..R``) and plausible values
(``<latent>_pv1..M``, pooled with Rubin's rules) are supported for PISA-style data.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

from .spec import AnalysisConfig, ResearcherModel

WEIGHT = "W_FSTUWT"
REP_PREFIX = "W_FSTURWT"


def frame_mask(df: pd.DataFrame, filt: dict[str, str] | None) -> np.ndarray:
    if not filt:
        return np.ones(len(df), dtype=bool)
    m = np.ones(len(df), dtype=bool)
    for a, v in filt.items():
        m &= (df[a].astype(str) == v).to_numpy()
    return m


def normalise(values: np.ndarray, school: np.ndarray, mode: str, w: np.ndarray | None = None) -> np.ndarray:
    """(Weighted) school-mean centering or within-school standardization of columns."""
    if mode == "none":
        return values.copy()
    codes, inv = np.unique(school, return_inverse=True)
    G = len(codes)
    ww = np.ones(len(values)) if w is None else w
    cnt = np.bincount(inv, weights=ww, minlength=G)
    out = np.empty_like(values)
    for j in range(values.shape[1]):
        mean = np.bincount(inv, weights=ww * values[:, j], minlength=G) / cnt
        c = values[:, j] - mean[inv]
        if mode == "within_std":
            sd = np.sqrt(np.bincount(inv, weights=ww * c**2, minlength=G) / cnt)
            c = np.divide(c, sd[inv], out=np.zeros_like(c), where=sd[inv] > 0)
        out[:, j] = c
    return out


def model_graph(model: ResearcherModel) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_nodes_from(model.latents)
    g.add_edges_from(model.edges())
    return g


@dataclass
class EqFit:
    child: str
    regs: list[str]
    coef: np.ndarray  # (k,) slopes
    intercept: float
    se: np.ndarray  # (k,) cluster-robust, naive or BRR (+ imputation variance with PVs)
    se_naive: np.ndarray
    n: int
    n_schools: int
    r2: float

    def as_dict(self) -> dict:
        return {"child": self.child, "coef": dict(zip(self.regs, self.coef.tolist(), strict=True)),
                "se": dict(zip(self.regs, self.se.tolist(), strict=True)), "n": self.n,
                "n_schools": self.n_schools, "r2": self.r2, "intercept": self.intercept}


def school_codes(school: np.ndarray) -> tuple[np.ndarray, int]:
    _, codes = np.unique(school, return_inverse=True)
    return codes.astype(np.int64), int(codes.max()) + 1 if len(codes) else 0


def ols_fit(X: np.ndarray, y: np.ndarray, school: np.ndarray, cluster: bool = True,
            standardize: bool = False, n_groups: int | None = None,
            weights: np.ndarray | None = None) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, float]:
    """(W)LS with intercept. Returns (slopes, intercept, se_cluster, se_naive, r2).

    ``school`` may be raw ids, or integer codes when ``n_groups`` is given (fast path).
    Cluster-robust SEs use the statsmodels small-sample correction G/(G-1) * (N-1)/(N-K).
    ``standardize`` rescales X and y by their own (weighted) SDs first (per-group standardization).
    ``weights`` are sampling weights (normalise to mean 1); without them every formula reduces to OLS.
    """
    n, k = X.shape
    w = np.ones(n) if weights is None else np.asarray(weights, float)
    if standardize:
        sw = w.sum()
        mx = (w @ X) / sw
        sx = np.sqrt((w @ (X - mx) ** 2) / sw)
        my = (w @ y) / sw
        sy = math.sqrt((w @ (y - my) ** 2) / sw)
        X = np.divide(X - mx, sx, out=np.zeros_like(X), where=sx > 0)
        y = (y - my) / sy if sy > 0 else y * 0
    X1 = np.column_stack([np.ones(n), X])
    Xw = X1 * w[:, None]
    xtx = Xw.T @ X1
    try:
        inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError:
        nan = np.full(k, np.nan)
        return nan, float("nan"), nan, nan, float("nan")
    beta = inv @ (Xw.T @ y)
    e = y - X1 @ beta
    K = k + 1
    s2 = (w * e) @ e / max(n - K, 1)
    se_naive = np.sqrt(np.clip(np.diag(inv) * s2, 0, None))[1:]
    if n_groups is None:
        codes, G_all = school_codes(school)
    else:
        codes, G_all = school, n_groups
    if cluster and n > K:
        scores = Xw * e[:, None]
        S = np.column_stack([np.bincount(codes, weights=scores[:, j], minlength=G_all) for j in range(K)])
        G = int(np.count_nonzero(np.bincount(codes, minlength=G_all)))
        meat = S.T @ S
        corr = G / (G - 1) * (n - 1) / max(n - K, 1) if G > 1 else float("nan")
        V = inv @ meat @ inv * corr
        se = np.sqrt(np.clip(np.diag(V), 0, None))[1:]
    else:
        se = se_naive
    ybar = (w @ y) / w.sum()
    vy = (w @ (y - ybar) ** 2)
    r2 = float(1 - ((w * e) @ e) / vy) if vy > 0 else 0.0
    return beta[1:], float(beta[0]), se, se_naive, r2


def rubin(coefs: np.ndarray, ses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rubin's rules over M imputations: mean estimate, SE^2 = mean(SE^2) + (1 + 1/M) * between-var."""
    M = coefs.shape[0]
    est = coefs.mean(0)
    within = (ses**2).mean(0)
    between = coefs.var(0, ddof=1) if M > 1 else np.zeros_like(est)
    return est, np.sqrt(within + (1 + 1 / M) * between)


class StatsBank:
    """Per-row sufficient statistics of all equations side by side, with a cache of the last mask.

    A candidate subgroup is evaluated for every equation and for its complement; all of those need
    the same per-school sums, so they are computed once per mask.
    """

    def __init__(self, blocks: list[np.ndarray], codes: np.ndarray, n_groups: int):
        from scipy import sparse
        n = len(codes)
        self.R = np.concatenate(blocks, axis=1) if blocks else np.zeros((n, 0))
        self.I = sparse.csr_matrix((np.ones(n), (codes, np.arange(n))), shape=(n_groups, n))
        self.total = np.asarray(self.I @ self.R)
        self.count = np.asarray(self.I @ np.ones(n)).ravel()
        self._key: bytes | None = None
        self._val: tuple[np.ndarray, np.ndarray] | None = None

    def sums(self, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        key = np.packbits(mask).tobytes()
        if key != self._key:
            idx = np.flatnonzero(mask)
            sub = np.asarray(self.I[:, idx] @ self.R[idx])
            cnt = np.asarray(self.I[:, idx].sum(axis=1)).ravel()
            self._key, self._val = key, (sub, cnt)
        assert self._val is not None
        return self._val


def row_stats(X: np.ndarray, y: np.ndarray, w: np.ndarray | None) -> np.ndarray:
    n, k = X.shape
    Z = np.column_stack([np.ones(n), X, y])
    m = k + 2
    R = (Z[:, :, None] * Z[:, None, :]).reshape(n, m * m)
    return R * w[:, None] if w is not None else R


class EqStats:
    """Sufficient statistics for fast repeated (W)LS fits of one equation on arbitrary row masks.

    Per row we store w * z z' with z = (1, X, y). Per-school sums of any mask are one sparse product,
    and the complement is total minus subgroup. Coefficients, cluster-robust and naive SEs, R² and
    per-group standardization follow exactly from these sums (standardization is an affine
    reparameterization, so SEs scale by sd_x / sd_y).
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, w: np.ndarray | None, codes: np.ndarray, n_groups: int,
                 bank: StatsBank | None = None, offset: int = 0):
        n, k = X.shape
        self.k, self.m, self.n, self.G = k, k + 2, n, n_groups
        if bank is None:
            bank = StatsBank([row_stats(X, y, w)], codes, n_groups)
        self.bank, self.lo, self.hi = bank, offset, offset + self.m * self.m
        self.total = bank.total[:, self.lo:self.hi]
        self.count = bank.count

    def sums(self, mask: np.ndarray | None, complement: bool = False) -> tuple[np.ndarray, np.ndarray]:
        if mask is None:
            return self.total, self.count
        sub_all, cnt = self.bank.sums(mask)
        sub = sub_all[:, self.lo:self.hi]
        if complement:
            return self.total - sub, self.count - cnt
        return sub, cnt

    def fit(self, mask: np.ndarray | None, *, complement: bool = False, cluster: bool = True,
            standardize: bool = False) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, float, int, int]:
        per, cnt = self.sums(mask, complement)
        k, m = self.k, self.m
        K = k + 1
        n = int(round(cnt.sum()))
        nan = np.full(k, np.nan)
        if n <= K:
            return nan, float("nan"), nan, nan, float("nan"), n, int((cnt > 0).sum())
        S = per.sum(0).reshape(m, m)
        xtx, xty, yty = S[:K, :K], S[:K, K], S[K, K]
        try:
            inv = np.linalg.inv(xtx)
        except np.linalg.LinAlgError:
            return nan, float("nan"), nan, nan, float("nan"), n, int((cnt > 0).sum())
        beta = inv @ xty
        ssr = float(yty - 2 * beta @ xty + beta @ xtx @ beta)
        sw = S[0, 0]
        ybar = S[0, K] / sw
        sst = float(yty - sw * ybar**2)
        r2 = 1 - ssr / sst if sst > 0 else 0.0
        s2 = max(ssr, 0.0) / max(n - K, 1)
        se_naive = np.sqrt(np.clip(np.diag(inv) * s2, 0, None))[1:]
        G = int((cnt > 0).sum())
        if cluster and G > 1:
            P = per.reshape(-1, m, m)
            u = P[:, :K, K] - P[:, :K, :K] @ beta  # per-school score sums
            meat = u.T @ u
            corr = G / (G - 1) * (n - 1) / max(n - K, 1)
            se = np.sqrt(np.clip(np.diag(inv @ meat @ inv) * corr, 0, None))[1:]
        else:
            se = se_naive
        b = beta[1:]
        if standardize:
            mx = S[0, 1:K] / sw
            sx = np.sqrt(np.clip(np.diag(S[1:K, 1:K]) / sw - mx**2, 0, None))
            sy = float(np.sqrt(max(yty / sw - ybar**2, 0.0)))
            f = np.where(sx > 0, sx, 0.0) / sy if sy > 0 else np.zeros_like(sx)
            b, se, se_naive = b * f, se * f, se_naive * f
            return b, 0.0, se, se_naive, r2, n, G
        return b, float(beta[0]), se, se_naive, r2, n, G


@dataclass
class Prepared:
    """Stage outputs for one dataset: transformed regressor/outcome arrays per equation."""

    model: ResearcherModel
    analysis: AnalysisConfig
    school: np.ndarray  # (n,) school ids within frame
    frame: np.ndarray  # (N,) bool mask over the original rows
    codes: np.ndarray  # (n,) integer school codes 0..n_groups-1
    n_groups: int
    latents: np.ndarray  # (n, L) after normalise + global standardize
    equations: dict[str, list[str]]
    X: dict[str, np.ndarray]  # child -> (n, k) regressors (after residualization)
    y: dict[str, np.ndarray]
    scale: dict[str, tuple[float, float]] = field(default_factory=dict)  # global (mean, sd)
    that: dict[str, float] = field(default_factory=dict)  # residualization coefficients used
    full: dict[str, EqFit] = field(default_factory=dict)
    w: np.ndarray | None = None  # sampling weights (mean 1) or None
    rep: np.ndarray | None = None  # (n, R) replicate weights on the same scale

    @property
    def n(self) -> int:
        return len(self.school)

    def stats(self, c: str) -> EqStats:
        cache = self.__dict__.setdefault("_stats", {})
        if not cache:
            blocks, offsets, off = [], {}, 0
            for e in self.equations:
                b = row_stats(self.X[e], self.y[e], self.w)
                blocks.append(b)
                offsets[e] = off
                off += b.shape[1]
            bank = StatsBank(blocks, self.codes, self.n_groups)
            for e in self.equations:
                cache[e] = EqStats(self.X[e], self.y[e], self.w, self.codes, self.n_groups, bank, offsets[e])
        return cache[c]

    SUFFSTATS_MAX_ROWS = 60_000  # above this (oracle simulations) per-row outer products cost too much memory

    def _fit_rows(self, c: str, mask: np.ndarray | None, cluster: bool, per_group: bool):
        X, y, codes, w = self.X[c], self.y[c], self.codes, self.w
        if mask is not None:
            X, y, codes = X[mask], y[mask], codes[mask]
            w = None if w is None else w[mask]
        b, a, se, sen, r2 = ols_fit(X, y, codes, cluster=cluster, standardize=per_group, n_groups=self.n_groups,
                                    weights=w)
        return b, a, se, sen, r2, int(len(y)), int(np.count_nonzero(np.bincount(codes, minlength=self.n_groups)))

    def _fit(self, c: str, mask: np.ndarray | None, complement: bool, cluster: bool, per_group: bool):
        if self.n > self.SUFFSTATS_MAX_ROWS:
            m = (~mask if complement else mask) if mask is not None else None
            return self._fit_rows(c, m, cluster, per_group)
        return self.stats(c).fit(mask, complement=complement, cluster=cluster, standardize=per_group)

    def fit_eq(self, c: str, mask: np.ndarray | None, per_group: bool, cluster: bool,
               complement: bool = False) -> tuple[np.ndarray, np.ndarray]:
        b, _, se, _, _, _, _ = self._fit(c, mask, complement, cluster, per_group)
        return b, se

    def _brr_se(self, c: str, mask: np.ndarray | None, per_group: bool, theta: np.ndarray) -> np.ndarray:
        assert self.rep is not None
        R = self.rep.shape[1]
        k = self.analysis.brr_fay_k
        X, y, codes = self.X[c], self.y[c], self.codes
        rep = self.rep
        if mask is not None:
            X, y, codes, rep = X[mask], y[mask], codes[mask], rep[mask]
        draws = np.array([ols_fit(X, y, codes, cluster=False, standardize=per_group, n_groups=self.n_groups,
                                  weights=rep[:, r])[0] for r in range(R)])
        return np.sqrt(((draws - theta) ** 2).sum(0) / (R * (1 - k) ** 2))

    def fit(self, mask: np.ndarray | None = None, *, per_group: bool | None = None,
            cluster: bool | None = None) -> dict[str, EqFit]:
        pg = (self.analysis.standardize == "per_group") if per_group is None else per_group
        cl = (self.analysis.se in ("cluster", "brr")) if cluster is None else cluster
        out = {}
        for c, regs in self.equations.items():
            b, a, se, sen, r2, n, n_sch = self._fit(c, mask, False, cl, pg)
            if self.analysis.se == "brr" and self.rep is not None and cluster is None:
                se = self._brr_se(c, mask, pg, b)
            out[c] = EqFit(c, regs, b, a, se, sen, n, n_sch, r2)
        return out


@dataclass
class MultiPrepared:
    """One Prepared per plausible value; fits are pooled with Rubin's rules (R3 §3, D)."""

    parts: list[Prepared]
    full: dict[str, EqFit] = field(default_factory=dict)

    def __getattr__(self, name: str):  # shared attributes (frame, codes, equations, X/y of PV1, ...)
        return getattr(self.parts[0], name)

    @property
    def n(self) -> int:
        return self.parts[0].n

    def fit_eq(self, c: str, mask: np.ndarray | None, per_group: bool, cluster: bool, complement: bool = False):
        bs, ses = zip(*[p.fit_eq(c, mask, per_group, cluster, complement) for p in self.parts], strict=True)
        return rubin(np.array(bs), np.array(ses))

    def fit(self, mask: np.ndarray | None = None, *, per_group: bool | None = None,
            cluster: bool | None = None) -> dict[str, EqFit]:
        fits = [p.fit(mask, per_group=per_group, cluster=cluster) for p in self.parts]
        out = {}
        for c in fits[0]:
            est, se = rubin(np.array([f[c].coef for f in fits]), np.array([f[c].se for f in fits]))
            f0 = fits[0][c]
            out[c] = EqFit(c, f0.regs, est, float(np.mean([f[c].intercept for f in fits])), se,
                           np.mean([f[c].se_naive for f in fits], 0), f0.n, f0.n_schools,
                           float(np.mean([f[c].r2 for f in fits])))
        return out


def pv_columns(df: pd.DataFrame, latent: str) -> list[str]:
    pat = re.compile(rf"^{re.escape(latent)}_pv(\d+)$")
    cols = [(int(m.group(1)), c) for c in df.columns if (m := pat.match(c))]
    return [c for _, c in sorted(cols)]


def prepare(df: pd.DataFrame, model: ResearcherModel, analysis: AnalysisConfig,
            filt: dict[str, str] | None = None) -> Prepared | MultiPrepared:
    """Run the stages; returns a MultiPrepared (Rubin pooling) when a model latent has PVs."""
    pvs = {lat: pv_columns(df, lat) for lat in model.latents if lat not in df.columns}
    pvs = {k: v for k, v in pvs.items() if v}
    missing = [lat for lat in model.latents if lat not in df.columns and lat not in pvs]
    if missing:
        raise ValueError(f"model latents not in the dataset: {missing}")
    if not pvs:
        return _prepare_one(df, model, analysis, filt, {})
    M = min(len(v) for v in pvs.values())
    if analysis.pv_mode == "first":
        return _prepare_one(df, model, analysis, filt, {k: v[0] for k, v in pvs.items()})
    if analysis.pv_mode == "mean":
        d2 = df.copy()
        for k, v in pvs.items():
            d2[k] = df[v].mean(axis=1)
        return _prepare_one(d2, model, analysis, filt, {})
    parts = [_prepare_one(df, model, analysis, filt, {k: v[m] for k, v in pvs.items()}) for m in range(M)]
    mp = MultiPrepared(parts)
    mp.full = mp.fit()
    return mp


def _prepare_one(df: pd.DataFrame, model: ResearcherModel, analysis: AnalysisConfig,
                 filt: dict[str, str] | None, colmap: dict[str, str]) -> Prepared:
    fm = frame_mask(df, filt)
    sub = df.loc[fm]
    lat = model.latents
    school = sub["school_id"].to_numpy()
    w = rep = None
    if analysis.weights == "final":
        if WEIGHT not in sub.columns:
            raise ValueError(f"weights='final' needs a {WEIGHT} column")
        raw = sub[WEIGHT].to_numpy(float)
        scale_w = raw.mean()
        w = raw / scale_w
        rcols = sorted([c for c in sub.columns if c.startswith(REP_PREFIX)], key=lambda c: int(c[len(REP_PREFIX):]))
        if rcols:
            rep = sub[rcols].to_numpy(float) / scale_w
    if analysis.se == "brr" and rep is None:
        rcols = sorted([c for c in sub.columns if c.startswith(REP_PREFIX)], key=lambda c: int(c[len(REP_PREFIX):]))
        if not rcols:
            raise ValueError("se='brr' needs replicate weights W_FSTURWT1..R")
        base = sub[WEIGHT].to_numpy(float).mean() if WEIGHT in sub.columns else 1.0
        rep = sub[rcols].to_numpy(float) / base
    vals = sub[[colmap.get(v, v) for v in lat]].to_numpy(dtype=float)
    vals = normalise(vals, school, analysis.normalisation, w)
    # standardize once on the full analysis frame (common metric)
    ww = np.ones(len(vals)) if w is None else w
    m = (ww @ vals) / ww.sum()
    s = np.sqrt((ww @ (vals - m) ** 2) / ww.sum())
    vals = np.divide(vals - m, s, out=np.zeros_like(vals), where=s > 0)
    scale = {v: (float(m[i]), float(s[i])) for i, v in enumerate(lat)}
    li = {v: i for i, v in enumerate(lat)}
    eqs = model.equations()

    # full-sample structural fit (unresidualized) for T-hat
    full0 = {}
    for c, regs in eqs.items():
        b, *_ = ols_fit(vals[:, [li[r] for r in regs]], vals[:, li[c]], school, cluster=False, weights=w)
        full0[c] = dict(zip(regs, b, strict=True))
    cov = np.cov(vals, rowvar=False, aweights=ww, bias=True) if len(lat) > 1 else np.ones((1, 1))
    coefs = residualization_coefs(model, analysis, full0, None, li, cov=cov)
    X, y = {}, {}
    for c, regs in eqs.items():
        cols = []
        for r in regs:
            vec = np.zeros(len(lat))
            vec[li[r]] = 1.0
            vec = apply_residualization(vec, r, model, analysis, coefs, li)
            cols.append(vals @ vec)
        X[c] = np.column_stack(cols) if cols else np.zeros((len(vals), 0))
        y[c] = vals[:, li[c]]
    codes, n_groups = school_codes(school)
    prep = Prepared(model, analysis, school, fm, codes, n_groups, vals, eqs, X, y, scale,
                    {f"{x}->{d}": v for (x, d), v in coefs.items()}, w=w, rep=rep)
    prep.full = prep.fit()
    return prep


def residualization_coefs(model: ResearcherModel, analysis: AnalysisConfig,
                          full_fit: dict[str, dict[str, float]], vals: np.ndarray | None,
                          li: dict[str, int], cov: np.ndarray | None = None) -> dict[tuple[str, str], float]:
    """Coefficient subtracted from descendant D for each clicked node X: D <- D - c * X.

    ``total_effect``: model-implied total effect T-hat(X -> D) from the fitted Researcher's Model.
    ``simple``: slope of D on X alone (the vision's literal "residual on X").
    """
    if not analysis.residualize_on:
        return {}
    g = model_graph(model)
    lat = model.latents
    Bh = np.zeros((len(lat), len(lat)))
    for c, d in full_fit.items():
        for r, b in d.items():
            Bh[li[c], li[r]] = b
    That = np.linalg.inv(np.eye(len(lat)) - Bh) - np.eye(len(lat))
    if cov is None and vals is not None:
        cov = np.cov(vals, rowvar=False, bias=True)
    out = {}
    for x in analysis.residualize_on:
        if x not in li:
            continue
        for d in nx.descendants(g, x):
            if analysis.residualize_method == "total_effect":
                out[(x, d)] = float(That[li[d], li[x]])
            else:
                assert cov is not None
                out[(x, d)] = float(cov[li[d], li[x]] / cov[li[x], li[x]])
    return out


def apply_residualization(vec: np.ndarray, reg: str, model: ResearcherModel, analysis: AnalysisConfig,
                          coefs: dict[tuple[str, str], float], li: dict[str, int]) -> np.ndarray:
    out = vec.copy()
    for x in analysis.residualize_on:
        if (x, reg) in coefs and reg != x:
            out[li[x]] -= coefs[(x, reg)]
    return out


def residualization_cases(model: ResearcherModel, x: str) -> dict[str, str]:
    """Label each equation affected by clicking x: 'triangle' or 'chain' (vision Part 4)."""
    g = model_graph(model)
    desc = nx.descendants(g, x) if x in g else set()
    out = {}
    for c, regs in model.equations().items():
        if c == x or not (set(regs) & desc):
            continue
        out[c] = "triangle" if x in regs else "chain"
    return out


def interval(b: float, se: float, level: float) -> tuple[float, float]:
    z = float(stats.norm.ppf(0.5 + level / 2))
    return b - z * se, b + z * se


AnyPrepared = Prepared | MultiPrepared
