"""Grading EMM findings against the World (system_design §7).

Each planted twist is matched to the best canonical finding (cover Jaccard >= tau and the right
sign on the twisted path). Every unmatched finding is classified by cause using its T3 truth:
``null``, ``decoy``, ``phantom_downstream``, ``variance_artifact``, ``level_shift_leak``, ``proxy``,
``scale_artifact`` (a real difference in the observed scores that the latents do not have).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from functools import lru_cache

import networkx as nx
import numpy as np
import pandas as pd

from .emm import Description, EMMResult, Finding, jaccard
from .estimand import EstimandEngine
from .generate import generate
from .spec import AnalysisConfig, GradingConfig, ResearcherModel, World, spec_hash
from .stages import AnyPrepared, Prepared, prepare
from .truth import NotAnalytic
from .world import CompiledWorld, compile_world, has_scales

FP_CLASSES = ["null", "decoy", "phantom_downstream", "variance_artifact", "level_shift_leak", "proxy", "scale_artifact"]


# ---------------------------------------------------------------------------------------------
# Truth provider: analytic (categorical subgroups) or oracle simulation (everything else)
# ---------------------------------------------------------------------------------------------


@lru_cache(maxsize=16)
def _oracle_data(world_json: str, n_students: int) -> pd.DataFrame:
    w = World.model_validate_json(world_json)
    cw = compile_world(w)
    per = w.schools.students_per_school.n
    n_schools = max(n_students // per, 200)
    return generate(cw, 987_654_321, n_schools=n_schools, students=per)


@lru_cache(maxsize=4)
def _oracle_prepared(world_json: str, n_students: int, model_json: str, analysis_json: str,
                     frame_json: str) -> tuple[pd.DataFrame, AnyPrepared]:
    """The oracle sample, prepared like the run; shared by every run of an experiment cell in one process."""
    df = _oracle_data(world_json, n_students)
    frame = json.loads(frame_json)
    prep = prepare(df, ResearcherModel.model_validate_json(model_json),
                   AnalysisConfig.model_validate_json(analysis_json), frame)
    return df.loc[prep.frame].reset_index(drop=True), prep


def _moment_slopes(prep, mask: np.ndarray) -> dict[str, dict[str, float]] | None:
    """OLS slopes on ``mask`` in common and within-standardized units from one cross-product per equation.

    Same numbers as ``prep.fit(mask, per_group=..., cluster=False)`` (within = slope * sd(x) / sd(y)) at half
    the passes over the oracle's rows. None for weighted or plausible-value data, or a singular fit.
    """
    if not isinstance(prep, Prepared) or prep.w is not None:
        return None
    n = int(mask.sum())
    out: dict[str, dict[str, float]] = {"common": {}, "within": {}}
    for c, regs in prep.equations.items():
        Z = np.column_stack([prep.X[c][mask], prep.y[c][mask]])
        mu = Z.mean(0)
        C = (Z.T @ Z) / n - np.outer(mu, mu)  # population covariance, as in per-group standardization
        k = len(regs)
        try:
            beta = np.linalg.solve(C[:k, :k], C[:k, k])
        except np.linalg.LinAlgError:
            return None
        sd = np.sqrt(np.clip(np.diag(C), 0, None))
        if not np.all(np.isfinite(beta)) or sd[k] <= 0 or np.any(sd[:k] <= 0):
            return None
        for i, r in enumerate(regs):
            out["common"][f"{r} -> {c}"] = float(beta[i])
            out["within"][f"{r} -> {c}"] = float(beta[i] * sd[i] / sd[k])
    return out


class TruthProvider:
    """T3 for any description: analytic when possible, else a large-N oracle simulation."""

    def __init__(self, cw: CompiledWorld, model: ResearcherModel, analysis: AnalysisConfig,
                 frame: dict[str, str] | None, grading: GradingConfig):
        self.cw, self.model, self.analysis, self.frame, self.grading = cw, model, analysis, frame, grading
        try:
            self.engine: EstimandEngine | None = EstimandEngine(cw, model, analysis, frame=frame)
        except NotAnalytic:
            self.engine = None
        self._oracle = None
        self._true_engine: EstimandEngine | None = None
        self._contrasts: dict[tuple, dict] = {}

    @property
    def oracle(self):
        if self._oracle is None:
            # the oracle sets its own number of schools, so Worlds that differ only in n_schools share one
            world = self.cw.world.model_copy(update={"schools": self.cw.world.schools.model_copy(
                update={"n_schools": 2})})
            self._oracle = _oracle_prepared(
                world.model_dump_json(by_alias=True), self.grading.oracle_students,
                self.model.model_dump_json(by_alias=True), self.analysis.model_dump_json(by_alias=True),
                json.dumps(self.frame, sort_keys=True))
        return self._oracle

    def contrast(self, desc: Description, baseline: str) -> dict:
        """Coefficient differences (subgroup - baseline) per path: common metric and within-standardized."""
        key = (desc.key(), baseline)
        if key not in self._contrasts:
            self._contrasts[key] = self._contrast(desc, baseline)
        return self._contrasts[key]

    def _contrast(self, desc: Description, baseline: str) -> dict:
        cat = desc.categorical()
        out: dict = {"source": "analytic"}
        if self.engine is not None and cat is not None:
            m = self.engine.mask(cat)
            base = (self.engine.mask(None) & ~m) if baseline == "complement" else self.engine.mask(None)
            if self.cw.pi[m].sum() <= 0 or self.cw.pi[base].sum() <= 0:
                return {"source": "analytic", "common": {}, "within": {}, "empty": True}
            for key, pg in (("common", False), ("within", True)):
                g, b = self.engine.fit(m, per_group=pg), self.engine.fit(base, per_group=pg)
                out[key] = {f"{r} -> {c}": float(g[c].coef[i] - b[c].coef[i])
                            for c in g for i, r in enumerate(g[c].regs)}
            return out
        df, prep = self.oracle
        mask = desc.mask(df)
        base = ~mask if baseline == "complement" else np.ones_like(mask)
        out["source"] = "oracle"
        if mask.sum() < 50 or base.sum() < 50:
            return {"source": "oracle", "common": {}, "within": {}, "empty": True}
        g, b = _moment_slopes(prep, mask), _moment_slopes(prep, base)
        if g is not None and b is not None:
            for key in ("common", "within"):
                out[key] = {p: g[key][p] - b[key][p] for p in g[key]}
            return out
        for key, pg in (("common", False), ("within", True)):
            g2, b2 = prep.fit(mask, per_group=pg, cluster=False), prep.fit(base, per_group=pg, cluster=False)
            out[key] = {f"{r} -> {c}": float(g2[c].coef[i] - b2[c].coef[i])
                        for c in g2 for i, r in enumerate(g2[c].regs)}
        return out

    def true_score_contrast(self, desc: Description, baseline: str) -> dict[str, float] | None:
        """Common-metric contrast computed on true latent scores: before measurement error and before any
        observation scale (None when the data have neither)."""
        cat = desc.categorical()
        scaled = has_scales(self.cw.world)
        if cat is None or not (scaled or (self.engine is not None and
                                          any(v.startswith(("obs:", "pv:")) for v in self.cw.vars))):
            return None
        if self._true_engine is None:
            self._true_engine = EstimandEngine(self.cw, self.model, self.analysis, frame=self.frame, true_scores=True)
        eng = self._true_engine
        m = eng.mask(cat)
        base = (eng.mask(None) & ~m) if baseline == "complement" else eng.mask(None)
        g, b = eng.fit(m, per_group=False), eng.fit(base, per_group=False)
        return {f"{r} -> {c}": float(g[c].coef[i] - b[c].coef[i]) for c in g for i, r in enumerate(g[c].regs)}

    def full_se(self) -> dict[str, dict[str, float]]:
        """Standard error of the T3 value itself: 0 when analytic, the oracle's cluster SE otherwise."""
        if self.engine is not None:
            return {c: {r: 0.0 for r in e.regs} for c, e in self.engine.full.items()}
        _, prep = self.oracle
        return {c: dict(zip(f.regs, f.se.tolist(), strict=True)) for c, f in prep.fit().items()}

    def full(self) -> dict[str, dict[str, float]]:
        """T3 of the full-sample fit, per equation (in the analysis' reporting units)."""
        if self.engine is not None:
            return {c: dict(zip(e.regs, e.coef.tolist(), strict=True)) for c, e in self.engine.full.items()}
        _, prep = self.oracle
        return {c: dict(zip(f.regs, f.coef.tolist(), strict=True)) for c, f in prep.fit(cluster=False).items()}


# ---------------------------------------------------------------------------------------------
# Matching and classification
# ---------------------------------------------------------------------------------------------


@dataclass
class Planted:
    label: str
    when: dict[str, str]
    arrow: str
    delta: float
    path: str | None  # the Researcher's Model path the twist should show up on
    true_diff: float  # T3 common-metric difference on that path (subgroup - complement)
    mask: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0, bool))


@dataclass
class Grade:
    hits: list[dict]
    misses: list[dict]
    redundant: list[dict]
    false_positives: list[dict]
    n_findings: int
    candidates: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"hits": self.hits, "misses": self.misses, "redundant": self.redundant,
                "false_positives": self.false_positives, "n_findings": self.n_findings,
                "candidates": self.candidates,
                "fp_by_class": {c: sum(1 for f in self.false_positives if f["class"] == c) for c in FP_CLASSES}}


def planted_twists(world: World, model: ResearcherModel, tp: TruthProvider, frame_df: pd.DataFrame,
                   baseline: str) -> list[Planted]:
    out = []
    eq_paths = {f"{r} -> {c}" for c, rs in model.equations().items() for r in rs}
    for i, t in enumerate(world.twists):
        desc = Description(tuple(_cond(a, v) for a, v in t.when.items()))
        con = tp.contrast(desc, "complement")
        common = con.get("common", {})
        key = f"{t.parent} -> {t.child}"
        # a twist the model lacks lands in its child's equation (other twists can move paths elsewhere)
        into_child = {k: v for k, v in common.items() if k.endswith(f"-> {t.child}")}
        if key in eq_paths:
            path = key
        elif into_child or common:
            pool = into_child or common
            path = max(pool, key=lambda k: abs(pool[k]))
        else:
            path = None
        true = common.get(path, 0.0) if path else 0.0
        if abs(true) < 1e-9:
            true = t.delta
        out.append(Planted(t.label or f"twist{i}", dict(t.when), key, t.delta, path, float(true),
                           desc.mask(frame_df)))
    return out


def _cond(a: str, v: str):
    from .emm import Condition
    return Condition(a, "==", v)


def grade(world: World, model: ResearcherModel, result: EMMResult, frame_df: pd.DataFrame, tp: TruthProvider,
          grading: GradingConfig, baseline: str, threshold: float | None = None, *,
          quality: str | None = None, z_crit: float = 1.959963984540054) -> Grade:
    """Match planted twists to findings (overlap >= tau, right sign on the twisted path) and classify the rest.

    With ``quality`` each hit also says whether the twisted path itself deviates under that quality function's rule
    (``path_deviates``) and the best-ranked matching finding where it does (``recovered_rank`` and
    ``recovered_quality``, None if none).

    ``candidates`` lists every (twist, finding) pair with overlap >= min(tau, proxy_min_jaccard), with its overlap,
    sign, effect and whether the path deviates, so another tau or recovery rule can be applied afterwards.
    """
    planted = planted_twists(world, model, tp, frame_df, baseline)
    findings = result.findings
    decoys = {n for n, a in world.attributes.items() if a.role == "decoy"} | \
             {n for n, v in world.nodes.items() if v.kind == "attribute" and v.role == "decoy"}
    g = nx.DiGraph()
    g.add_edges_from((a.parent, a.child) for a in world.arrows)
    hits, misses, redundant, logged = [], [], [], []
    used: set[int] = set()
    floor = min(grading.tau, grading.proxy_min_jaccard)
    for p in planted:
        cands = []
        for f in findings:
            j_direct = jaccard(f.mask, p.mask)
            j_comp = jaccard(~f.mask, p.mask) if baseline == "complement" else 0.0
            orient = 1.0 if j_direct >= j_comp else -1.0
            j = max(j_direct, j_comp)
            if j < floor:
                continue
            est = _path_diff(f, p.path)
            sign_ok = est is not None and np.sign(orient * est) == np.sign(p.true_diff)
            logged.append({"twist": p.label, "path": p.path, "description": str(f.description), "rank": f.rank,
                           "quality": f.quality, "n": int(f.mask.sum()), "jaccard": j, "via_complement": orient < 0,
                           "est_diff": None if est is None else orient * est, "true_diff": p.true_diff,
                           "sign_ok": bool(sign_ok),
                           "path_deviates": None if quality is None else _path_deviates(f, p.path, quality, z_crit)})
            if j < grading.tau:
                continue
            cands.append((f, j, orient, est, sign_ok))
        good = [c for c in cands if c[4]]
        if good:
            best = min(good, key=lambda c: c[0].rank)
            f, j, orient, est, _ = best
            dev = None if quality is None else [c for c in good if _path_deviates(c[0], p.path, quality, z_crit)]
            rec = min(dev, key=lambda c: c[0].rank)[0] if dev else None
            hits.append({"twist": p.label, "path": p.path, "rank": f.rank, "jaccard": j, "n_g": int(p.mask.sum()),
                         "est_diff": orient * est, "true_diff": p.true_diff, "quality": f.quality,
                         "significant": threshold is None or f.quality >= threshold,
                         "description": str(f.description),
                         "path_deviates": None if dev is None else any(c[0] is f for c in dev),
                         "recovered_rank": None if rec is None else rec.rank,
                         "recovered_quality": None if rec is None else rec.quality})
            used.add(id(f))
            for c in good:
                if c[0] is not f:
                    redundant.append({"twist": p.label, "description": str(c[0].description), "rank": c[0].rank})
                    used.add(id(c[0]))
        else:
            misses.append({"twist": p.label, "path": p.path, "true_diff": p.true_diff, "n_g": int(p.mask.sum()),
                           "near": [{"description": str(c[0].description), "jaccard": c[1]} for c in cands]})
    fps = []
    for f in findings:
        if id(f) in used:
            continue
        fps.append({"description": str(f.description), "rank": f.rank, "quality": f.quality,
                    "class": classify(f, planted, decoys, g, tp, grading, baseline),
                    "significant": threshold is None or f.quality >= threshold})
    return Grade(hits, misses, redundant, fps, len(findings), logged)


def _path_deviates(f: Finding, path: str | None, quality: str, z_crit: float) -> bool:
    return any(p.path == path and p.deviates(quality, z_crit) for p in f.evaluation.paths)


def _path_diff(f: Finding, path: str | None) -> float | None:
    for p in f.evaluation.paths:
        if p.path == path:
            return p.diff
    return None


def classify(f: Finding, planted: list[Planted], decoys: set[str], g: nx.DiGraph, tp: TruthProvider,
             grading: GradingConfig, baseline: str) -> str:
    # 1. matches a planted subgroup but on the wrong path / sign: a downstream echo of the twist
    for p in planted:
        j = max(jaccard(f.mask, p.mask), jaccard(~f.mask, p.mask) if baseline == "complement" else 0.0)
        if j >= grading.tau:
            return "phantom_downstream"
    # 2. uses a decoy attribute
    if f.description.attrs() & decoys:
        return "decoy"
    # 3. overlaps a planted subgroup through a correlated attribute
    for p in planted:
        j = max(jaccard(f.mask, p.mask), jaccard(~f.mask, p.mask) if baseline == "complement" else 0.0)
        if j >= grading.proxy_min_jaccard:
            return "proxy"
    # 4./5. use the T3 contrast on the finding's best path
    best = f.evaluation.best
    con = tp.contrast(f.description, baseline)
    if con.get("empty") or best is None:
        return "null"
    common = con["common"].get(best.path, 0.0)
    within = con["within"].get(best.path, 0.0)
    if abs(common) >= grading.epsilon:
        # a difference that vanishes on true scores is measurement driven: error variance, or the observation scale
        true_c = tp.true_score_contrast(f.description, baseline)
        if true_c is not None and abs(true_c.get(best.path, 0.0)) < grading.epsilon:
            return "scale_artifact" if has_scales(tp.cw.world) else "variance_artifact"
        return "level_shift_leak"
    if abs(within) >= grading.epsilon:
        return "variance_artifact"
    return "null"


def estimation_block(prep_full: dict, tp: TruthProvider, level: float = 0.95) -> list[dict]:
    """Full-sample path estimates vs T3: error, SE, CI coverage indicator (per Run)."""
    from scipy import stats
    z = stats.norm.ppf(0.5 + level / 2)
    t3 = tp.full()
    rows = []
    for c, f in prep_full.items():
        for i, r in enumerate(f.regs):
            truth = t3[c][r]
            b, se = float(f.coef[i]), float(f.se[i])
            rows.append({"path": f"{r} -> {c}", "estimate": b, "se": se, "se_naive": float(f.se_naive[i]),
                         "t3": truth, "error": b - truth, "covered": abs(b - truth) <= z * se,
                         "covered_naive": abs(b - truth) <= z * float(f.se_naive[i])})
    return rows


def world_key(world: World) -> str:
    return spec_hash(world)


def safe(x: float) -> float | None:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else x
