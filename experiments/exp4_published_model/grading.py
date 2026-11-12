"""Grading one search on one fake Greece: was the planted subgroup found again, and what else was reported?

The rules follow Experiments 1-4 (``emmlab.grade``), adapted to a real-data World with one planted subgroup:

- the **match** is the reported subgroup that overlaps the planted one most (Jaccard). A subgroup whose own complement
  is the planted one counts as the planted one seen from the other side (the real run itself reports the repeater
  pair ``yes`` / ``no``), with its direction reversed. The mirror overlap is Jaccard(not found, planted), not
  Jaccard(found, not planted): the latter is near 1 for any large subgroup when the planted one is small (``repeat=no``,
  4,446 students, against the complement of a 55-student group), which a first grading counted as recoveries;
- **recovered**: match Jaccard >= ``TAU`` (0.8), the path deviates (path search: the two 95% intervals are disjoint;
  all-paths search: at least one path), and, for the path search, the difference points the planted way;
- every reported subgroup is classed ``planted`` (Jaccard >= TAU, directly or as the mirror), ``overlap``
  (0.3 <= direct Jaccard < TAU: a refinement or a coarsening of it) or ``unrelated``. In M0 everything is unrelated
  by construction (nothing is planted).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from search import Finding

TAU = 0.8
OVERLAP = 0.3


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / union) if union else 0.0


def overlaps(cover: np.ndarray, planted: np.ndarray) -> tuple[float, float]:
    """(Jaccard with the planted subgroup, Jaccard of the found subgroup's complement with it)."""
    return jaccard(cover, planted), jaccard(~cover, planted)


def classify(findings: list[Finding], planted: np.ndarray | None) -> list[dict]:
    """One row per reported subgroup: its overlap with the planted subgroup (or its complement) and its class."""
    rows = []
    for f in findings:
        j, jc = overlaps(f.cover, planted) if planted is not None else (0.0, 0.0)
        # the mirror only decides "is the planted subgroup"; for a large planted subgroup the complement of almost any
        # small subgroup overlaps it by more than OVERLAP, so "overlap" uses the direct overlap only
        if planted is None:
            cls = "unrelated"
        elif max(j, jc) >= TAU:
            cls = "planted"
        else:
            cls = "overlap" if j >= OVERLAP else "unrelated"
        rows.append({"rank": f.rank, "description": f.description, "size": f.size, "quality": f.quality,
                     "deviating": f.deviating, "estimate_sg": f.estimate_sg,
                     "estimate_complement": f.estimate_complement,
                     "jaccard": j, "jaccard_complement": jc, "mirror": jc > j, "class": cls})
    return rows


def grade(findings: list[Finding], planted: np.ndarray | None, direction: float, quality: str) -> dict:
    """Summary of one search: top score, counts per class, and (M1) whether the planted subgroup was recovered.

    ``direction`` is the sign of (subgroup - rest) on the searched path in the planted World (path search only)."""
    rows = classify(findings, planted)
    out = {"n_findings": len(rows), "top_quality": rows[0]["quality"] if rows else 0.0,
           "top_description": rows[0]["description"] if rows else "",
           "n_unrelated": sum(r["class"] == "unrelated" for r in rows),
           "n_overlap": sum(r["class"] == "overlap" for r in rows)}
    if planted is None:
        return out
    match = max(rows, key=lambda r: max(r["jaccard"], r["jaccard_complement"]), default=None)
    out.update({"match_rank": np.nan, "match_jaccard": 0.0, "match_quality": np.nan, "match_mirror": False,
                "match_estimate_sg": np.nan, "match_estimate_rest": np.nan, "recovered": False, "top_is_planted": False,
                "n_planted_listed": sum(r["class"] == "planted" for r in rows)})
    if match is None:
        return out
    jm = max(match["jaccard"], match["jaccard_complement"])
    sg, rest = match["estimate_sg"], match["estimate_complement"]
    if match["mirror"]:
        sg, rest = rest, sg
    right_way = True if quality != "path" else bool(np.sign(sg - rest) == np.sign(direction))
    out.update({"match_rank": match["rank"], "match_jaccard": jm, "match_quality": match["quality"],
                "match_mirror": bool(match["mirror"]), "match_estimate_sg": sg, "match_estimate_rest": rest,
                "recovered": bool(jm >= TAU and match["deviating"] >= 1 and right_way),
                "top_is_planted": rows[0]["class"] == "planted"})
    return out


def real_finding_hits(findings: list[Finding], real_masks: dict[str, np.ndarray]) -> dict:
    """For control runs: is each of the real run's subgroups (or its mirror) in the reported list, and at which rank?"""
    out = {}
    for name, m in real_masks.items():
        ranks = [f.rank for f in findings if max(overlaps(f.cover, m)) >= TAU]
        out[f"listed_{name}"] = bool(ranks)
        out[f"rank_{name}"] = min(ranks) if ranks else np.nan
    return out


def ranked_qualities(findings: list[Finding], k: int = 6) -> dict:
    """Quality at ranks 1..k (0 when fewer subgroups were reported), to compare rank by rank with the real run."""
    q = [f.quality for f in findings] + [0.0] * k
    return {f"q{i + 1}": q[i] for i in range(k)}


def findings_frame(findings: list[Finding], planted: np.ndarray | None) -> pd.DataFrame:
    return pd.DataFrame(classify(findings, planted))
