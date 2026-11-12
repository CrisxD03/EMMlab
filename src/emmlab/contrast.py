"""Contrastive analysis of a found conjunction: which part of ``A AND B`` carries the deviation?

EMM compares a subgroup with its complement, so for ``A AND B`` it cannot say whether the deviation needs
both conditions, only one of them, or is chance. The post-discovery step (not a second search):

1. Fit the Researcher's Model on the finding and on the slices that differ from it by *flipping* conditions:
   ``A AND NOT B``, ``NOT A AND B`` (and, with more conditions, every other sign pattern).
2. Compare the finding with each slice path by path. "Deviates" is judged by the quality function's own
   rule (``PathStat.deviates``): non-overlapping intervals for ``current``, |z| >= 1.96 otherwise.
3. A condition is *needed* when flipping it makes the deviation disappear. All needed: conjunction-specific.
   Some needed: carried by those. None needed: shared (either condition reproduces it, or the slices are
   too small to tell, which is reported as ``low_power``).
4. Each condition on its own against everyone else is reported too (what a depth-1 search would have said).

Everything is fitted with the settings of the run (``PathQuality``), so the numbers are the ones the search used.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np
import pandas as pd

from .emm import Description, PathQuality, PathStat
from .run import effective_schools
from .stages import AnyPrepared

MIN_FIT_N = 20  # below this a slice's fit is flagged unreliable (the researcher's pipeline needs >= 20)


def _num(x: float | None) -> float | None:
    return float(x) if x is not None and math.isfinite(x) else None


def verdict_for(conditions: list[str], flipped_deviates: list[bool], *, low_power: bool) -> dict[str, Any]:
    """Read the verdict off which single-condition flips make the deviation disappear."""
    k = len(conditions)
    if k <= 1:
        return {"kind": "single", "needed": [], "low_power": low_power,
                "text": "A single condition: there is nothing to decompose."}
    needed = [c for c, dev in zip(conditions, flipped_deviates, strict=True) if dev]
    others = [c for c in conditions if c not in needed]
    if len(needed) == k:
        kind = "conjunction_specific"
        text = "Only the combination shows it: dropping any single condition removes the deviation."
    elif needed:
        kind = "carried_by"
        text = (f"Carried by {' and '.join(needed)}: dropping {' or '.join(others)} leaves the deviation, "
                f"dropping {' or '.join(needed)} removes it.")
    else:
        kind = "shared"
        text = ("Not isolated: dropping any single condition leaves the deviation, so either condition "
                "reproduces it (or the slices are too small to tell).")
    return {"kind": kind, "needed": needed, "low_power": low_power, "text": text}


def alone_note(kind: str, needed: list[str], alone_deviates: dict[str, bool]) -> str | None:
    """Caution when the conditions a verdict relies on do not deviate on their own (like chance in a small group)."""
    if kind not in ("carried_by", "conjunction_specific") or not needed:
        return None
    if any(alone_deviates.get(c, False) for c in needed):
        return None
    who = " or ".join(needed)
    return (f"But {who} does not deviate on its own against everyone else. A deviation that shows up only in a "
            "small combination and in none of its parts is also what chance looks like: check it on a larger sample "
            "or another seed.")


def _path_cmp(ps: PathStat, quality: str, z_crit: float) -> dict[str, Any]:
    finite = all(math.isfinite(v) for v in (ps.b_g, ps.se_g, ps.b_b, ps.se_b))
    if not finite:
        return {"diff": None, "z": None, "disjoint": False, "deviating": False}
    return {"diff": float(ps.diff), "z": float(ps.z), "disjoint": ps.disjoint,
            "deviating": ps.deviates(quality, z_crit)}


def _summary(pq: PathQuality, prep: AnyPrepared, mask: np.ndarray) -> dict[str, Any]:
    fits_raw = pq.fit_slice(mask)
    fits: dict[str, Any] | None = {}
    z = float(pq.z)
    for path, (b, se) in fits_raw.items():
        fits[path] = ({"b": b, "se": se, "ci": [b - z * se, b + z * se]}
                      if math.isfinite(b) and math.isfinite(se) else None)
    n = int(mask.sum())
    flags = effective_schools(prep.codes, mask, prep.n_groups)
    reliable = n >= MIN_FIT_N and all(v is not None for v in fits.values())
    return {"n": n, "schools": flags["schools"], "g_eff": flags["g_eff"], "tier": flags["tier"],
            "reliable": bool(reliable), "fits": fits}


def contrast_finding(frame: pd.DataFrame, prep: AnyPrepared, description: Description,
                     pq: PathQuality) -> dict[str, Any]:
    """Contrast one finding with its flipped slices, every sign pattern, its complement and its single conditions."""
    conds = list(description.conditions)
    k = len(conds)
    labels = [str(c) for c in conds]
    cmask = [c.mask(frame) for c in conds]
    q, z_crit = pq.emm.quality, float(pq.z)

    def cell(pattern: tuple[bool, ...]) -> np.ndarray:
        m = np.ones(len(frame), dtype=bool)
        for holds, cm in zip(pattern, cmask, strict=True):
            m &= cm if holds else ~cm
        return m

    def label(pattern: tuple[bool, ...]) -> str:
        return " AND ".join(lab if holds else f"NOT {lab}" for lab, holds in zip(labels, pattern, strict=True))

    all_true = (True,) * k
    finding = cell(all_true)
    specs: list[tuple[str, str, str, np.ndarray, tuple[bool, ...] | None]] = [
        ("finding", "finding", label(all_true), finding, all_true)]
    flips = [tuple(j != i for j in range(k)) for i in range(k)]
    if k >= 2:
        for i, pat in enumerate(flips):
            specs.append((f"flip:{i}", "flip", label(pat), cell(pat), pat))
        others = sorted((p for p in itertools.product((True, False), repeat=k) if sum(not b for b in p) >= 2),
                        key=lambda p: (sum(not b for b in p), [int(b) for b in p]))
        for pat in others:
            specs.append(("cell:" + "".join("1" if b else "0" for b in pat), "cell", label(pat), cell(pat), pat))
    specs.append(("complement", "complement", f"NOT ({label(all_true)})", ~finding, None))
    if pq.emm.baseline == "population":
        specs.append(("population", "population", "all students", np.ones(len(frame), dtype=bool), None))

    baseline_mask = ~finding if pq.emm.baseline == "complement" else np.ones(len(frame), dtype=bool)
    focal_stats = pq.compare(finding, baseline_mask)
    paths = [ps.path for ps in focal_stats]
    focal = [ps.path for ps in focal_stats if _path_cmp(ps, q, z_crit)["deviating"]]

    slices: list[dict[str, Any]] = []
    for key, role, lab, mask, pattern in specs:
        item: dict[str, Any] = {"key": key, "role": role, "label": lab,
                                "pattern": list(pattern) if pattern is not None else None,
                                **_summary(pq, prep, mask)}
        if key == "finding":
            item["vs_finding"] = None
        else:
            cmp_ = pq.compare(finding, mask)
            item["vs_finding"] = {ps.path: _path_cmp(ps, q, z_crit) for ps in cmp_}
        slices.append(item)
    by_key = {s["key"]: s for s in slices}

    # every pair of grid cells (rows, columns, diagonals), and for two conditions the interaction
    masks = {key: mask for key, role, _, mask, _ in specs if role in ("finding", "flip", "cell")}
    pairs = []
    if k >= 2:
        for a, b in itertools.combinations(list(masks), 2):
            pairs.append({"a": a, "b": b,
                          "vs": {ps.path: _path_cmp(ps, q, z_crit) for ps in pq.compare(masks[a], masks[b])}})
    interaction: dict[str, Any] = {}
    if k == 2:
        for path in paths:
            f = [by_key[key]["fits"].get(path) for key in ("finding", "flip:1", "flip:0", "cell:00")]
            if any(x is None for x in f):
                continue
            est = (f[0]["b"] - f[1]["b"]) - (f[2]["b"] - f[3]["b"])
            se = math.sqrt(sum(x["se"] ** 2 for x in f))
            z = est / se if se > 0 else 0.0
            interaction[path] = {"estimate": est, "se": se, "z": z, "ci": [est - z_crit * se, est + z_crit * se],
                                 "additive": bool(abs(z) < z_crit)}

    singles = []
    if k >= 2:
        for lab, cm in zip(labels, cmask, strict=True):
            cmp_ = pq.compare(cm, ~cm)
            singles.append({"condition": lab, "alone": _summary(pq, prep, cm), "rest": _summary(pq, prep, ~cm),
                            "vs_rest": {ps.path: _path_cmp(ps, q, z_crit) for ps in cmp_}})

    low_power = any((not by_key[f"flip:{i}"]["reliable"]) or by_key[f"flip:{i}"]["tier"] != "ok" for i in range(k)
                    if k >= 2)
    verdicts = {}
    for path in focal:
        flipped = [bool(by_key[f"flip:{i}"]["vs_finding"][path]["deviating"]) for i in range(k)] if k >= 2 else []
        v = verdict_for(labels, flipped, low_power=low_power)
        alone = {lab: bool(sg["vs_rest"][path]["deviating"]) for lab, sg in zip(labels, singles, strict=False)}
        v["alone_deviates"] = alone
        v["note"] = alone_note(v["kind"], v["needed"], alone)
        verdicts[path] = v

    return {
        "description": str(description), "conditions": labels, "quality": q, "baseline": pq.emm.baseline,
        "interval_level": float(pq.emm.interval_level), "interval_se": pq.emm.interval_se,
        "interval_method": "wald",  # also under current_exact: bootstrap intervals exist only for subgroup vs baseline
        "paths": paths, "focal_paths": focal, "slices": slices, "singles": singles, "verdicts": verdicts,
        "pairs": pairs, "interaction": interaction,
    }
