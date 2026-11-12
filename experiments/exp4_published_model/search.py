"""The EMM search of the real Greek study, run unchanged on any frame (real or simulated).

This module is an adapter around the earlier pipeline in ``Hyp-Gen-via-EMM/scratch``: the same frame preparation
(``emm_utils.group_mean_center`` on every manifest column, ``build_pisa_config``), the same exhaustive DFS
(``emm_utils.run_emm_task``: depth 2, result set 10, min quality -inf) and the same quality functions:

- ``path``: ``custom_qf.SEMQFPathEntropyCIComplement`` on feedback -> Achievement (the finding tested);
- ``all``: ``SEM_model_target.SEMQFEntropyCIComplement`` (ln-entropy x number of the 17 paths with disjoint 95%
  intervals, subgroup vs complement), the unrestricted search of the original EMM-SEM paper.

Nothing here re-implements the search; Exp 4 judges the procedure that produced the finding, so it calls that code.
It needs the old pipeline's pinned environment (pandas 2.3.3, numpy 1.26.4, pysubgroup 0.9.0, plspm 0.5.7): plspm
0.5.7 fails under pandas 3.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

_root = os.environ.get("HYPGEN_ROOT")
if not _root:
    raise RuntimeError("Set HYPGEN_ROOT to a checkout of the earlier pipeline (Hyp-Gen-via-EMM); Experiment 4 reads its code and the raw PISA 2015 files.")
HYPGEN = Path(_root)
SCRATCH = HYPGEN / "scratch"
if str(SCRATCH) not in sys.path:
    sys.path.insert(0, str(SCRATCH))

import emm_utils as eu  # noqa: E402

eu.load_sem_extension(HYPGEN)
import custom_qf  # noqa: E402
import pysubgroup as ps  # noqa: E402

DESCRIPTORS: list[str] = list(eu.DESCRIPTIVE_COLS)
CONSTRUCTS = ("Gender", "feedback", "selfeff", "interest", "enjoy", "perfgoal", "extrinsic", "Achievement")
PATHS: list[tuple[str, str]] = list(eu.PUBLISHED_FIGURE2)  # the 17 arrows of Figure 2, in the paper's order


@dataclass(frozen=True)
class SearchSpec:
    """One EMM procedure: which quality function, on which path, how deep."""

    quality: str = "path"  # "path" (the original search) or "all" (every path counts)
    from_col: str = "feedback"
    to_col: str = "Achievement"
    z: float = 1.96
    depth: int = 2
    result_set_size: int = 10
    descriptors: tuple[str, ...] = field(default_factory=lambda: tuple(DESCRIPTORS))

    def make_qf(self, config):
        if self.quality == "path":
            return custom_qf.SEMQFPathEntropyCIComplement(config, self.z, self.from_col, self.to_col)
        if self.quality == "all":
            return ps.SEMQFEntropyCIComplement(config, self.z)
        raise ValueError(f"unknown quality {self.quality!r}")


@dataclass(frozen=True)
class Finding:
    """One subgroup in a search's result set (positive quality only, as reported in the original study)."""

    rank: int
    description: str
    size: int
    quality: float
    deviating: int  # path: 1 if the two intervals are disjoint; all: number of disjoint paths
    estimate_sg: float  # path search only (NaN for "all")
    estimate_complement: float
    cover: np.ndarray  # boolean mask over the frame's rows


def prepare(frame: pd.DataFrame) -> tuple[pd.DataFrame, object]:
    """The pipeline's preparation (``emm_utils.prepare_frame`` minus file reading and the descriptor filter):
    centre every manifest column on its school mean, then build the plspm config on the centred frame."""
    df = eu.group_mean_center(frame.reset_index(drop=True), "school_id", eu.all_manifest_columns(frame))
    return df, eu.build_pisa_config(df)


def prepare_simulated(frame: pd.DataFrame) -> tuple[pd.DataFrame, object]:
    """Preparation for a fake Greece: the generator is fitted on the centred real frame, so what it draws is already a
    centred frame; only the config is built. Centring it again would centre twice: each school's fake mean would mix
    the rest of the school into small subgroups (1-3 repeaters in a school) and pull their slopes towards the rest's
    (repeaters 0.096 planted -> 0.06 found on 20 stacked fake Greeces)."""
    df = frame.reset_index(drop=True)
    return df, eu.build_pisa_config(df)


def cover_of(description: str, frame: pd.DataFrame) -> np.ndarray:
    """The rows a result description (``a=='x' AND b=='y'``, as pysubgroup prints it) selects. Fake Greeces keep the
    real descriptors, so a description read from a results file gives the exact cover in any of them."""
    mask = np.ones(len(frame), dtype=bool)
    for cond in description.split(" AND "):
        col, val = cond.split("==")
        mask &= (frame[col.strip()] == val.strip().strip("'")).to_numpy()
    return mask


def real_frame() -> pd.DataFrame:
    """The real Greek frame before centring: imputed file, ``gender_num`` added, rows with a missing descriptor
    dropped (4,564 students). ``prepare(real_frame())`` equals ``emm_utils.prepare_frame()``."""
    df = eu.add_gender_num(pd.read_csv(SCRATCH / "data" / "pisa_grc_2015_imputed.csv"))
    return df.loc[~df[DESCRIPTORS].isna().any(axis=1)].reset_index(drop=True)


def run_search(df: pd.DataFrame, config, spec: SearchSpec) -> list[Finding]:
    """Run the exhaustive search on a prepared frame; return the positive-quality result set, best first."""
    qf = spec.make_qf(config)
    result, _, _ = eu.run_emm_task(df[list(spec.descriptors) + _model_columns(df)], config, qf, list(spec.descriptors),
                                   result_set_size=spec.result_set_size, depth=spec.depth)
    rows = [r for r in _results(result) if r[0] > 0]
    rows.sort(key=lambda r: -r[0])
    out = []
    for i, (q, sg, stats) in enumerate(rows, start=1):
        cover = np.asarray(sg.covers(df), dtype=bool)
        if spec.quality == "path":
            dev, e_sg, e_c = int(stats["disjoint_ci"]), float(stats["estimate_sg"]), float(stats["estimate_complement"])
        else:
            # SEMQFEntropyCIComplement stores its disjoint-interval count in the field named changes_sig_sg
            dev, e_sg, e_c = int(stats["changes_sig_sg"]), float("nan"), float("nan")
        out.append(Finding(i, str(sg), int(cover.sum()), float(q), dev, e_sg, e_c, cover))
    return out


def _model_columns(df: pd.DataFrame) -> list[str]:
    return eu.all_manifest_columns(df)


def _results(result):
    """(quality, subgroup, statistics dict) for every entry of a pysubgroup result."""
    for q, sg, stats in result.results:
        yield q, sg, (stats._asdict() if hasattr(stats, "_asdict") else dict(stats))
