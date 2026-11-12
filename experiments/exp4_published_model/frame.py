"""The real Greek frame of Exp 4: PISA 2015 Greece with the least preprocessing PLS needs.

Built from the raw files (``Hyp-Gen-via-EMM/data_paper2/cy6_ms_cmb_stu_qqq.sas7bdat`` and the school file), not
from the earlier pipeline's CSVs:

1. all 5,532 Greek students; the 32 construct items recoded exactly as the earlier build
   (``build_pisa_grc_dataset.recode_construct_items``: PISA special codes to missing, reverse-coding);
2. one **scale score per construct** = the mean of the items the student answered (the paper also used one
   summed score per construct). A student needs at least one item of every construct: 4,996 remain;
3. gender (``gender_num``, 1 = female) and achievement (PV1SCIE) are never missing;
4. the ten searchable descriptors of the original search, binned as the earlier builds bin them (ESCS quartiles on all Greek
   students; school tertiles and the school-average-feedback tertile on the 4,996);
5. students missing any descriptor are dropped (Cristian's decision, 2026-11-01): **4,564 students**.

No imputation of items, no school-mean centring, no other normalisation (PLS standardises inside each fit).
The model columns are named ``<construct>_score`` so ``emm_utils.build_pisa_config`` builds the single-indicator
model on them unchanged (it matches columns by the prefix ``<construct>_``).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_root = os.environ.get("HYPGEN_ROOT")
if not _root:
    raise RuntimeError("Set HYPGEN_ROOT to a checkout of the earlier pipeline (Hyp-Gen-via-EMM); Experiment 4 reads its code and the raw PISA 2015 files.")
HYPGEN = Path(_root)
if str(HYPGEN / "scratch") not in sys.path:
    sys.path.insert(0, str(HYPGEN / "scratch"))

import build_pisa_grc_dataset as B  # noqa: E402

STUDENT_FILE = HYPGEN / "data_paper2" / "cy6_ms_cmb_stu_qqq.sas7bdat"
SCHOOL_FILE = HYPGEN / "scratch" / "data" / "CY6_MS_CMB_SCH_QQQ.sav"
CONSTRUCTS = ("feedback", "selfeff", "interest", "enjoy", "perfgoal", "extrinsic")
SCORE = {c: f"{c}_score" for c in CONSTRUCTS}
DESCRIPTORS = ["gender", "school_type", "escs_quartile", "immigrant_background", "grade_repetition",
               "school_fb_tertile", "school_urbanicity", "school_size_tertile", "school_stratio_tertile",
               "school_staffshort_tertile"]
MODEL_COLUMNS = [SCORE[c] for c in CONSTRUCTS] + ["gender_num", "achievement"]


@dataclass(frozen=True)
class FrameCounts:
    students_raw: int
    schools_raw: int
    with_every_construct: int
    final: int
    schools_final: int
    dropped_for_descriptors: int


def load_real_frame() -> tuple[pd.DataFrame, FrameCounts]:
    """The 4,564-student frame (see the module docstring) and the counts at each step."""
    import pyreadstat

    stu, _ = pyreadstat.read_sas7bdat(str(STUDENT_FILE), usecols=B.USECOLS)
    stu = stu[stu["CNT"] == "GRC"].reset_index(drop=True)
    sch, _ = pyreadstat.read_sav(str(SCHOOL_FILE), usecols=["CNT", "CNTSCHID"] + B.SCHOOL_DESCRIPTOR_COLS)
    sch = sch[sch["CNT"] == "GRC"].drop(columns="CNT")

    items = B.recode_construct_items(stu)
    df = pd.DataFrame({"student_id": stu["CNTSTUID"], "school_id": stu["CNTSCHID"]})
    for c, (src, _, _) in B.CONSTRUCT_BLOCKS.items():
        cols = [f"{c}_{i}" for i in range(1, len(src) + 1)]
        df[SCORE[c]] = items[cols].mean(axis=1)  # NaN when no item of the construct was answered
    df["gender_num"] = (stu["ST004D01T"] == 1).astype(float)
    df["achievement"] = stu["PV1SCIE"]
    df["gender"] = stu["ST004D01T"].map({1.0: "female", 2.0: "male"})
    df["immigrant_background"] = stu["IMMIG"].map({1.0: "native", 2.0: "second_generation", 3.0: "first_generation"})
    df["grade_repetition"] = stu["REPEAT"].map({0.0: "no", 1.0: "yes"})
    df["escs_quartile"] = pd.qcut(stu["ESCS"], 4, labels=["Q1_lowest", "Q2", "Q3", "Q4_highest"]).astype(object)

    s = sch.assign(
        school_type=sch["SCHLTYPE"].map({1.0: "private_independent", 2.0: "private_government_dependent",
                                          3.0: "public"}),
        school_urbanicity=sch["SC001Q01TA"].map({1.0: "village_or_rural", 2.0: "small_town", 3.0: "town",
                                                 4.0: "city", 5.0: "large_city"}),
        school_size=sch["SCHSIZE"], school_stratio=sch["STRATIO"], school_staffshort=sch["STAFFSHORT"])
    df = df.merge(s[["CNTSCHID", "school_type", "school_urbanicity", "school_size", "school_stratio",
                     "school_staffshort"]], left_on="school_id", right_on="CNTSCHID", how="left")
    df = df.drop(columns="CNTSCHID")

    n_raw, sch_raw = len(df), int(df["school_id"].nunique())
    df = df[df[[SCORE[c] for c in CONSTRUCTS]].notna().all(axis=1)].reset_index(drop=True)
    n_every = len(df)
    # binning on the 4,996, as the earlier builds bin on their analysed sample
    _, fb_tert = B.school_feedback_tertile_labels(df, "school_id", [SCORE["feedback"]])
    df["school_fb_tertile"] = df["school_id"].map(fb_tert).astype(object)
    for raw, tert in [("school_size", "school_size_tertile"), ("school_stratio", "school_stratio_tertile"),
                      ("school_staffshort", "school_staffshort_tertile")]:
        df[tert] = B.tertile_on_sample(df[raw])
    complete = df[DESCRIPTORS].notna().all(axis=1)
    df = df[complete].reset_index(drop=True)
    for col in DESCRIPTORS:
        df[col] = df[col].astype(str)
    counts = FrameCounts(n_raw, sch_raw, n_every, len(df), int(df["school_id"].nunique()), int((~complete).sum()))
    return df, counts


def prepared(df: pd.DataFrame):
    """The frame the search sees (descriptors + model columns) and its single-indicator plspm config."""
    import emm_utils as eu

    frame = df[["school_id", *DESCRIPTORS, *MODEL_COLUMNS]].reset_index(drop=True)
    return frame, eu.build_pisa_config(frame)


if __name__ == "__main__":
    f, n = load_real_frame()
    print(n)
    print(f[MODEL_COLUMNS].describe().round(3).T)
    print({c: f[c].value_counts().to_dict() for c in DESCRIPTORS})
