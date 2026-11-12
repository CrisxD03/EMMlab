"""Tests for Exp 4 (pinned environment: ``source experiments/exp4_published_model/env.sh; exp4py -m pytest -p no:cacheprovider
--rootdir experiments/exp4_published_model experiments/exp4_published_model/test_exp4.py``). Reading the raw PISA file takes about 15 s."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import search as S  # noqa: E402
from frame import DESCRIPTORS, MODEL_COLUMNS, load_real_frame, prepared  # noqa: E402
from pls_estimates import inner_estimates  # noqa: E402
from recipe import FIGURE2, Planted, Recipe, implied  # noqa: E402


@pytest.fixture(scope="module")
def real():
    df, counts = load_real_frame()
    frame, cfg = prepared(df)
    return df, counts, frame, cfg


def test_frame_counts(real):
    df, counts, _, _ = real
    assert (counts.students_raw, counts.schools_raw) == (5532, 211)
    assert (counts.with_every_construct, counts.final, counts.schools_final) == (4996, 4564, 190)
    assert not df[MODEL_COLUMNS + DESCRIPTORS].isna().any().any()


def test_same_students_as_the_earlier_imputed_frame(real):
    """Same eligibility rule (an answer in every construct, every descriptor known) as the original frame."""
    df, _, _, _ = real
    old = S.real_frame()
    assert set(df["student_id"]) == set(old["student_id"])


def test_no_centring(real):
    _, _, frame, _ = real
    school_means = frame.groupby("school_id")["achievement"].mean()
    assert school_means.std() > 10  # between-school differences are kept (achievement in PISA points)


def test_single_indicator_pls_is_standardised_ols(real):
    _, _, frame, cfg = real
    est = inner_estimates(frame, cfg).loc["feedback -> Achievement", "estimate"]
    z = (frame[MODEL_COLUMNS] - frame[MODEL_COLUMNS].mean()) / frame[MODEL_COLUMNS].std(ddof=0)
    X = np.column_stack([np.ones(len(z))] + [z[c] for c in ["gender_num", "feedback_score", "selfeff_score",
                                                             "interest_score", "enjoy_score", "perfgoal_score",
                                                             "extrinsic_score"]])
    beta = np.linalg.lstsq(X, z["achievement"], rcond=None)[0]
    assert est == pytest.approx(beta[2], abs=1e-6)


def test_recipe_variances_are_one():
    S_, r2 = implied(FIGURE2)
    assert np.allclose(np.diag(S_), 1.0) and all(0 < v < 1 for v in r2.values())


def test_simulation_keeps_students_and_is_reproducible(real):
    _, _, frame, _ = real
    rec = Recipe(FIGURE2)
    a, b, c = rec.simulate(frame, 3), rec.simulate(frame, 3), rec.simulate(frame, 4)
    assert a.equals(b) and not a.equals(c)
    keep = ["school_id", "gender_num", *DESCRIPTORS]
    assert a[keep].equals(frame[keep])


def test_planted_slope_only_changes_inside(real):
    _, _, frame, _ = real
    m = (frame["school_size_tertile"] == "medium").to_numpy()
    rec = Recipe(FIGURE2, Planted("x", m, 0.4, -0.4))
    fake, cfg = prepared(rec.simulate(frame, 1))
    inside = inner_estimates(fake, cfg, m).loc["feedback -> Achievement", "estimate"]
    outside = inner_estimates(fake, cfg, ~m).loc["feedback -> Achievement", "estimate"]
    assert inside > 0.25 and outside < -0.25  # planted 0.4 and -0.4; about 0.03 sampling SE at these sizes
    other = inner_estimates(fake, cfg, m).loc["selfeff -> Achievement", "estimate"]
    assert abs(other - FIGURE2[("selfeff", "Achievement")]) < 0.1  # other arrows are not planted
