"""The recipe of Exp 4: the paper's Figure 2 as a data generator (the supervisor's loop).

"We trust you": every construct is made from the ones before it with the 17 published standardised coefficients
(Katsantonis, McLellan & Torres 2023, Figure 2, within-school level), plus a normal error. The error's SD is the
part of the construct the arrows leave unexplained, sqrt(1 - R^2), so every construct has variance 1 as the
standardised coefficients assume:

    construct_j = sum_k b_jk * parent_k + e_j,   e_j ~ Normal(0, 1 - R^2_j)

Gender is the real student's gender (standardised), so a fake student keeps gender, school and every descriptor of
a real one; in the recipe no descriptor other than gender has any effect.

A **planted** recipe changes one arrow for one group: inside the planted subgroup feedback -> achievement is
``inside``, outside it ``outside``. Everything else, including the error SDs, is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from frame import SCORE

FIGURE2: dict[tuple[str, str], float] = {
    ("Gender", "feedback"): -0.165, ("Gender", "Achievement"): -0.058,
    ("feedback", "selfeff"): 0.094, ("feedback", "interest"): 0.156, ("feedback", "enjoy"): 0.164,
    ("feedback", "extrinsic"): 0.121, ("feedback", "perfgoal"): 0.030, ("feedback", "Achievement"): -0.135,
    ("selfeff", "interest"): 0.282, ("selfeff", "enjoy"): 0.276, ("selfeff", "perfgoal"): 0.191,
    ("selfeff", "extrinsic"): 0.303, ("selfeff", "Achievement"): 0.119,
    ("interest", "Achievement"): 0.180, ("enjoy", "Achievement"): 0.193, ("perfgoal", "Achievement"): 0.079,
    ("extrinsic", "Achievement"): -0.061,
}
ORDER = ("Gender", "feedback", "selfeff", "interest", "enjoy", "perfgoal", "extrinsic", "Achievement")
COLUMN = {**SCORE, "Gender": "gender_num", "Achievement": "achievement"}


def implied(coefs: dict[tuple[str, str], float]) -> tuple[np.ndarray, dict[str, float]]:
    """The covariance matrix the standardised recipe implies and each endogenous construct's R^2."""
    k = len(ORDER)
    pos = {v: i for i, v in enumerate(ORDER)}
    B = np.zeros((k, k))
    for (f, t), b in coefs.items():
        B[pos[t], pos[f]] = b
    S = np.zeros((k, k))
    S[0, 0] = 1.0
    r2 = {}
    for j in range(1, k):
        b = B[j, :j]
        r2[ORDER[j]] = float(b @ S[:j, :j] @ b)
        S[j, :j] = S[:j, :j] @ b
        S[:j, j] = S[j, :j]
        S[j, j] = 1.0
    return S, r2


@dataclass(frozen=True)
class Planted:
    """One subgroup with its own feedback -> achievement slope."""

    description: str
    mask: np.ndarray
    inside: float
    outside: float


@dataclass
class Recipe:
    coefs: dict[tuple[str, str], float]
    planted: Planted | None = None

    def __post_init__(self) -> None:
        _, r2 = implied(self.coefs)
        if any(v >= 1 for v in r2.values()):
            raise ValueError("the coefficients imply R^2 >= 1")
        self.error_sd = {t: float(np.sqrt(1.0 - v)) for t, v in r2.items()}

    def simulate(self, template: pd.DataFrame, seed: int | np.random.SeedSequence) -> pd.DataFrame:
        """A fake Greece: the template's students (school, descriptors, gender), new construct scores."""
        rng = np.random.default_rng(seed)
        n = len(template)
        g = template["gender_num"].to_numpy(float)
        score = {"Gender": (g - g.mean()) / g.std()}
        for t in ORDER[1:]:
            v = rng.normal(0.0, self.error_sd[t], n)
            for (f, to), b in self.coefs.items():
                if to != t:
                    continue
                if self.planted is not None and (f, to) == ("feedback", "Achievement"):
                    b = np.where(self.planted.mask, self.planted.inside, self.planted.outside)
                v = v + b * score[f]
            score[t] = v
        out = template.copy()
        for t in ORDER[1:]:
            out[COLUMN[t]] = score[t]
        return out
