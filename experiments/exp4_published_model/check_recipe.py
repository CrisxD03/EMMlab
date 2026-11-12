"""Gate for Exp 4: the recipe reproduces Figure 2, and PLS reads it back without bias at large n.

    source experiments/exp4_published_model/env.sh; exp4py experiments/exp4_published_model/check_recipe.py

1. The implied variance of every construct is 1 (the error SDs are right).
2. On 40 stacked fake Greeces (182,560 students) the single-indicator PLS fit returns Figure 2's 17 numbers.
3. A planted recipe returns its inside/outside slopes when PLS is fitted on the subgroup and on the rest.
Writes ``results/exp4_published_model/recipe_check.csv``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from frame import load_real_frame, prepared  # noqa: E402
from pls_estimates import inner_estimates  # noqa: E402
from recipe import FIGURE2, Planted, Recipe, implied  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "exp4_published_model"


def stack(recipe: Recipe, template: pd.DataFrame, copies: int, seed: int) -> pd.DataFrame:
    frames = [recipe.simulate(template, s) for s in np.random.SeedSequence(seed).spawn(copies)]
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    S, r2 = implied(FIGURE2)
    assert np.allclose(np.diag(S), 1.0)
    real, _ = load_real_frame()
    template, _ = prepared(real)
    rec = Recipe(FIGURE2)
    print("error SDs:", {k: round(v, 3) for k, v in rec.error_sd.items()})
    big, cfg = prepared(stack(rec, template, 40, 1))
    est = inner_estimates(big, cfg)["estimate"]
    rows = [{"check": "null, 40 copies", "path": f"{f} -> {t}", "recipe": b, "pls": est[f"{f} -> {t}"]}
            for (f, t), b in FIGURE2.items()]
    medium = (template["school_size_tertile"] == "medium").to_numpy()
    pl = Recipe(FIGURE2, Planted("school_size_tertile=='medium'", medium, -0.08, -0.16))
    bigp, cfgp = prepared(stack(pl, template, 40, 2))
    m = np.tile(medium, 40)
    for part, mask, want in [("inside", m, -0.08), ("outside", ~m, -0.16)]:
        rows.append({"check": f"planted, {part}", "path": "feedback -> Achievement", "recipe": want,
                     "pls": inner_estimates(bigp, cfgp, mask).loc["feedback -> Achievement", "estimate"]})
    df = pd.DataFrame(rows)
    df["diff"] = df["pls"] - df["recipe"]
    df.to_csv(OUT / "recipe_check.csv", index=False)
    print(df.round(4).to_string(index=False))
    print("largest |pls - recipe|:", round(float(df["diff"].abs().max()), 4))


if __name__ == "__main__":
    main()
