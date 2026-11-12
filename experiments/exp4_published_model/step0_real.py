"""Step 0 of Exp 4, on the real 4,564-student frame (no simulation).

    source experiments/exp4_published_model/env.sh; exp4py experiments/exp4_published_model/step0_real.py

1. The method gap: the single-indicator PLS fit next to Figure 2 (``real_fit.csv``).
2. The real subgroup list: the original search (path quality function on feedback -> achievement, exhaustive DFS,
   depth 2, top 10) on this frame (``real_findings.csv``), with how far each finding cleared the interval bar.
3. The frame's counts (``frame.json``).
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import search as S  # noqa: E402  (the earlier pipeline)
from frame import load_real_frame, prepared  # noqa: E402
from pls_estimates import inner_estimates  # noqa: E402
from recipe import FIGURE2  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / "exp4_published_model"
PATH = "feedback -> Achievement"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    real, counts = load_real_frame()
    frame, cfg = prepared(real)
    (OUT / "frame.json").write_text(json.dumps(asdict(counts), indent=1))

    fit = inner_estimates(frame, cfg)
    fit["published"] = [FIGURE2[tuple(p.split(" -> "))] for p in fit.index]
    fit["gap"] = fit["estimate"] - fit["published"]
    fit.to_csv(OUT / "real_fit.csv")
    print(fit.round(3).to_string())

    t0 = time.perf_counter()
    found = S.run_search(frame, cfg, S.SearchSpec("path"))
    secs = time.perf_counter() - t0
    rows = []
    for f in found:
        a = inner_estimates(frame, cfg, f.cover).loc[PATH]
        b = inner_estimates(frame, cfg, ~f.cover).loc[PATH]
        bar = 1.96 * (a["se"] + b["se"])
        rows.append({"rank": f.rank, "description": f.description, "size": f.size, "quality": f.quality,
                     "estimate_sg": f.estimate_sg, "estimate_rest": f.estimate_complement,
                     "diff": f.estimate_sg - f.estimate_complement, "bar": bar,
                     "margin": abs(f.estimate_sg - f.estimate_complement) / bar})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "real_findings.csv", index=False)
    print(f"\nsearch: {secs:.1f} s, {len(found)} subgroups with positive quality")
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
