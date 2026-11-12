"""PLS path estimates of a fit on a subset of rows (as the quality functions fit them)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from plspm.plspm import Plspm
from plspm.scheme import Scheme


def inner_estimates(prepared: pd.DataFrame, config, mask: np.ndarray | None = None) -> pd.DataFrame:
    """The 17 path estimates and standard errors of a PLS fit on the masked rows (as the quality functions fit)."""
    d = prepared if mask is None else prepared.iloc[np.flatnonzero(mask)]
    m = Plspm(d, config, Scheme.PATH, 1000, 1e-8, False).inner_model()
    m = m.rename(columns={"std error": "se"})
    m["path"] = m["from"] + " -> " + m["to"]
    return m.set_index("path")[["estimate", "se"]]
