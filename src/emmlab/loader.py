"""Real-data loader: PISA-format files -> the internal Dataset layout (origin: real, no answer key).

The mapping names which source column feeds each internal variable. Plausible values are given as a
pattern with ``{m}`` (e.g. ``PV{m}SCIE``); categorical recodes map source codes to labels. Rows with
missing model variables are dropped listwise and recorded in the Frame description.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import Field

from .spec import _Base
from .stages import REP_PREFIX, WEIGHT


class RealDataSpec(_Base):
    path: str
    school_id: str = "CNTSCHID"
    latents: dict[str, str] = {}  # internal name -> source column
    plausible_values: dict[str, str] = {}  # internal name -> pattern with {m}, e.g. "PV{m}SCIE"
    n_pv: int = Field(default=10, ge=1)
    attributes: dict[str, str] = {}  # internal name -> source column
    recode: dict[str, dict[str, str]] = {}  # internal attribute -> {source value: label}
    numeric_attributes: list[str] = []  # attributes kept numeric (e.g. school size)
    weight: str | None = "W_FSTUWT"
    replicate_prefix: str | None = "W_FSTURWT"
    n_replicates: int = 80
    filter: dict[str, Any] = {}  # source column -> required value (e.g. {"CNT": "GRC"})


def read_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    if p.suffix.lower() in (".sav", ".zsav"):
        try:
            import pyreadstat  # type: ignore[import-not-found]
        except ImportError as e:  # pragma: no cover
            raise ImportError("reading SPSS files needs `uv add pyreadstat`") from e
        df, _ = pyreadstat.read_sav(str(p))
        return df
    return pd.read_csv(p)


def load_real(spec: RealDataSpec) -> tuple[pd.DataFrame, dict]:
    raw = read_table(spec.path)
    n0 = len(raw)
    for col, val in spec.filter.items():
        raw = raw[raw[col].astype(str) == str(val)]
    out = pd.DataFrame({"school_id": pd.factorize(raw[spec.school_id])[0]})
    for name, col in spec.attributes.items():
        s = raw[col]
        if name in spec.numeric_attributes:
            out[name] = pd.to_numeric(s, errors="coerce").to_numpy()
            continue
        s = s.astype(str)
        if name in spec.recode:
            s = s.map(lambda v, r=spec.recode[name]: r.get(str(v), r.get(str(v).split(".")[0], str(v))))
        out[name] = pd.Categorical(s.to_numpy())
    for name, col in spec.latents.items():
        out[name] = pd.to_numeric(raw[col], errors="coerce").to_numpy()
    for name, pat in spec.plausible_values.items():
        for m in range(1, spec.n_pv + 1):
            out[f"{name}_pv{m}"] = pd.to_numeric(raw[pat.format(m=m)], errors="coerce").to_numpy()
    if spec.weight:
        out[WEIGHT] = pd.to_numeric(raw[spec.weight], errors="coerce").to_numpy()
    if spec.replicate_prefix:
        for r in range(1, spec.n_replicates + 1):
            out[f"{REP_PREFIX}{r}"] = pd.to_numeric(raw[f"{spec.replicate_prefix}{r}"], errors="coerce").to_numpy()
    needed = [c for c in out.columns if c not in spec.numeric_attributes]
    before = len(out)
    out = out.dropna(subset=needed).reset_index(drop=True)
    frame = {"origin": "real", "source": str(spec.path), "rows_in_file": n0, "rows_after_filter": before,
             "rows_after_listwise": len(out), "schools": int(out["school_id"].nunique()), "filter": spec.filter,
             "missing_rule": "listwise deletion on model variables, weights and attributes"}
    return out, frame
