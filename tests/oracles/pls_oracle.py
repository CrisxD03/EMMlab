"""Independent PLS-SEM oracle (plspm), run in an isolated uv env with pandas<3 (plspm is pandas-2 only)."""
import json  # noqa: E402
import sys  # noqa: E402
import warnings  # noqa: E402

warnings.simplefilter("ignore")
import pandas as pd  # noqa: E402
import plspm.config as c  # noqa: E402
from plspm.mode import Mode  # noqa: E402
from plspm.plspm import Plspm  # noqa: E402
from plspm.scheme import Scheme  # noqa: E402

spec = json.load(sys.stdin)
df = pd.DataFrame(spec["data"])
lvs = spec["latents"]
path = pd.DataFrame(0, index=lvs, columns=lvs)
for p, ch in spec["edges"]:
    path.loc[ch, p] = 1
cfg = c.Config(path, scaled=True)
for lv in lvs:
    cfg.add_lv(lv, Mode.A, c.MV(lv + "_mv"))
res = Plspm(df.rename(columns={lv: lv + "_mv" for lv in lvs}), cfg, Scheme.PATH)
pc = res.path_coefficients()
print(json.dumps({ch: {p: float(pc.loc[ch, p]) for p, c2 in spec["edges"] if c2 == ch} for ch in set(c2 for _, c2 in spec["edges"])}))
