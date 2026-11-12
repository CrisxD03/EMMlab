"""M2 acceptance: generator, stages, artifact store (implementation_plan §3.2)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from emmlab import truth
from emmlab.estimand import EstimandEngine
from emmlab.generate import generate, seed_sequence
from emmlab.grade import TruthProvider
from emmlab.run import execute_run
from emmlab.spec import (
    AnalysisConfig,
    GradingConfig,
    ResearcherModel,
    RunSpec,
    World,
    load_world,
    preset,
    to_plain,
)
from emmlab.stages import normalise, ols_fit, prepare
from emmlab.store import Store, parquet_bytes
from emmlab.world import compile_world
from tests.worlds import PRESETS, world_a, world_b

crit = pytest.mark.criterion


@crit("GE1")
def test_generation_deterministic():
    cw = compile_world(load_world(PRESETS / "pisa_like.yaml"))
    a, b = generate(cw, seed_sequence(5, 0, 1)), generate(cw, seed_sequence(5, 0, 1))
    pd.testing.assert_frame_equal(a, b)
    assert parquet_bytes(a) == parquet_bytes(b)
    c = generate(cw, seed_sequence(5, 0, 2))
    assert not np.allclose(a["feedback"], c["feedback"])


@crit("GE2")
def test_replication_streams_addressable():
    cw = compile_world(world_a())
    loop = [generate(cw, seed_sequence(11, 3, k)) for k in range(5)]
    alone = generate(cw, seed_sequence(11, 3, 4))
    pd.testing.assert_frame_equal(loop[4], alone)


@crit("GE3")
def test_school_level_constant_and_counts():
    w = load_world(PRESETS / "pisa_like.yaml")
    cw = compile_world(w)
    df = generate(cw, 1)
    assert len(df) == w.schools.n_schools * w.schools.students_per_school.n
    assert (df.groupby("school_id").size() == 35).all()
    for col in ("school_type", "school_size"):
        assert (df.groupby("school_id")[col].nunique() == 1).all()


def _batch_z(vals: np.ndarray, analytic: float) -> float:
    return float((vals.mean() - analytic) / (vals.std(ddof=1) / np.sqrt(len(vals))))


@crit("GE4")
def test_realized_icc_proportions_and_moments():
    w = load_world(PRESETS / "pisa_like.yaml")
    w.schools.n_schools = 2000
    cw = compile_world(w)
    df = generate(cw, 2024)
    icc_true = truth.node_icc(cw)
    sid = df["school_id"].to_numpy()
    batches = np.array_split(np.arange(2000), 40)
    zs = []
    for v in ("feedback", "achievement", "selfeff"):
        est = []
        for b in batches:
            sub = df[(sid >= b[0]) & (sid <= b[-1])]
            m = sub.groupby("school_id")[v].transform("mean")
            within = ((sub[v] - m) ** 2).mean() * 35 / 34
            total = sub[v].var(ddof=0)
            est.append(1 - within / total)
        zs.append(_batch_z(np.array(est), icc_true[v]))
    # conditional proportions P(ses_q=q1 | vocational)
    est = []
    for b in batches:
        sub = df[(sid >= b[0]) & (sid <= b[-1]) & (df["school_type"] == "vocational")]
        est.append((sub["ses_q"] == "q1").mean())
    zs.append(_batch_z(np.array(est), 0.40))
    mu, sig = truth.pooled_moments(cw)
    for v in ("feedback", "achievement"):
        est = [df[(sid >= b[0]) & (sid <= b[-1])][v].mean() for b in batches]
        zs.append(_batch_z(np.array(est), mu[cw.index[v]]))
    assert np.all(np.abs(zs) < 4), zs


@crit("ST1")
def test_centering_zero_school_means():
    df = generate(compile_world(world_b()), 3)
    out = normalise(df[["feedback", "achievement"]].to_numpy(), df["school_id"].to_numpy(), "centered")
    means = pd.DataFrame(out).groupby(df["school_id"].to_numpy()).mean().to_numpy()
    assert np.abs(means).max() < 1e-12


@crit("ST2")
def test_standardization_once_vs_per_group():
    w = world_b()
    df = generate(compile_world(w), 4)
    model = ResearcherModel.from_world(w)
    rec = prepare(df, model, preset("recommended").analysis)
    assert all(abs(s) > 0 for _, s in rec.scale.values())
    assert np.allclose(rec.latents.mean(0), 0, atol=1e-12) and np.allclose(rec.latents.std(0), 1, atol=1e-12)
    mask = (df["gender"] == "f").to_numpy()
    common = rec.fit(mask)["achievement"].coef
    X, y = rec.X["achievement"][mask], rec.y["achievement"][mask]
    b_raw, *_ = ols_fit(X, y, rec.school[mask], cluster=False)
    np.testing.assert_allclose(common, b_raw, atol=1e-12)  # no re-standardization in the subgroup
    cur = prepare(df, model, preset("current").analysis)
    assert cur.analysis.standardize == "per_group"
    per = cur.fit(mask)["achievement"].coef
    Xs = (X - X.mean(0)) / X.std(0)
    ys = (y - y.mean()) / y.std()
    b_std, *_ = ols_fit(Xs, ys, cur.school[mask], cluster=False)
    np.testing.assert_allclose(per, b_std, atol=1e-12)


@crit("ST3")
def test_ols_and_cluster_se_match_statsmodels():
    import statsmodels.api as sm
    df = generate(compile_world(world_b()), 5)
    X = df[["feedback", "selfeff"]].to_numpy()
    y = df["achievement"].to_numpy()
    b, a, se, se_naive, r2 = ols_fit(X, y, df["school_id"].to_numpy())
    m = sm.OLS(y, sm.add_constant(X)).fit(cov_type="cluster", cov_kwds={"groups": df["school_id"].to_numpy()})
    np.testing.assert_allclose(b, m.params[1:], atol=1e-10)
    np.testing.assert_allclose(se, m.bse[1:], rtol=1e-10)
    m2 = sm.OLS(y, sm.add_constant(X)).fit()
    np.testing.assert_allclose(se_naive, m2.bse[1:], rtol=1e-10)
    assert r2 == pytest.approx(m2.rsquared, abs=1e-12)


@crit("ST4")
@pytest.mark.oracle
def test_pls_equivalence_single_indicator():
    w = world_b()
    df = generate(compile_world(w), 6)
    model = ResearcherModel(latents=["feedback", "selfeff", "achievement"],
                            arrows=["feedback -> selfeff", "feedback -> achievement", "selfeff -> achievement"])
    prep = prepare(df, model, AnalysisConfig(normalisation="none"))
    ours = {c: dict(zip(f.regs, f.coef, strict=True)) for c, f in prep.full.items()}
    payload = json.dumps({"data": {c: df[c].tolist() for c in model.latents}, "latents": model.latents,
                          "edges": [list(e) for e in model.edges()]})
    script = Path(__file__).parent / "oracles" / "pls_oracle.py"
    try:
        out = subprocess.run(["uv", "run", "--offline", "--no-project", "--isolated", "--with", "plspm",
                              "--with", "pandas<3", "python", str(script)], input=payload, capture_output=True,
                             text=True, timeout=300, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:  # pragma: no cover
        pytest.fail(f"PLS oracle unavailable offline: {getattr(e, 'stderr', e)}")
    pls = json.loads(out.stdout.strip().splitlines()[-1])
    for c, d in pls.items():
        for p, v in d.items():
            assert ours[c][p] == pytest.approx(v, abs=1e-6)


def _calibration(w: World, norm: str, reps: int, model: ResearcherModel) -> list[tuple[str, float, float]]:
    cw = compile_world(w)
    an = AnalysisConfig(normalisation=norm)
    tp = TruthProvider(cw, model, an, None, GradingConfig(oracle_students=4_000_000))
    t3, t3_se = tp.full(), tp.full_se()
    ests: dict[str, list[float]] = {}
    for k in range(reps):
        df = generate(cw, seed_sequence(77, k))
        prep = prepare(df, model, an)
        for c, f in prep.full.items():
            for i, r in enumerate(f.regs):
                ests.setdefault(f"{r} -> {c}", []).append(float(f.coef[i]))
    out = []
    for path, v in ests.items():
        r, c = [x.strip() for x in path.split("->")]
        a = np.array(v)
        mcse = a.std(ddof=1) / np.sqrt(len(a))
        se = np.sqrt(mcse**2 + t3_se[c][r] ** 2)  # oracle truths carry their own Monte Carlo error
        out.append((path, (a.mean() - t3[c][r]) / se, a.mean() - t3[c][r]))
    return out


@crit("ST5")
@pytest.mark.slow
@pytest.mark.parametrize("norm", ["none", "centered", "within_std"])
def test_estimand_calibration_world_b(norm):
    w = world_b()
    model = ResearcherModel.from_world(w)
    res = _calibration(w, norm, 1000, model)
    assert all(abs(z) < 3.0 for _, z, _ in res), res


@crit("ST6")
@pytest.mark.slow
def test_clustering_lesson_coverage():
    w = World.model_validate({
        "name": "clustered", "schools": {"n_schools": 200, "students_per_school": {"n": 35}},
        "nodes": {"feedback": {"root": {"mean": 0, "sd": 1}, "target": {"icc": 0.4}},
                  "achievement": {"recipe": {}, "target": {"sd": 1.0, "icc": 0.4}}},
        "arrows": [{"feedback -> achievement": 0.2}]})
    cw = compile_world(w)
    model = ResearcherModel.from_world(w)
    an = AnalysisConfig(normalisation="none")
    t3 = EstimandEngine(cw, model, an).full["achievement"].coef[0]
    cov_c = cov_n = 0
    reps = 1000
    for k in range(reps):
        prep = prepare(generate(cw, seed_sequence(99, k)), model, an)
        f = prep.full["achievement"]
        cov_c += abs(f.coef[0] - t3) <= 1.96 * f.se[0]
        cov_n += abs(f.coef[0] - t3) <= 1.96 * f.se_naive[0]
    assert cov_n / reps < 0.90, cov_n / reps
    assert 0.93 <= cov_c / reps <= 0.97, cov_c / reps


@crit("AS1")
def test_cache_hits_and_downstream_invalidation(tmp_path):
    w = world_a()
    spec = RunSpec(world=w, model=ResearcherModel.from_world(w), method=preset("recommended"), seed=3)
    s = Store(tmp_path)
    r1 = execute_run(spec, s)
    first = list(s.computed)
    assert {"run", "dataset", "fits", "emm", "grade"} <= set(first)
    s2 = Store(tmp_path)
    r2 = execute_run(spec, s2)
    assert s2.computed == [] and r1["run_key"] == r2["run_key"]
    spec3 = spec.model_copy(update={"grading": spec.grading.model_copy(update={"tau": 0.8})}, deep=True)
    s3 = Store(tmp_path)
    execute_run(spec3, s3)
    assert sorted(s3.computed) == ["grade", "grade_no_cover", "grade_no_prune", "run"], s3.computed  # all gradings only


@crit("AS2")
def test_manifest_contents(tmp_path):
    w = world_a()
    spec = RunSpec(world=w, model=ResearcherModel.from_world(w), method=preset("current"), seed=8)
    spec.method.emm.top_k = 7
    s = Store(tmp_path)
    r = execute_run(spec, s)
    man = json.loads((s.path(r["run_key"]) / "manifest.json").read_text())
    for k in ("world", "model", "method", "method_preset", "overrides", "seed_path", "versions", "stages"):
        assert k in man, k
    assert man["method_preset"] == "current" and man["overrides"] == {"emm.top_k": 7}
    assert man["seed_path"] == [8]
    assert {"numpy", "pysubgroup", "emmlab"} <= set(man["versions"])
    assert to_plain(spec)["world"]["name"] == "minimal"


def _skew_world(skew: float | None = None, scale: dict | None = None) -> World:
    d = {"name": "shape", "attributes": {"g": {"values": {"a": .5, "b": .5}}},
         "nodes": {"feedback": {"recipe": {"noise_sd": 1.0}}, "achievement": {"recipe": {"noise_sd": 1.0}},
                   "later": {"recipe": {"noise_sd": 1.0}}},
         "arrows": [{"feedback -> achievement": 0.3}, {"achievement -> later": 0.5}],
         "shifts": [{"when": {"g": "b"}, "target": "feedback", "delta": 0.5}]}
    if skew is not None:
        d["nodes"]["achievement"]["noise"] = {"skew": skew}
    if scale is not None:
        d["nodes"]["achievement"]["scale"] = scale
    return World.model_validate(d)


@crit("GE5")
def test_skewed_noise_keeps_moments_and_truth():
    from scipy import stats
    base = generate(compile_world(_skew_world()), 7, n_schools=4000)
    # skew 0 is the normal World, draw for draw
    assert base.equals(generate(compile_world(_skew_world(0.0)), 7, n_schools=4000))
    for g in (2.0, -1.0):
        w = _skew_world(g)
        df = generate(compile_world(w), 7, n_schools=4000)
        noise = df["achievement"] - 0.3 * df["feedback"]
        noise0 = base["achievement"] - 0.3 * base["feedback"]
        assert abs(noise.mean()) < 0.01 and abs(noise.std() - 1) < 0.01
        assert abs(stats.skew(noise) - g) < 0.1
        # the same students keep their rank (quantile transform of the same normal draw)
        assert stats.spearmanr(noise, noise0).statistic > 0.999
        assert df["feedback"].equals(base["feedback"])
        # moments are unchanged, so the analytic truth still holds for the observed slope
        eng = EstimandEngine(compile_world(w), ResearcherModel.from_world(w), AnalysisConfig(normalisation="none"))
        b = np.polyfit(df["achievement"], df["later"], 1)[0] * df["achievement"].std() / df["later"].std()
        assert abs(b - eng.full["later"].coef[0]) < 0.02


@crit("GE6")
def test_observation_scales_transform_only_the_observed_column():
    from scipy import stats

    from emmlab.world import validate_world
    base = generate(compile_world(_skew_world()), 3, n_schools=4000)
    cw = compile_world(_skew_world())
    m, cov = truth.pooled_moments(cw)
    mu, sd = m[cw.index["achievement"]], float(np.sqrt(cov[cw.index["achievement"], cw.index["achievement"]]))
    z = (base["achievement"] - mu) / sd
    cens = generate(compile_world(_skew_world(scale={"type": "censored", "high": 1.0})), 3, n_schools=4000)
    assert cens["achievement"].max() <= mu + sd + 1e-12
    assert abs((cens["achievement"] >= mu + sd - 1e-12).mean() - stats.norm.sf(1.0)) < 0.01
    assert np.allclose(cens["achievement"], np.minimum(base["achievement"], mu + sd))
    ordinal = generate(compile_world(_skew_world(scale={"type": "ordinal", "cuts": [-1, 0, 1]})), 3, n_schools=4000)
    assert set(ordinal["achievement"].unique()) == {1.0, 2.0, 3.0, 4.0}
    assert (ordinal["achievement"] == 1 + (z >= -1).astype(int) + (z >= 0) + (z >= 1)).all()
    count = generate(compile_world(_skew_world(scale={"type": "count", "mean": 1.2, "spread": 0.8})), 3,
                     n_schools=4000)
    assert (count["achievement"] == count["achievement"].round()).all() and count["achievement"].min() == 0
    assert abs(count["achievement"].mean() - 1.2) < 0.05
    # children use the latent, not the observed score; everything else is drawn as in the unscaled World
    for df in (cens, ordinal, count):
        assert df[["feedback", "later", "g", "school_id"]].equals(base[["feedback", "later", "g", "school_id"]])
    # scales have no closed-form truth
    with pytest.raises(truth.NotAnalytic):
        sw = _skew_world(scale={"type": "censored", "high": 1.0})
        EstimandEngine(compile_world(sw), ResearcherModel.from_world(sw), AnalysisConfig())
    bad = _skew_world(scale={"type": "censored", "high": 1.0})
    bad.nodes["achievement"].reliability = 0.8
    assert any("observation scale" in str(i) for i in validate_world(bad))
