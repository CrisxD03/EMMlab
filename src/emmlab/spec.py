"""Specification models (the single source of truth for Worlds, models and settings).

Every spec is a pydantic model that round-trips through YAML. Hashes are computed on a
canonical JSON form so they are independent of key order and whitespace.
"""

from __future__ import annotations

import hashlib
import io
import itertools
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from ruamel.yaml import YAML

Level = Literal["school", "student"]
Role = Literal["normal", "decoy"]


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# ---------------------------------------------------------------------------------------------
# World
# ---------------------------------------------------------------------------------------------


class Distribution(_Base):
    """Distribution of a continuous root: normal with mean and SD."""

    mean: float = 0.0
    sd: float = Field(default=1.0, gt=0)


class Recipe(_Base):
    """Recipe of a continuous child: intercept + sum(slope * parent) + noise."""

    intercept: float = 0.0
    noise_sd: float | None = Field(default=None, ge=0)


class Target(_Base):
    """Optional targets solved for the node's noise SD and/or school random-intercept SD.

    - ``sd``: total SD of the node in the reference (pooled) population.
    - ``r2``: share of the node's variance explained by its parents and attribute effects.
    - ``icc``: share of the node's variance that lies between schools.
    """

    sd: float | None = Field(default=None, gt=0)
    r2: float | None = Field(default=None, gt=0, lt=1)
    icc: float | None = Field(default=None, ge=0, lt=1)

    @model_validator(mode="after")
    def _one_of_sd_r2(self) -> Target:
        if self.sd is not None and self.r2 is not None:
            raise ValueError("target: give either sd or r2, not both")
        return self


class ReliabilityRule(_Base):
    """Group-specific reliability (measurement non-invariance), e.g. lower for one gender."""

    when: dict[str, str]
    reliability: float = Field(gt=0, le=1)


class PlausibleValues(_Base):
    """Report this latent only as plausible values (PISA-style achievement).

    A noisy test score with ``reliability`` is generated from the true latent; ``n`` PVs are drawn
    from the posterior under a main-effects conditioning model on ``conditioning`` (attribute names
    and/or continuous node names). Interactions are deliberately absent from the conditioning model.
    """

    n: int = Field(default=10, ge=2, le=50)
    reliability: float = Field(default=0.9, gt=0, lt=1)
    conditioning: list[str] = []


class NoiseShape(_Base):
    """Shape of a node's student-level noise. Skewed noise keeps the mean (0) and SD, so every slope, mean and
    covariance (and so every analytic truth) is unchanged; only the distribution's shape differs.

    ``skew``: the noise's skewness (0 = normal; > 0 a long right tail, < 0 a long left tail). Drawn as a
    standardized gamma by quantile transform of the same normal draw, so a skewed dataset pairs with its
    normal twin student by student.
    """

    skew: float = Field(default=0.0, ge=-3, le=3)


class ObservationScale(_Base):
    """How the researcher observes a latent: the data column is a transform of the latent value, while the
    World's causal relations stay linear in the latent (children use the latent, not the observed score).

    Cut points are in SD units of the latent's pooled population distribution (0 = its mean):

    - ``censored``: scores outside [``low``, ``high``] are set to the bound (an exam's maximum or minimum score);
    - ``count``: a Poisson count with mean ``mean`` x exp(``spread`` z - ``spread``^2 / 2), z the standardized
      latent (number of activities: many zeros, a long right tail);
    - ``ordinal``: category 1..len(``cuts``) + 1 by the latent's position among ``cuts`` (a rating scale).
    """

    type: Literal["censored", "count", "ordinal"]
    low: float | None = None
    high: float | None = None
    mean: float = Field(default=1.0, gt=0)
    spread: float = Field(default=0.5, gt=0)
    cuts: list[float] = []

    @model_validator(mode="after")
    def _complete(self) -> ObservationScale:
        if self.type == "censored" and self.low is None and self.high is None:
            raise ValueError("a censored scale needs low and/or high")
        if self.type == "censored" and self.low is not None and self.high is not None and self.low >= self.high:
            raise ValueError("censored scale: low must be below high")
        if self.type == "ordinal" and (not self.cuts or any(b <= a for a, b in itertools.pairwise(self.cuts))):
            raise ValueError("an ordinal scale needs increasing cuts")
        return self


class ContinuousNode(_Base):
    """A continuous variable: a latent, or a continuous attribute (searchable descriptor)."""

    kind: Literal["latent", "attribute"] = "latent"
    level: Level = "student"
    role: Role = "normal"
    root: Distribution | None = None
    recipe: Recipe | None = None
    target: Target | None = None
    school_sd: float = Field(default=0.0, ge=0, description="SD of a school random intercept (student-level only)")
    observed: bool = True  # unobserved latents generate data but are not in the Dataset
    reliability: float | None = Field(default=None, gt=0, le=1)  # observed index = true + error
    reliability_when: list[ReliabilityRule] = []
    plausible_values: PlausibleValues | None = None
    noise: NoiseShape | None = None  # None = normal
    scale: ObservationScale | None = None  # None = the latent itself is observed
    label: str | None = None


class CategoricalAttribute(_Base):
    """A categorical descriptor with value proportions, optionally conditional on another one."""

    level: Level = "student"
    role: Role = "normal"
    values: dict[str, float]
    depends_on: str | None = None
    table: dict[str, dict[str, float]] | None = None
    ordinal: bool = False
    label: str | None = None

    @field_validator("values")
    @classmethod
    def _non_empty(cls, v: dict[str, float]) -> dict[str, float]:
        if len(v) < 2:
            raise ValueError("a categorical attribute needs at least two values")
        if any(p < 0 for p in v.values()):
            raise ValueError("proportions must be non-negative")
        return v


def _parse_arrow_key(key: str) -> tuple[str, str]:
    if "->" not in key:
        raise ValueError(f"arrow '{key}' must look like 'parent -> child'")
    a, b = key.split("->", 1)
    return a.strip(), b.strip()


class Arrow(_Base):
    """A direct cause between continuous nodes. Give a raw ``slope`` or a standardized ``std``."""

    parent: str = Field(alias="from")
    child: str = Field(alias="to")
    slope: float | None = None
    std: float | None = None

    @model_validator(mode="before")
    @classmethod
    def _compact(cls, data: Any) -> Any:
        # Accept the compact form {"feedback -> selfeff": 0.3}
        if isinstance(data, dict) and len(data) == 1:
            (k, v), = data.items()
            if isinstance(k, str) and "->" in k:
                p, c = _parse_arrow_key(k)
                return {"from": p, "to": c, "slope": v}
        return data

    @model_validator(mode="after")
    def _slope_or_std(self) -> Arrow:
        if (self.slope is None) == (self.std is None):
            raise ValueError(f"arrow {self.parent} -> {self.child}: give exactly one of slope or std")
        return self

    @property
    def key(self) -> str:
        return f"{self.parent} -> {self.child}"


class Shift(_Base):
    """Level shift: when all conditions hold, add ``delta`` to the target node's intercept."""

    when: dict[str, str]
    target: str
    delta: float


class Twist(_Base):
    """Planted subgroup: when all conditions hold, add ``delta`` to the slope on ``arrow``."""

    arrow: str
    when: dict[str, str]
    delta: float
    label: str | None = None

    @property
    def parent(self) -> str:
        return _parse_arrow_key(self.arrow)[0]

    @property
    def child(self) -> str:
        return _parse_arrow_key(self.arrow)[1]


class StudentsPerSchool(_Base):
    dist: Literal["fixed", "uniform", "poisson"] = "fixed"
    n: int = Field(default=35, ge=2)
    min: int = Field(default=2, ge=2)
    max: int = Field(default=60, ge=2)


class Oversample(_Base):
    when: dict[str, str]  # school-level attribute condition
    factor: float = Field(gt=0)


class SamplingDesign(_Base):
    """Unequal school selection (oversampled strata) -> final student weights and Fay-BRR replicates."""

    oversample: list[Oversample] = []
    replicate_weights: bool = True
    n_replicates: int = Field(default=80, ge=4)
    fay_k: float = Field(default=0.5, ge=0, lt=1)
    population_students: float = Field(default=500_000.0, gt=0)


class SchoolDesign(_Base):
    n_schools: int = Field(default=200, ge=2)
    students_per_school: StudentsPerSchool = StudentsPerSchool()
    # "stratified": school-level attribute combinations get fixed, proportional school counts (PISA's
    # explicit stratification); "random": each school's attributes are drawn independently.
    allocation: Literal["stratified", "random"] = "stratified"
    sampling: SamplingDesign | None = None


class World(_Base):
    name: str
    description: str = ""
    schools: SchoolDesign = SchoolDesign()
    attributes: dict[str, CategoricalAttribute] = {}
    nodes: dict[str, ContinuousNode] = {}
    arrows: list[Arrow] = []
    shifts: list[Shift] = []
    twists: list[Twist] = []
    generation: Literal["naive", "internally_standardized"] = "naive"
    # Presentation only: force the canvas layout (left to right, or top to bottom). Unset = automatic.
    # Not part of the World's content hash.
    layout_direction: Literal["RIGHT", "DOWN"] | None = None

    def without_twists(self) -> World:
        return self.model_copy(update={"twists": [], "name": f"{self.name}__null"}, deep=True)

    def latents(self) -> list[str]:
        return [n for n, v in self.nodes.items() if v.kind == "latent"]


# ---------------------------------------------------------------------------------------------
# Analysis side
# ---------------------------------------------------------------------------------------------


class ResearcherModel(_Base):
    name: str = "model"
    latents: list[str]
    arrows: list[str]  # "parent -> child"

    def edges(self) -> list[tuple[str, str]]:
        return [_parse_arrow_key(a) for a in self.arrows]

    def equations(self) -> dict[str, list[str]]:
        """Endogenous node -> ordered list of its parents (in latent order)."""
        eq: dict[str, list[str]] = {}
        for p, c in self.edges():
            eq.setdefault(c, []).append(p)
        order = {n: i for i, n in enumerate(self.latents)}
        return {c: sorted(ps, key=lambda x: order.get(x, 0)) for c, ps in sorted(
            eq.items(), key=lambda kv: order.get(kv[0], 0))}

    @classmethod
    def from_world(cls, world: World, name: str | None = None) -> ResearcherModel:
        lat = [n for n, v in world.nodes.items() if v.kind == "latent" and v.level == "student" and v.observed]
        arrows = [a.key for a in world.arrows if a.parent in lat and a.child in lat]
        return cls(name=name or f"{world.name}_model", latents=lat, arrows=arrows)


Normalisation = Literal["none", "centered", "within_std"]


class AnalysisConfig(_Base):
    normalisation: Normalisation = "centered"
    standardize: Literal["global_once", "per_group"] = "global_once"
    residualize_on: list[str] = []
    residualize_method: Literal["total_effect", "simple"] = "total_effect"
    se: Literal["cluster", "naive", "brr"] = "cluster"
    weights: Literal["none", "final"] = "none"
    # how plausible values enter: all PVs + Rubin pooling (correct), PV1 only, or the PV mean (wrong)
    pv_mode: Literal["rubin", "first", "mean"] = "rubin"
    brr_fay_k: float = Field(default=0.5, ge=0, lt=1)


class Binning(_Base):
    n_bins: int = Field(default=5, ge=2, le=20)
    one_sided: bool = True
    weighting: Literal["students", "schools"] = "students"


class SearchSpace(_Base):
    attributes: list[str] | None = None  # None = all attributes in the dataset
    binning: dict[str, Binning] = {}
    default_binning: Binning = Binning()
    depth: int = Field(default=2, ge=1, le=3)
    min_n: int = Field(default=100, ge=1)
    min_n_complement: int = Field(default=100, ge=1)
    min_schools: int = Field(default=0, ge=0)
    filter: dict[str, str] | None = None


QualityName = str  # a key of emm.QUALITY_FUNCTIONS, checked by EMMConfig


class EMMConfig(_Base):
    quality: QualityName = "ent_z"

    @field_validator("quality")
    @classmethod
    def _known_quality(cls, v: str) -> str:
        from .emm import QUALITY_FUNCTIONS  # late import: emm imports this module

        if v not in QUALITY_FUNCTIONS:
            raise ValueError(f"unknown quality function {v!r}; known: {', '.join(QUALITY_FUNCTIONS)}")
        return v

    baseline: Literal["complement", "population"] = "complement"
    paths: list[str] | Literal["all"] = "all"
    algorithm: Literal["beam", "exhaustive"] = "beam"
    beam_width: int = Field(default=20, ge=1)
    top_k: int = Field(default=10, ge=1)
    interval_level: float = Field(default=0.95, gt=0, lt=1)
    exact_bootstrap_reps: int = Field(default=99, ge=19)
    nominal_min_schools: int = Field(default=25, ge=0)  # the researcher's own fragile rule (current preset)
    # Standard errors behind the intervals the quality function and the findings use. "analysis" follows
    # AnalysisConfig.se (cluster-robust by default); "naive" is the student-level OLS error plspm reports.
    interval_se: Literal["analysis", "naive"] = "analysis"
    # Last canonicalization step (d): choose the top_k with a penalty for students already covered by higher picks
    # (van Leeuwen & Knobbe 2012). Off: the top_k highest-quality distinct findings, overlaps allowed.
    cover_selection: bool = True
    # Canonicalization steps (b) and (c): shorten A AND B to A when B adds no quality, and drop A AND B when it does
    # not beat A and B alone. Off: refinements stay in the list as found (two-condition twists are not absorbed).
    prune_refinements: bool = True


class MethodPreset(_Base):
    name: Literal["current", "recommended", "custom"]
    analysis: AnalysisConfig
    emm: EMMConfig


def preset(name: Literal["current", "recommended"]) -> MethodPreset:
    """The two method presets (design T1). ``current`` mirrors the researcher's pipeline.

    Both start without normalisation (pooled data); school-mean centering stays selectable (decision of
    Oct 2026: centering is a later analysis choice, not a default).
    """
    if name == "current":
        return MethodPreset(
            name="current",
            analysis=AnalysisConfig(normalisation="none", standardize="per_group",
                                    residualize_method="simple"),
            emm=EMMConfig(quality="current", baseline="complement", interval_se="naive"),
        )
    return MethodPreset(
        name="recommended",
        analysis=AnalysisConfig(normalisation="none", standardize="global_once",
                                residualize_method="total_effect"),
        emm=EMMConfig(quality="ent_z", baseline="complement"),
    )


class BootstrapConfig(_Base):
    reps: int = Field(default=399, ge=19)
    level: float = Field(default=0.95, gt=0, lt=1)
    stratify: bool = True
    top_k: int = Field(default=5, ge=0)


class GradingConfig(_Base):
    tau: float = Field(default=0.7, gt=0, le=1)
    proxy_min_jaccard: float = Field(default=0.3, ge=0, le=1)
    epsilon: float = Field(default=0.02, gt=0)
    oracle_students: int = Field(default=400_000, ge=10_000)


class RunSpec(_Base):
    """Everything needed to produce one Run (one seed)."""

    world: World
    model: ResearcherModel
    method: MethodPreset
    search: SearchSpace = SearchSpace()
    bootstrap: BootstrapConfig | None = None
    grading: GradingConfig = GradingConfig()
    permutation_reps: int = Field(default=0, ge=0)  # search-level permutation null (0 = off)
    holdout: bool = False  # discovery/confirmation split with Benjamini-Hochberg
    seed: int = 0


class Factor(_Base):
    path: str  # dotted path into RunSpec, e.g. "world.schools.n_schools"
    levels: list[Any]
    linked: list[str] = []  # more paths set to the same level (e.g. every twist's delta at once), not crossed


class ExperimentSpec(_Base):
    name: str
    aim: str = ""
    base: RunSpec
    presets: list[Literal["current", "recommended"]] = ["recommended"]
    factors: list[Factor] = []
    n_sim: int = Field(default=200, ge=1)
    master_seed: int = 0
    include_null_world: bool = True
    null_world: World | None = None  # default: the base World with its twists removed
    target_mcse: float | None = None
    max_sim: int | None = None


# ---------------------------------------------------------------------------------------------
# YAML I/O and hashing
# ---------------------------------------------------------------------------------------------


def _yaml() -> YAML:
    y = YAML()
    y.default_flow_style = False
    y.sort_base_mapping_type_on_output = False  # type: ignore[attr-defined]
    return y


def load_yaml(path: str | Path) -> Any:
    with open(path) as f:
        return _yaml().load(f)


def dump_yaml(data: Any) -> str:
    buf = io.StringIO()
    _yaml().dump(data, buf)
    return buf.getvalue()


def to_plain(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(mode="json", by_alias=True, exclude_none=True)


def canonical_json(model: BaseModel | dict[str, Any]) -> str:
    data = to_plain(model) if isinstance(model, BaseModel) else model
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


def spec_hash(model: BaseModel | dict[str, Any]) -> str:
    data = to_plain(model) if isinstance(model, BaseModel) else dict(model)
    data.pop("layout_direction", None)  # how the canvas is drawn is not part of what the World is
    return hashlib.sha256(canonical_json(data).encode()).hexdigest()[:16]


def load_world(path: str | Path) -> World:
    return World.model_validate(_plainify(load_yaml(path)))


def _plainify(obj: Any) -> Any:
    """Convert ruamel's commented containers into plain dict/list."""
    if isinstance(obj, dict):
        return {str(k): _plainify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_plainify(v) for v in obj]
    return obj


def load_spec(path: str | Path, cls: type[BaseModel]) -> Any:
    return cls.model_validate(_plainify(load_yaml(path)))
