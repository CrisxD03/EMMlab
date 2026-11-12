// Types mirroring the pydantic specs in src/emmlab/spec.py and the JSON the server returns.
// Hand-written (the API returns plain dicts for most routes, so a generated OpenAPI client would add
// little): keep in sync with spec.py, truth.derived, run.execute_run and experiment.aggregate.

export type Level = "school" | "student";
export type Role = "normal" | "decoy";

export interface Distribution {
  mean?: number;
  sd?: number;
}

export interface Recipe {
  intercept?: number;
  noise_sd?: number;
}

export interface Target {
  sd?: number;
  r2?: number;
  icc?: number;
}

export interface ContinuousNode {
  kind?: "latent" | "attribute";
  level?: Level;
  role?: Role;
  root?: Distribution;
  recipe?: Recipe;
  target?: Target;
  school_sd?: number;
  observed?: boolean;
  label?: string;
}

export interface CategoricalAttribute {
  level?: Level;
  role?: Role;
  values: Record<string, number>;
  depends_on?: string;
  table?: Record<string, Record<string, number>>;
  ordinal?: boolean;
  label?: string;
}

export interface Arrow {
  from: string;
  to: string;
  slope?: number;
  std?: number;
}

export interface Shift {
  when: Record<string, string>;
  target: string;
  delta: number;
}

export interface Twist {
  arrow: string;
  when: Record<string, string>;
  delta: number;
  label?: string;
}

export interface StudentsPerSchool {
  dist?: "fixed" | "uniform" | "poisson";
  n?: number;
  min?: number;
  max?: number;
}

export interface SchoolDesign {
  n_schools?: number;
  students_per_school?: StudentsPerSchool;
  allocation?: "stratified" | "random";
}

export interface World {
  name: string;
  description?: string;
  schools?: SchoolDesign;
  attributes: Record<string, CategoricalAttribute>;
  nodes: Record<string, ContinuousNode>;
  arrows: Arrow[];
  shifts: Shift[];
  twists: Twist[];
  generation?: "naive" | "internally_standardized";
  /** Presentation only: force the canvas layout; unset = automatic. */
  layout_direction?: "RIGHT" | "DOWN" | null;
}

export interface ResearcherModel {
  name?: string;
  latents: string[];
  arrows: string[];
}

export type Normalisation = "none" | "centered" | "within_std";

export interface AnalysisConfig {
  normalisation: Normalisation;
  standardize: "global_once" | "per_group";
  residualize_on: string[];
  residualize_method: "total_effect" | "simple";
  se: "cluster" | "naive";
}

/** A key of emm.QUALITY_FUNCTIONS on the server (GET /api/qualities lists them). */
export type QualityName = string;

export interface QualityInfo {
  name: QualityName;
  description: string;
  /** true: a path deviates when its intervals do not overlap; false: when |z| ≥ 1.96 */
  uses_intervals: boolean;
  /** offered in the Search & method dropdown */
  in_menu: boolean;
}

export interface EMMConfig {
  quality: QualityName;
  baseline: "complement" | "population";
  paths: string[] | "all";
  algorithm: "beam" | "exhaustive";
  beam_width: number;
  top_k: number;
  interval_level: number;
  exact_bootstrap_reps: number;
  nominal_min_schools: number;
  /** Standard errors behind the intervals: the analysis setting, or student-level (as plspm reports them). */
  interval_se: "analysis" | "naive";
  /** Last canonicalization step: spread the top list over different students (off = top k by quality). */
  cover_selection: boolean;
  /** Steps (b)+(c): shorten A AND B to A when B adds nothing, drop A AND B when it does not beat A and B alone. */
  prune_refinements: boolean;
}

export interface MethodPreset {
  name: "current" | "recommended" | "custom";
  analysis: AnalysisConfig;
  emm: EMMConfig;
}

export interface Binning {
  n_bins: number;
  one_sided: boolean;
  weighting: "students" | "schools";
}

export interface SearchSpace {
  attributes?: string[];
  binning: Record<string, Binning>;
  default_binning: Binning;
  depth: number;
  min_n: number;
  min_n_complement: number;
  min_schools: number;
  filter?: Record<string, string>;
}

export interface RunSpec {
  world: World;
  model: ResearcherModel;
  method: MethodPreset;
  search: SearchSpace;
  seed: number;
}

export interface Factor {
  path: string;
  levels: unknown[];
  /** More paths set to the same level (not crossed). */
  linked?: string[];
}

export interface ExperimentSpec {
  name: string;
  aim?: string;
  base: RunSpec;
  presets: ("current" | "recommended")[];
  factors: Factor[];
  n_sim: number;
  master_seed?: number;
  include_null_world?: boolean;
}

// ----------------------------------------------------------------------------- server responses

export interface Issue {
  element: string;
  message: string;
}

export interface Preset {
  name: string;
  description: string;
  world: World;
  yaml: string;
  twists: number;
  null_variant: boolean;
}

export interface GlossaryEntry {
  definition: string;
  source?: string;
  amendment?: string;
}
export type Glossary = Record<string, GlossaryEntry>;

export interface GroupMoments {
  mean: number;
  sd: number;
}

export interface DerivedNode {
  mean: number;
  sd: number;
  r2: number;
  icc: number;
  groups: Record<string, GroupMoments>;
  noise_sd?: number;
  school_sd?: number;
  explained_var?: number;
}

export interface DerivedArrowGroup {
  slope: number;
  total: number;
  std: number;
}

export interface DerivedArrow {
  baseline_slope: number;
  avg_slope: number;
  avg_total: number;
  std: number;
  groups: Record<string, DerivedArrowGroup>;
}

export interface Derived {
  nodes: Record<string, DerivedNode>;
  arrows: Record<string, DerivedArrow>;
  total_effects: Record<string, number>;
  varsortability: number;
  n_cells: number;
  warnings: string[];
}

/** One term of a coefficient polynomial: `coef` times the indicators `[attr=value]` in `when` (all must hold). */
export interface EqTerm {
  coef: number;
  when: [string, string][];
}

export interface UnfoldedNoise {
  symbol: string; // "e:<node>" (the node's own noise) or "u:<node>" (its school random intercept)
  kind: "e" | "u";
  node: string;
  sd: number | null; // null for a fitted model's residual (not part of a fit summary)
  poly: EqTerm[];
}

/** A node's generating equation: constant part + parents (direct form only) + noise sources. */
export interface UnfoldedEq {
  node: string;
  form: "roots" | "direct" | "fitted";
  const: EqTerm[];
  vars: { name: string; poly: EqTerm[] }[];
  noise: UnfoldedNoise[];
}

export interface Unfolded {
  equations: { roots: Record<string, UnfoldedEq>; direct: Record<string, UnfoldedEq> };
  primitives: {
    attributes: {
      name: string;
      level: Level;
      role: Role;
      values: Record<string, number>;
      depends_on: string | null;
    }[];
    noise: { symbol: string; kind: "e" | "u"; node: string; sd: number }[];
  };
  unavailable: Record<string, string>;
}

export interface TruthResponse {
  ok: boolean;
  issues: Issue[];
  derived?: Derived;
  equations?: string[];
  unfolded?: Unfolded;
  hash?: string;
  warnings?: string[];
}

export interface EstimandEquation {
  regs: string[];
  coef: number[];
  coef_before: number[];
  r2: number;
  r2_before: number;
  case: "triangle" | "chain" | null;
}

export interface EstimandsResponse {
  source: string;
  equations: Record<string, EstimandEquation>;
}

export type Descriptor =
  | { type: "categorical"; level: Level; role: Role; values: Record<string, number> }
  | { type: "numeric"; level: Level; role: Role; hist: number[]; edges: number[]; cuts: number[] };

export interface FindingPath {
  path: string;
  b_g: number;
  b_b: number;
  se_g: number;
  se_b: number;
  ci_g: [number, number];
  ci_b: [number, number];
  diff: number;
  z: number;
  disjoint: boolean;
}

export interface Finding {
  rank: number;
  description: string;
  conditions: { attr: string; op: string; value: string | number }[];
  quality: number;
  n: number;
  n_c: number;
  n_schools: number;
  g_eff: number;
  g_eff_c: number;
  best_path: string;
  paths: FindingPath[];
}

export interface GradeHit {
  twist: string;
  description: string;
  path: string;
  rank: number;
  jaccard: number;
  est_diff: number;
  true_diff: number;
  quality: number;
  significant: boolean;
  n_g: number;
}

export interface GradeMiss {
  twist: string;
  true_diff?: number;
  n_g?: number;
}

export interface FalsePositive {
  class: string;
  description: string;
  quality: number;
  rank: number;
  significant: boolean;
}

export interface Grade {
  hits: GradeHit[];
  misses: GradeMiss[];
  false_positives: FalsePositive[];
  fp_by_class: Record<string, number>;
  redundant: { description: string; twist?: string; rank?: number }[];
  n_findings: number;
}

export interface PathTruth {
  t1_direct_avg: number | null;
  t2_total_avg: number | null;
  t3: number;
  units_t1_t2: string;
  units_t3: string;
}

export interface Manifest {
  world: string;
  model: string;
  method: string;
  method_preset: string;
  overrides: Record<string, unknown>;
  search: string;
  grading: string;
  bootstrap: string | null;
  seed_path: number[];
  stages: Record<string, string>;
  label: string;
}

export interface EstimationRow {
  path: string;
  estimate: number;
  se: number;
  se_naive: number;
  t3: number;
  error: number;
  covered: boolean;
  covered_naive: boolean;
}

export interface Flag {
  description: string;
  g_eff: number;
  tier: "ok" | "fragile" | "unreliable";
  subgroup: { schools: number; g_eff: number; tier: string };
  complement: { schools: number; g_eff: number; tier: string };
  current_rule_fragile?: boolean;
}

export interface RunResult {
  run_key: string;
  truths: Record<string, PathTruth>;
  estimand_label: string;
  finding_truth: { source: string; common?: Record<string, number>; within?: Record<string, number> }[];
  manifest: Manifest;
  n: number;
  n_schools: number;
  estimation: EstimationRow[];
  emm: {
    findings: Finding[];
    raw: Finding[];
    evaluations: number;
    descriptors: unknown;
    coverage?: SearchCoverage;
  };
  grade: Grade;
  flags: Flag[];
  truth_source: string;
  /** Generating equations of the World that made this run's data (absent in older stored runs). */
  generating?: Unfolded | null;
  /** Fitted equations unfolded to the model's starting latents and residuals (full sample; subgroup and
   * complement per finding rank). Absent in older stored runs. */
  fitted_unfolded?: {
    full: Record<string, UnfoldedEq>;
    findings: Record<
      string,
      { subgroup: Record<string, UnfoldedEq>; complement: Record<string, UnfoldedEq> }
    >;
  } | null;
  spec: RunSpec;
}

/** How far the search got (absent in older stored runs). */
export interface SearchCoverage {
  scored: number;
  positive: number;
  zero: number;
  too_small: number;
  kept_limit: number;
  kept_positive: number;
  truncated: boolean;
}

export interface ContrastFit {
  b: number;
  se: number;
  ci: [number, number];
}

export interface ContrastCompare {
  diff: number | null;
  z: number | null;
  disjoint: boolean;
  deviating: boolean;
}

export interface ContrastSummary {
  n: number;
  schools: number;
  g_eff: number;
  tier: string;
  reliable: boolean;
  fits: Record<string, ContrastFit | null>;
}

export interface ContrastSlice extends ContrastSummary {
  key: string;
  role: "finding" | "flip" | "cell" | "complement" | "population";
  label: string;
  pattern: boolean[] | null;
  /** The finding compared with this slice, per path (null for the finding itself). */
  vs_finding: Record<string, ContrastCompare> | null;
}

export type VerdictKind = "conjunction_specific" | "carried_by" | "shared" | "single";

export interface ContrastResult {
  rank: number;
  description: string;
  conditions: string[];
  quality: QualityName;
  baseline: "complement" | "population";
  interval_level: number;
  interval_se: "analysis" | "naive";
  interval_method: "wald";
  paths: string[];
  focal_paths: string[];
  slices: ContrastSlice[];
  singles: {
    condition: string;
    alone: ContrastSummary;
    rest: ContrastSummary;
    vs_rest: Record<string, ContrastCompare>;
  }[];
  /** Every pair of grid cells (finding, flips, other sign patterns), per path. */
  pairs: { a: string; b: string; vs: Record<string, ContrastCompare> }[];
  /** Two conditions only: (A∧B − A∧¬B) − (¬A∧B − ¬A∧¬B) per path; additive when |z| < 1.96. */
  interaction: Record<
    string,
    { estimate: number; se: number; z: number; ci: [number, number]; additive: boolean }
  >;
  verdicts: Record<
    string,
    {
      kind: VerdictKind;
      needed: string[];
      low_power: boolean;
      text: string;
      /** Does each condition deviate on its own against everyone else? */
      alone_deviates?: Record<string, boolean>;
      /** A caution when the conditions the verdict relies on do not deviate alone. */
      note?: string | null;
    }
  >;
}

export interface ScatterResponse {
  path: string;
  subgroup: string | null;
  x: number[];
  y: number[];
  in_group: boolean[];
}

export interface Rate {
  value: number | null;
  mcse: number | null;
  n: number;
}

export interface TwistAggregate {
  hit_rate: Rate;
  hit_rate_significant: Rate;
  mean_rank: Rate;
  /** Same runs graded on the top k without cover-based selection (step d); absent in older results. */
  hit_rate_no_cover?: Rate;
  mean_rank_no_cover?: Rate;
  /** Hit whose twisted path also deviates under the run's rule; on the reported list and without steps (b)+(c). */
  recovery_rate?: Rate;
  recovery_rate_no_prune?: Rate;
  est_diff: Rate;
  true_diff: number | null;
  winners_curse: number | null;
  n_g: number | null;
  unstable: boolean;
}

export interface PresetAggregate {
  n_planted: number;
  n_null: number;
  threshold: number | null;
  fwer_null: Rate;
  twists: Record<string, TwistAggregate>;
  fdp: Record<string, Rate>;
  fdp_significant: Rate;
  fdp_no_cover?: Rate;
  decoy_rate: Rate;
}

export interface SweepCurve {
  points: { n_g: number; hit_rate: number; mcse: number; n: number; factors: Record<string, unknown> }[];
  n50: number | null;
  n80: number | null;
}

export interface Aggregate {
  name: string;
  aim: string;
  n_sim: number;
  presets: string[];
  factors: Factor[];
  conditions: { index: number; factors: Record<string, unknown>; presets: Record<string, PresetAggregate> }[];
  sweep: Record<string, SweepCurve & { significant: SweepCurve }>;
}

export type JobStatus = "queued" | "running" | "done" | "cancelled" | "error";

export interface JobSummary {
  id: string;
  name: string;
  status: JobStatus;
  created: number;
  total: number;
  done: number;
}

export interface Job extends JobSummary {
  spec: ExperimentSpec;
  updated: number;
  error: string | null;
  aggregate: Aggregate | null;
}
