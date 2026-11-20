# EMMlab

EMMlab is a ground-truth laboratory for Exceptional Model Mining (EMM) on path models. You describe a synthetic World where every number is known, the tool generates PISA-like data from it, runs the analysis a researcher would run (a path model plus subgroup discovery), and grades every finding against the truth. The same analysis also runs on real PISA-format data, which has no answer key.

## Requirements

- Python 3.12 or newer, installed through [uv](https://docs.astral.sh/uv/) (uv can fetch Python itself)
- Node 22 and pnpm 10 for the web app (tested with these versions); if `pnpm` is missing, run `npm install -g pnpm@10` first
- macOS or Linux: also `make`
- Windows: no `make` needed; see the Windows steps below

The install steps below fetch every Python and web library from the lock files (`uv.lock`, `web/pnpm-lock.yaml`), so there is nothing to install by hand.

## Quickstart

### macOS and Linux

```bash
make sync     # installs Python and web dependencies
make serve    # builds the web app and serves it at http://127.0.0.1:8765
make check    # ruff, basedpyright, pytest, web type-check, lint and tests
```

### Windows (PowerShell)

`make` is not available on Windows by default, so run the steps it wraps:

```powershell
uv sync                                # installs Python dependencies
pnpm -C web install                    # installs web dependencies
pnpm -C web build                      # builds the web app
uv run lab serve --port 8765           # serves it at http://127.0.0.1:8765
```

For the checks, run `uv run ruff check src tests`, `uv run basedpyright`, `uv run pytest -q`, then `pnpm -C web run check` and `pnpm -C web test`.

**Platform status:** EMMlab was developed and tested on macOS. Windows has not been tested. It is expected to work, since the Python code uses portable paths and the `spawn` process model, but it is unverified. Known Windows gaps: `make` and the `.sh` scripts (`scripts/smoke_clean_clone.sh`, `experiments/exp4_published_model/env.sh`) need WSL or Git Bash, and `scripts/offline_preview.py` may need `npx` available as `npx.cmd`. Please report any Windows problems as an issue.

## Command line

| Command | What it does |
|---|---|
| `uv run lab presets` | List the preset Worlds |
| `uv run lab truth <world>` | Equations and the true values of a World |
| `uv run lab run <world>` | One run: generate, analyse, search, grade, HTML report |
| `uv run lab experiment <spec.yaml>` | A replicated experiment with a null World; resumable |
| `uv run lab real <data.yaml> <model.yaml>` | The same analysis on real PISA-format data |
| `uv run lab reproduce <manifest.json>` | Re-execute a stored run or experiment and check the results are identical |

## Layout

| Folder | Contents |
|---|---|
| `src/emmlab/` | The tool: Worlds, data generation, truths, analysis stages, EMM search and quality functions, grading, CLI and server |
| `web/` | The React web app |
| `tests/` | The test suite |
| `examples/` | Example experiment specifications |
| `experiments/` | The experiment setups, scripts and results |
| `scripts/` | The clean-clone smoke test and an offline preview |

## Experiments

Each setup is a YAML file in `experiments/`, run with `uv run python experiments/run_experiments.py <name>`. Results are in `experiments/results/<name>/`. All five share the World in `experiments/dutch_two_twists.yaml`.

| Experiment | Setup and results | Question |
|---|---|---|
| 0 | the null runs inside `exp1a_current` and `exp1b_measures` (`null_runs.csv`) | What does EMM find when nothing is planted? |
| 1 | `exp1a_current` (the `current` quality function), `exp1b_measures` (`ent_z` and `cook`) | How do the number of schools, the size of the planted twist and the quality function change recovery? |
| 2 | `exp2_models` | What if the Researcher's Model is slightly wrong? |
| 3 | `exp3a_skew`, `exp3b_scales` | What if the data are skewed, bounded or counted? |
| 4 | `exp4_published_model` | Could chance alone change a published result? |

Experiment 4 cannot be rerun from this repository alone. Its scripts read the raw PISA 2015 files and the earlier analysis pipeline from a separate checkout; set `HYPGEN_ROOT` to that checkout, then `source experiments/exp4_published_model/env.sh`. The scripts and results are included for reference.
