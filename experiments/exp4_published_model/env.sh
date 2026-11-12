# The old pipeline's pinned environment (Hyp-Gen-migration-scratch/requirements.txt): plspm 0.5.7 needs pandas 2.
# Usage: source experiments/exp4_published_model/env.sh; exp4py experiments/exp4_published_model/<script>.py (used by the scripts in this folder)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
exp4py() {
  uv run --no-project --python 3.12 --with plspm==0.5.7 --with pandas==2.3.3 --with numpy==1.26.4 \
    --with pysubgroup==0.9.0 --with scipy --with matplotlib --with pytest --with pyreadstat python "$@"
}
