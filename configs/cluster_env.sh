#!/usr/bin/env bash
# =============================================================================
# Software environment activation used by the Slurm scripts -- EDIT for your cluster.
# Sourced at the top of every job in slurm/.
# =============================================================================

# Conda environment created from environment.yml (see README.md -> Installation)
export MOAA_CONDA_ENV="/scratch/work/bijoym1/conda_envs/moaa"

# ---- Aalto Triton modules ----
if command -v module >/dev/null 2>&1; then
    module load mamba
    module load cuda/12.6.2 2>/dev/null || module load cuda
fi

# ---- Activate environment ----
if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source activate "${MOAA_CONDA_ENV}" 2>/dev/null || conda activate "${MOAA_CONDA_ENV}"
fi
