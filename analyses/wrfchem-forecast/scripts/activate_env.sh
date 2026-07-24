#!/bin/bash
# Activate the davinci conda environment. Sourced by the other scripts
# in this directory so they work both interactively and from a sparse cron
# job (e.g. cron.hpc.ucar.edu) where no modules/conda init are pre-loaded.
#
# Override with DAVINCI_CONDA_BASE / DAVINCI_CONDA_ENV if the install moves.

DAVINCI_CONDA_BASE="${DAVINCI_CONDA_BASE:-/glade/work/fillmore/miniforge3}"
DAVINCI_CONDA_ENV="${DAVINCI_CONDA_ENV:-davinci}"

# shellcheck disable=SC1091
source "${DAVINCI_CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${DAVINCI_CONDA_ENV}"

if ! command -v davinci >/dev/null 2>&1; then
    echo "ERROR: DAVINCI is not installed in conda environment '${DAVINCI_CONDA_ENV}'." >&2
    echo "From the DAVINCI checkout, run: python -m pip install --no-deps -e ." >&2
    return 1
fi
