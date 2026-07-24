#!/bin/bash
# Activate the conda env and run davinci against the wrfchem-forecast
# example config. Picks up YYYY/MM/DD from the environment exported by
# qsub_wrfchem_daily.sh.
#
# This indirection exists so the PBS compute node (which gets a sparse env)
# can find `davinci` without depending on the user's ~/.bashrc.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYSIS_DIR="$(dirname "${SCRIPT_DIR}")"
CONFIG="${DAVINCI_CONFIG:-${ANALYSIS_DIR}/configs/wrfchem-forecast.example.yaml}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/activate_env.sh"

davinci run "${CONFIG}"
