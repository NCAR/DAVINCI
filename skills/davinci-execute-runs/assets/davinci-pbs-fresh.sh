#!/bin/bash
# Render placeholders before use. Scheduler logs must remain outside aNNN.
#PBS -N __RUN_ID__-__ATTEMPT_ID__
#PBS -A __PROJECT__
#PBS -q __QUEUE__
#PBS -l select=1:ncpus=__NCPUS__:mem=__MEMORY__
#PBS -l walltime=__WALLTIME__
#PBS -o __REVISION_ROOT__/scheduler-logs/__ATTEMPT_ID__.out
#PBS -e __REVISION_ROOT__/scheduler-logs/__ATTEMPT_ID__.err

set -euo pipefail

source __CONDA_SH__
conda activate davinci

export DAVINCI_RUN_ROOT="__ATTEMPT_ROOT__"
cd "__REPOSITORY_ROOT__"

davinci validate "__CONTROL_PATH__" --strict --readiness
davinci run "__CONTROL_PATH__"
