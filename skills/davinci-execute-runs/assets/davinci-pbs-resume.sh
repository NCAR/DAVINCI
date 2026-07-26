#!/bin/bash
# Render placeholders before use. This job continues the same incomplete aNNN.
#PBS -N __RUN_ID__-__ATTEMPT_ID__-resume
#PBS -A __PROJECT__
#PBS -q __QUEUE__
#PBS -l select=1:ncpus=__NCPUS__:mem=__MEMORY__
#PBS -l walltime=__WALLTIME__
#PBS -o __REVISION_ROOT__/scheduler-logs/__ATTEMPT_ID__-resume-__SUBMISSION_TAG__.out
#PBS -e __REVISION_ROOT__/scheduler-logs/__ATTEMPT_ID__-resume-__SUBMISSION_TAG__.err

set -euo pipefail

source __CONDA_SH__
conda activate davinci

export DAVINCI_RUN_ROOT="__ATTEMPT_ROOT__"
cd "__REPOSITORY_ROOT__"

davinci validate "__CONTROL_PATH__" --strict --readiness --resume
davinci run "__CONTROL_PATH__" --resume-plan
davinci run "__CONTROL_PATH__" --resume
