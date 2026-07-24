#!/bin/bash
# Fetch the previous UTC day's AirNow surface datasets into the standard
# AirNow archive used by the daily WRF-Chem forecast evaluation.
#
# Replicates the current melodies-scripts/get_airnow.sh: writes one file per
# day, named AirNow_YYYYMMDD.nc, where YYYYMMDD is yesterday's date.
#
# Usage:
#   ./fetch_airnow.sh
#
# Or pass an explicit date:
#   ./fetch_airnow.sh 20250801
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/activate_env.sh"

AIRNOW_DIR=${AIRNOW_DIR:-/glade/work/fillmore/Data/AirNow}

if [ "$#" -ge 1 ]; then
    fetch_date=$1
else
    fetch_date=$(date --date=yesterday '+%Y%m%d')
fi

next_date=$(date --date="${fetch_date} + 1 day" '+%Y%m%d')

mkdir -p "${AIRNOW_DIR}"

echo "Fetching AirNow ${fetch_date} -> ${next_date}"
davinci get airnow \
    --start-date "${fetch_date}" \
    --end-date   "${next_date}" \
    -o "${AIRNOW_DIR}/AirNow_${fetch_date}.nc"
