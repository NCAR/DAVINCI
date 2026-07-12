#!/bin/bash
# PBS worker for one monthly slice of an Earthaccess AOD staging request.

set -euo pipefail
umask 027

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BASE="${DAVINCI_CONDA_BASE:-/glade/work/fillmore/miniforge3}"

: "${AOD_START:?AOD_START is required}"
: "${AOD_END:?AOD_END is required}"
: "${AOD_ROOT:?AOD_ROOT is required}"
: "${AOD_PRODUCTS:=all}"
: "${AOD_ALLOW_MISSING:=0}"
: "${AOD_SEARCH_ONLY:=0}"

array_index="${PBS_ARRAY_INDEX:-0}"
first_month="$(date -I -d "${AOD_START:0:7}-01")"
month_start="$(date -I -d "$first_month + $array_index months")"
month_end="$(date -I -d "$month_start + 1 month - 1 day")"

job_start="$month_start"
job_end="$month_end"
[[ "$job_start" < "$AOD_START" ]] && job_start="$AOD_START"
[[ "$job_end" > "$AOD_END" ]] && job_end="$AOD_END"

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate davinci

args=(
    --start "$job_start"
    --end "$job_end"
    --products "$AOD_PRODUCTS"
    --root "$AOD_ROOT"
)
[[ "$AOD_ALLOW_MISSING" == 1 ]] && args+=(--allow-missing)
[[ "$AOD_SEARCH_ONLY" == 1 ]] && args+=(--dry-run)

echo "Staging AOD inputs for $job_start through $job_end (array index $array_index)"
python "$SCRIPT_DIR/download_aod_earthdata.py" "${args[@]}"
