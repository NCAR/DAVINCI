#!/bin/bash
# PBS worker for one UTC day of AOD variable subsetting.

set -euo pipefail
umask 027

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BASE="${DAVINCI_CONDA_BASE:-/glade/work/fillmore/miniforge3}"

: "${AOD_SUBSET_START:?AOD_SUBSET_START is required}"
: "${AOD_SUBSET_END:?AOD_SUBSET_END is required}"
: "${AOD_SUBSET_SOURCE_ROOT:?AOD_SUBSET_SOURCE_ROOT is required}"
: "${AOD_SUBSET_OUTPUT_ROOT:?AOD_SUBSET_OUTPUT_ROOT is required}"
: "${AOD_SUBSET_PRODUCTS:=all}"
: "${AOD_SUBSET_ALLOW_MISSING:=0}"
: "${AOD_SUBSET_OVERWRITE:=0}"
: "${AOD_SUBSET_LAT_MIN:=}"
: "${AOD_SUBSET_LAT_MAX:=}"
: "${AOD_SUBSET_LON_MIN:=}"
: "${AOD_SUBSET_LON_MAX:=}"

array_index="${PBS_ARRAY_INDEX:-0}"
day="$(date -I -d "$AOD_SUBSET_START + $array_index days")"
[[ "$day" > "$AOD_SUBSET_END" ]] && {
    printf 'Array date %s is after requested end %s; nothing to do.\n' "$day" "$AOD_SUBSET_END"
    exit 0
}

# shellcheck disable=SC1090
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate davinci

args=(
    --start "$day"
    --end "$day"
    --products "$AOD_SUBSET_PRODUCTS"
    --source-root "$AOD_SUBSET_SOURCE_ROOT"
    --output-root "$AOD_SUBSET_OUTPUT_ROOT"
)
[[ "$AOD_SUBSET_ALLOW_MISSING" == 1 ]] && args+=(--allow-missing)
[[ "$AOD_SUBSET_OVERWRITE" == 1 ]] && args+=(--overwrite)
if [[ -n "$AOD_SUBSET_LAT_MIN" ]]; then
    args+=(
        --lat-min "$AOD_SUBSET_LAT_MIN"
        --lat-max "$AOD_SUBSET_LAT_MAX"
        --lon-min "$AOD_SUBSET_LON_MIN"
        --lon-max "$AOD_SUBSET_LON_MAX"
    )
fi

echo "Subsetting AOD inputs for $day (array index $array_index)"
python "$SCRIPT_DIR/subset_aod_earthdata.py" "${args[@]}"
