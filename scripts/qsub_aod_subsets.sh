#!/bin/bash
# Submit daily AOD subsetting tasks as a throttled PBS array.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<'EOF'
Usage: qsub_aod_subsets.sh --start YYYY-MM-DD --end YYYY-MM-DD [options]

Options:
  --products LIST       Comma-separated merra2,terra,aqua,terra-l2,aqua-l2.
                        Default: all five staged products.
  --source-root PATH    Default: /glade/derecho/scratch/$USER/DAVINCI-AOD/raw
  --output-root PATH    Default:
                        /glade/work/fillmore/Data/CERES-SARB-CAM7/AOD_SUBSETS
  --lat-min VALUE       Optional southern bound; requires all four bounds.
  --lat-max VALUE       Optional northern bound.
  --lon-min VALUE       Optional western bound in [-180, 180].
  --lon-max VALUE       Optional eastern bound in [-180, 180].
  --account ID          PBS project. Default: P19010000.
  --queue NAME          PBS queue. Default: casper@casper-pbs.
  --walltime HH:MM:SS   Per-day walltime. Default: 02:00:00.
  --max-concurrent N    Maximum simultaneous days. Default: 4.
  --allow-missing       Do not fail a day when a selected product is absent.
  --overwrite           Replace existing non-empty subset files.
  --print-only          Print the qsub command without submitting it.
  -h, --help            Show this help.

Each PBS array task processes one UTC day. Outputs and per-day manifests are
written atomically; reruns skip existing non-empty subsets unless --overwrite
is supplied.
EOF
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

start=""
end=""
products="all"
source_root="/glade/derecho/scratch/$USER/DAVINCI-AOD/raw"
output_root="/glade/work/fillmore/Data/CERES-SARB-CAM7/AOD_SUBSETS"
lat_min=""
lat_max=""
lon_min=""
lon_max=""
account="${DAVINCI_ACCOUNT:-P19010000}"
queue="${DAVINCI_QUEUE:-casper@casper-pbs}"
walltime="02:00:00"
max_concurrent=4
allow_missing=0
overwrite=0
print_only=0

while (($# > 0)); do
    case "$1" in
        --start|--end|--products|--source-root|--output-root|--lat-min|--lat-max|--lon-min|--lon-max|--account|--queue|--walltime|--max-concurrent)
            (($# >= 2)) || die "Missing value for $1"
            case "$1" in
                --start) start="$2" ;;
                --end) end="$2" ;;
                --products) products="$2" ;;
                --source-root) source_root="$2" ;;
                --output-root) output_root="$2" ;;
                --lat-min) lat_min="$2" ;;
                --lat-max) lat_max="$2" ;;
                --lon-min) lon_min="$2" ;;
                --lon-max) lon_max="$2" ;;
                --account) account="$2" ;;
                --queue) queue="$2" ;;
                --walltime) walltime="$2" ;;
                --max-concurrent) max_concurrent="$2" ;;
            esac
            shift 2
            ;;
        --allow-missing)
            allow_missing=1
            shift
            ;;
        --overwrite)
            overwrite=1
            shift
            ;;
        --print-only)
            print_only=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "Unknown argument: $1"
            ;;
    esac
done

[[ -n "$start" && -n "$end" ]] || die "--start and --end are required."
start="$(date -I -d "$start")" || die "Invalid --start date."
end="$(date -I -d "$end")" || die "Invalid --end date."
[[ "$start" > "$end" ]] && die "--start must not be after --end."
[[ "$max_concurrent" =~ ^[1-9][0-9]*$ ]] || die "--max-concurrent must be positive."
[[ -d "$source_root" ]] || die "Source root does not exist: $source_root"

bounds=("$lat_min" "$lat_max" "$lon_min" "$lon_max")
bound_count=0
for value in "${bounds[@]}"; do
    [[ -z "$value" ]] || ((bound_count += 1))
done
((bound_count == 0 || bound_count == 4)) || die "Specify all four geographic bounds, or none."

day_count=$(( ($(date -u -d "$end" +%s) - $(date -u -d "$start" +%s)) / 86400 + 1 ))
array_args=()
if ((day_count > 1)); then
    array_args=(-J "0-$((day_count - 1))%${max_concurrent}")
fi

log_dir="$output_root/logs"
mkdir -p "$log_dir"

export AOD_SUBSET_START="$start"
export AOD_SUBSET_END="$end"
export AOD_SUBSET_PRODUCTS="$products"
export AOD_SUBSET_SOURCE_ROOT="$source_root"
export AOD_SUBSET_OUTPUT_ROOT="$output_root"
export AOD_SUBSET_ALLOW_MISSING="$allow_missing"
export AOD_SUBSET_OVERWRITE="$overwrite"
export AOD_SUBSET_LAT_MIN="$lat_min"
export AOD_SUBSET_LAT_MAX="$lat_max"
export AOD_SUBSET_LON_MIN="$lon_min"
export AOD_SUBSET_LON_MAX="$lon_max"

variables="AOD_SUBSET_START,AOD_SUBSET_END,AOD_SUBSET_PRODUCTS,AOD_SUBSET_SOURCE_ROOT"
variables+=",AOD_SUBSET_OUTPUT_ROOT,AOD_SUBSET_ALLOW_MISSING,AOD_SUBSET_OVERWRITE"
variables+=",AOD_SUBSET_LAT_MIN,AOD_SUBSET_LAT_MAX,AOD_SUBSET_LON_MIN,AOD_SUBSET_LON_MAX"
command=(
    qsub
    -N davinci-aod-subset
    -A "$account"
    -q "$queue"
    -l "walltime=$walltime"
    -l select=1:ncpus=1:mem=4GB
    "${array_args[@]}"
    -j oe
    -o "$log_dir/"
    -v "$variables"
    -- "$SCRIPT_DIR/run_aod_subsets_pbs.sh"
)

printf 'Daily tasks: %d; concurrency limit: %d\n' "$day_count" "$max_concurrent"
if [[ "$print_only" == 1 ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
else
    "${command[@]}"
fi
