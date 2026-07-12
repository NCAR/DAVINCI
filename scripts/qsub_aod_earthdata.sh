#!/bin/bash
# Submit monthly Earthaccess AOD acquisition tasks as a throttled PBS array.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<'EOF'
Usage: qsub_aod_earthdata.sh --start YYYY-MM-DD --end YYYY-MM-DD [options]

Options:
  --products LIST       Comma-separated merra2,terra,aqua,terra-l2,aqua-l2.
                        Default: all daily products; L2 is opt-in.
  --root PATH           Raw staging root. Default:
                        /glade/derecho/scratch/$USER/DAVINCI-AOD/raw
  --account ID          PBS project. Default: P19010000.
  --queue NAME          PBS queue. Default: casper@casper-pbs.
  --walltime HH:MM:SS   Per-month walltime. Default: 06:00:00.
  --max-concurrent N    Maximum simultaneous array tasks. Default: 2.
  --allow-missing       Do not fail a task when CMR has a product-day gap.
  --search-only         Queue CMR searches and size estimates without downloads.
  --print-only          Print the qsub command without submitting it.
  -h, --help            Show this help.

The submitter creates one PBS array task per calendar month. Earthdata
credentials are read non-interactively from ~/.netrc by each worker.
EOF
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

start=""
end=""
products="all"
root="/glade/derecho/scratch/$USER/DAVINCI-AOD/raw"
account="${DAVINCI_ACCOUNT:-P19010000}"
queue="${DAVINCI_QUEUE:-casper@casper-pbs}"
walltime="06:00:00"
max_concurrent=2
allow_missing=0
search_only=0
print_only=0

while (($# > 0)); do
    case "$1" in
        --start|--end|--products|--root|--account|--queue|--walltime|--max-concurrent)
            (($# >= 2)) || die "Missing value for $1"
            case "$1" in
                --start) start="$2" ;;
                --end) end="$2" ;;
                --products) products="$2" ;;
                --root) root="$2" ;;
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
        --search-only)
            search_only=1
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
[[ -r "$HOME/.netrc" ]] || die "Earthdata credentials not found at $HOME/.netrc."

start_year=$((10#${start:0:4}))
start_month=$((10#${start:5:2}))
end_year=$((10#${end:0:4}))
end_month=$((10#${end:5:2}))
month_count=$(( (end_year - start_year) * 12 + end_month - start_month + 1 ))
array_end=$((month_count - 1))
array_spec="0-${array_end}%${max_concurrent}"
array_args=()
if ((month_count > 1)); then
    array_args=(-J "$array_spec")
fi

log_dir="$root/logs"
mkdir -p "$log_dir"

export AOD_START="$start"
export AOD_END="$end"
export AOD_ROOT="$root"
export AOD_PRODUCTS="$products"
export AOD_ALLOW_MISSING="$allow_missing"
export AOD_SEARCH_ONLY="$search_only"

command=(
    qsub
    -N davinci-aod
    -A "$account"
    -q "$queue"
    -l "walltime=$walltime"
    -l select=1:ncpus=1:mem=4GB
    "${array_args[@]}"
    -j oe
    -o "$log_dir/"
    -v AOD_START,AOD_END,AOD_ROOT,AOD_PRODUCTS,AOD_ALLOW_MISSING,AOD_SEARCH_ONLY
    -- "$SCRIPT_DIR/run_aod_earthdata_pbs.sh"
)

printf 'Monthly tasks: %d; concurrency limit: %d\n' "$month_count" "$max_concurrent"
if [[ "$print_only" == 1 ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
else
    "${command[@]}"
fi
