#!/usr/bin/env bash
# Subset GEOS-IT 3-hourly 2-D AOD and average the selected fields by UTC day.

set -euo pipefail

readonly DEFAULT_SOURCE_ROOT="/ASDC_archive/GMAO/GEOSIT"
readonly DEFAULT_OUTPUT_ROOT="/CERES/sarb/dfillmor/DAVINCI"
readonly EXPECTED_DAILY_INPUTS=8
readonly AOD_VARIABLES="time,lat,lon,TOTEXTTAU"

usage() {
    cat <<'EOF'
Usage: subset_geosit_aod_daily.sh --start YYYY-MM-DD --end YYYY-MM-DD [options]

Required:
  --start DATE          First UTC date, inclusive.
  --end DATE            Last UTC date, inclusive.

Optional:
  --lat-min VALUE       Southern latitude bound. Requires all four bounds.
  --lat-max VALUE       Northern latitude bound. Requires all four bounds.
  --lon-min VALUE       Western longitude bound. Requires all four bounds.
  --lon-max VALUE       Eastern longitude bound. Requires all four bounds.
  --source-root PATH    Default: /ASDC_archive/GMAO/GEOSIT
  --output-root PATH    Default: /CERES/sarb/dfillmor/DAVINCI
  --skip-list PATH      Source files to treat as missing. Default:
                        <output-root>/GEOSIT_SKIPPED_SOURCE_FILES.txt when present.
  --allow-incomplete-day
                        Average a day with fewer than eight 3-hourly inputs.
  --allow-missing-day
                        Skip dates with no GEOS-IT AOD inputs.
  --overwrite           Replace an existing output file.
  --conda-sh PATH       Default: $HOME/miniforge3/etc/profile.d/conda.sh
  -h, --help            Show this help text.

Longitude limits must use the native GEOS-IT convention and cannot cross the
dateline in one invocation. Outputs contain only total 550-nm aerosol
extinction optical thickness (TOTEXTTAU).
EOF
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

is_number() {
    [[ "$1" =~ ^-?([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]
}

source_root="$DEFAULT_SOURCE_ROOT"
output_root="$DEFAULT_OUTPUT_ROOT"
skip_list=""
conda_sh="${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
start=""
end=""
lat_min=""
lat_max=""
lon_min=""
lon_max=""
allow_incomplete_day=false
allow_missing_day=false
overwrite=false

while (($# > 0)); do
    case "$1" in
        --start|--end|--lat-min|--lat-max|--lon-min|--lon-max|--source-root|--output-root|--skip-list|--conda-sh)
            (($# >= 2)) || die "Missing value for $1"
            case "$1" in
                --start) start="$2" ;;
                --end) end="$2" ;;
                --lat-min) lat_min="$2" ;;
                --lat-max) lat_max="$2" ;;
                --lon-min) lon_min="$2" ;;
                --lon-max) lon_max="$2" ;;
                --source-root) source_root="$2" ;;
                --output-root) output_root="$2" ;;
                --skip-list) skip_list="$2" ;;
                --conda-sh) conda_sh="$2" ;;
            esac
            shift 2
            ;;
        --allow-incomplete-day)
            allow_incomplete_day=true
            shift
            ;;
        --allow-missing-day)
            allow_missing_day=true
            shift
            ;;
        --overwrite)
            overwrite=true
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
[[ -d "$source_root" ]] || die "Source root does not exist: $source_root"
[[ -r "$conda_sh" ]] || die "conda.sh is not readable: $conda_sh"
if [[ -z "$skip_list" ]]; then
    skip_list="$output_root/GEOSIT_SKIPPED_SOURCE_FILES.txt"
fi
if [[ -e "$skip_list" && ! -r "$skip_list" ]]; then
    die "GEOS-IT skip list is not readable: $skip_list"
fi

bounds=()
if [[ -n "$lat_min$lat_max$lon_min$lon_max" ]]; then
    [[ -n "$lat_min" && -n "$lat_max" && -n "$lon_min" && -n "$lon_max" ]] || \
        die "Specify all four latitude and longitude bounds, or none."
    for value in "$lat_min" "$lat_max" "$lon_min" "$lon_max"; do
        is_number "$value" || die "Bounds must be numeric: $value"
    done
    awk "BEGIN { exit !($lat_min <= $lat_max && $lon_min <= $lon_max) }" || \
        die "Minimum bounds must not exceed maximum bounds."
    bounds=(-d "lat,$lat_min,$lat_max" -d "lon,$lon_min,$lon_max")
fi

# shellcheck disable=SC1090
source "$conda_sh"
conda activate nco
command -v ncks >/dev/null || die "ncks is unavailable after activating nco."
command -v ncra >/dev/null || die "ncra is unavailable after activating nco."
command -v ncatted >/dev/null || die "ncatted is unavailable after activating nco."
declare -A skipped_paths=()
if [[ -r "$skip_list" ]]; then
    printf 'Using GEOS-IT skip list: %s\n' "$skip_list"
    while IFS= read -r skip_path || [[ -n "$skip_path" ]]; do
        [[ -z "$skip_path" || "$skip_path" == \#* ]] && continue
        skipped_paths["$skip_path"]=1
    done < "$skip_list"
fi

is_listed_skip() {
    [[ -n "${skipped_paths[$1]:-}" ]]
}

tmp_dir=""
cleanup() {
    [[ -z "$tmp_dir" ]] || rm -rf "$tmp_dir"
}
trap cleanup EXIT

processed=0
missing_or_incomplete=0
missing_days=0
skipped_listed=0
day="$start"
while ! [[ "$day" > "$end" ]]; do
    year="${day:0:4}"
    month="${day:5:2}"
    compact_day="${day//-/}"
    source_dir="$source_root/$year/$month"
    output_dir="$output_root/GMAO/GEOSIT/$year/$month"
    output_file="$output_dir/GEOSIT_TOTEXTTAU550_daily.${compact_day}.nc"

    mapfile -t inputs < <(
        find "$source_dir" -maxdepth 1 -type f \
            -name "GEOS.it.asm.aer_tavg_3hr_glo_L576x361_slv.*.${day}T*.nc4" \
            -print 2>/dev/null | sort
    )

    skip_day=false
    for input_file in "${inputs[@]}"; do
        if is_listed_skip "$input_file"; then
            printf 'Skipping listed GEOS-IT source file for %s: %s\n' \
                "$day" "$input_file" >&2
            skip_day=true
            ((skipped_listed += 1))
        fi
    done
    if [[ "$skip_day" == true ]]; then
        ((missing_days += 1))
        if [[ "$allow_missing_day" != true ]]; then
            ((missing_or_incomplete += 1))
        fi
        day="$(date -I -d "$day + 1 day")"
        continue
    fi

    if ((${#inputs[@]} == 0)); then
        printf 'No GEOS-IT AOD inputs for %s\n' "$day" >&2
        ((missing_days += 1))
        if [[ "$allow_missing_day" != true ]]; then
            ((missing_or_incomplete += 1))
        fi
        day="$(date -I -d "$day + 1 day")"
        continue
    fi
    if ((${#inputs[@]} != EXPECTED_DAILY_INPUTS)) && [[ "$allow_incomplete_day" != true ]]; then
        printf 'Expected %d GEOS-IT AOD inputs for %s; found %d.\n' \
            "$EXPECTED_DAILY_INPUTS" "$day" "${#inputs[@]}" >&2
        ((missing_or_incomplete += 1))
        day="$(date -I -d "$day + 1 day")"
        continue
    fi

    mkdir -p "$output_dir"
    [[ ! -e "$output_file" || "$overwrite" == true ]] || \
        die "Output exists (use --overwrite): $output_file"

    tmp_dir="$(mktemp -d "$output_dir/.tmp.geosit-aod.XXXXXX")"
    tmp_output="$tmp_dir/$(basename "$output_file")"
    tmp_files=()
    for index in "${!inputs[@]}"; do
        tmp_file="$tmp_dir/input_${index}.nc"
        ncks -O -4 -L 1 -v "$AOD_VARIABLES" "${bounds[@]}" "${inputs[$index]}" "$tmp_file"
        tmp_files+=("$tmp_file")
    done

    ncra -O -4 -L 1 "${tmp_files[@]}" "$tmp_output"
    ncatted -O \
        -a title,global,o,c,"GEOS-IT daily mean total 550 nm aerosol extinction optical thickness" \
        -a time_coverage_start,global,o,c,"${day}T00:00:00Z" \
        -a time_coverage_end,global,o,c,"${day}T23:59:59Z" \
        -a source_file_count,global,o,c,"${#inputs[@]}" \
        "$tmp_output"
    ncks -M "$tmp_output" >/dev/null
    mv -f "$tmp_output" "$output_file"
    cleanup
    tmp_dir=""

    printf 'Wrote %s from %d GEOS-IT inputs\n' "$output_file" "${#inputs[@]}"
    ((processed += 1))
    day="$(date -I -d "$day + 1 day")"
done

((processed > 0)) || die "No daily GEOS-IT outputs were written."
if ((missing_days > 0)); then
    printf 'Skipped %d date(s) with no GEOS-IT AOD inputs.\n' "$missing_days" >&2
fi
if ((skipped_listed > 0)); then
    printf 'Skipped %d source file(s) listed as missing.\n' "$skipped_listed" >&2
fi
((missing_or_incomplete == 0)) || die "$missing_or_incomplete day(s) were not written."
