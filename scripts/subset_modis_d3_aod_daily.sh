#!/usr/bin/env bash
# Subset MODIS C6.1 daily 1-degree combined Dark Target/Deep Blue AOD.

set -euo pipefail

readonly DEFAULT_SOURCE_ROOT="/ASDC_archive/MODIS"
readonly DEFAULT_OUTPUT_ROOT="/CERES/sarb/dfillmor/DAVINCI"
readonly DEFAULT_HDF4_NCKS="/usr/local/bin/ncks"
readonly D3_VARIABLE="AOD_550_Dark_Target_Deep_Blue_Combined_Mean"
readonly D3_COORDINATES="XDim,YDim"

usage() {
    cat <<'EOF'
Usage: subset_modis_d3_aod_daily.sh --start YYYY-MM-DD --end YYYY-MM-DD [options]

Subset MOD08_D3 and MYD08_D3 C6.1 daily 1-degree AOD files when available.

Required:
  --start DATE          First UTC date, inclusive.
  --end DATE            Last UTC date, inclusive.

Optional:
  --platform VALUE      terra, aqua, or both. Default: both.
  --collection NAME     Source collection directory. Default: C61.
  --lat-min VALUE       Southern latitude cell-center bound. Requires all four bounds.
  --lat-max VALUE       Northern latitude cell-center bound. Requires all four bounds.
  --lon-min VALUE       Western longitude cell-center bound. Requires all four bounds.
  --lon-max VALUE       Eastern longitude cell-center bound. Requires all four bounds.
  --row-start INDEX     First YDim:mod08 row. Requires all four index bounds.
  --row-end INDEX       Last YDim:mod08 row. Requires all four index bounds.
  --col-start INDEX     First XDim:mod08 column. Requires all four index bounds.
  --col-end INDEX       Last XDim:mod08 column. Requires all four index bounds.
  --source-root PATH    Default: /ASDC_archive/MODIS
  --output-root PATH    Default: /CERES/sarb/dfillmor/DAVINCI
  --ncks-hdf4 PATH      HDF4-capable ncks reader. Defaults to /usr/local/bin/ncks
                        when available, otherwise the activated nco ncks.
  --overwrite           Replace an existing output file.
  --conda-sh PATH       Default: $HOME/miniforge3/etc/profile.d/conda.sh
  -h, --help            Show this help text.

Without geographic or index bounds, the script writes the global 1-degree
field. Geographic bounds select cell centers in the D3 grid: XDim spans -179.5
to 179.5 and YDim spans 89.5 to -89.5. Rows and columns are zero-based D3
indices and cannot be combined with geographic bounds.
EOF
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

is_nonnegative_integer() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

is_number() {
    [[ "$1" =~ ^-?([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]
}

source_root="$DEFAULT_SOURCE_ROOT"
output_root="$DEFAULT_OUTPUT_ROOT"
conda_sh="${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
ncks_hdf4="${NCKS_HDF4_BIN:-}"
start=""
end=""
platform="both"
collection="C61"
lat_min=""
lat_max=""
lon_min=""
lon_max=""
row_start=""
row_end=""
col_start=""
col_end=""
overwrite=false

while (($# > 0)); do
    case "$1" in
        --start|--end|--platform|--collection|--lat-min|--lat-max|--lon-min|--lon-max|--row-start|--row-end|--col-start|--col-end|--source-root|--output-root|--ncks-hdf4|--conda-sh)
            (($# >= 2)) || die "Missing value for $1"
            case "$1" in
                --start) start="$2" ;;
                --end) end="$2" ;;
                --platform) platform="$2" ;;
                --collection) collection="$2" ;;
                --lat-min) lat_min="$2" ;;
                --lat-max) lat_max="$2" ;;
                --lon-min) lon_min="$2" ;;
                --lon-max) lon_max="$2" ;;
                --row-start) row_start="$2" ;;
                --row-end) row_end="$2" ;;
                --col-start) col_start="$2" ;;
                --col-end) col_end="$2" ;;
                --source-root) source_root="$2" ;;
                --output-root) output_root="$2" ;;
                --ncks-hdf4) ncks_hdf4="$2" ;;
                --conda-sh) conda_sh="$2" ;;
            esac
            shift 2
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

case "$platform" in
    terra) platforms=("Terra") ;;
    aqua) platforms=("Aqua") ;;
    both) platforms=("Terra" "Aqua") ;;
    *) die "--platform must be terra, aqua, or both." ;;
esac

has_explicit_index_bounds=false
if [[ -n "$row_start$row_end$col_start$col_end" ]]; then
    has_explicit_index_bounds=true
fi

has_geographic_bounds=false
if [[ -n "$lat_min$lat_max$lon_min$lon_max" ]]; then
    has_geographic_bounds=true
    [[ "$has_explicit_index_bounds" != true ]] || \
        die "Geographic bounds cannot be combined with row and column bounds."
    [[ -n "$lat_min" && -n "$lat_max" && -n "$lon_min" && -n "$lon_max" ]] || \
        die "Specify all four latitude and longitude bounds, or none."
    for value in "$lat_min" "$lat_max" "$lon_min" "$lon_max"; do
        is_number "$value" || die "Bounds must be numeric: $value"
    done
    awk "BEGIN { exit !($lat_min <= $lat_max && $lon_min <= $lon_max && $lat_min >= -89.5 && $lat_max <= 89.5 && $lon_min >= -179.5 && $lon_max <= 179.5) }" || \
        die "Bounds must be ordered and within the D3 cell-center ranges."
    read -r row_start row_end col_start col_end < <(
        awk -v south="$lat_min" -v north="$lat_max" -v west="$lon_min" -v east="$lon_max" '
            function ceil(value, whole) {
                whole = int(value)
                return value > whole ? whole + 1 : whole
            }
            BEGIN {
                printf "%d %d %d %d\n", ceil(89.5 - north), int(89.5 - south), ceil(west + 179.5), int(east + 179.5)
            }
        '
    )
fi

has_index_bounds=false
if [[ -n "$row_start$row_end$col_start$col_end" ]]; then
    has_index_bounds=true
    [[ -n "$row_start" && -n "$row_end" && -n "$col_start" && -n "$col_end" ]] || \
        die "Specify all four row and column bounds, or none."
    for value in "$row_start" "$row_end" "$col_start" "$col_end"; do
        is_nonnegative_integer "$value" || die "Grid indices must be nonnegative integers: $value"
    done
    ((row_start <= row_end && col_start <= col_end)) || \
        die "Start indices must not exceed end indices."
    ((row_end < 180 && col_end < 360)) || \
        die "D3 grid indices must be within 180 rows and 360 columns."
fi

slice_bounds=()
if [[ "$has_index_bounds" == true ]]; then
    slice_bounds=(-d "YDim:mod08,$row_start,$row_end" -d "XDim:mod08,$col_start,$col_end")
fi

# shellcheck disable=SC1090
source "$conda_sh"
conda activate nco
ncks_bin="$(command -v ncks)"
if [[ -z "$ncks_hdf4" ]]; then
    if [[ -x "$DEFAULT_HDF4_NCKS" ]]; then
        ncks_hdf4="$DEFAULT_HDF4_NCKS"
    else
        ncks_hdf4="$ncks_bin"
    fi
fi
[[ -x "$ncks_hdf4" ]] || die "HDF4 ncks reader is not executable: $ncks_hdf4"
printf 'Using HDF4 NCO reader: %s\n' "$ncks_hdf4"

tmp_dir=""
cleanup() {
    [[ -z "$tmp_dir" ]] || rm -rf "$tmp_dir"
}
trap cleanup EXIT

processed=0
day="$start"
while ! [[ "$day" > "$end" ]]; do
    year="${day:0:4}"
    day_of_year="$(date -d "$day" +%j)"

    for current_platform in "${platforms[@]}"; do
        case "$current_platform" in
            Terra) prefix="MOD08_D3" ;;
            Aqua) prefix="MYD08_D3" ;;
        esac

        source_dir="$source_root/$current_platform/$collection/$year/$day_of_year"
        mapfile -t inputs < <(
            find "$source_dir" -maxdepth 1 -type f \
                -name "${prefix}.A${year}${day_of_year}.*.hdf" \
                -print 2>/dev/null | sort
        )

        if ((${#inputs[@]} == 0)); then
            printf 'No %s source file for %s; skipping.\n' "$current_platform" "$day" >&2
            continue
        fi
        if ((${#inputs[@]} > 1)); then
            die "Multiple ${prefix} files found for $day; resolve the source directory before subsetting."
        fi

        input_file="${inputs[0]}"
        relative_dir="${input_file#"$source_root"/}"
        relative_dir="${relative_dir%/*}"
        output_dir="$output_root/MODIS/$relative_dir"
        output_name="$(basename "${input_file%.hdf}").nc"
        output_file="$output_dir/$output_name"

        mkdir -p "$output_dir"
        [[ ! -e "$output_file" || "$overwrite" == true ]] || \
            die "Output exists (use --overwrite): $output_file"

        tmp_dir="$(mktemp -d "$output_dir/.tmp.modis-d3.XXXXXX")"
        tmp_output="$tmp_dir/$output_name"

        "$ncks_hdf4" -m -v "$D3_VARIABLE,$D3_COORDINATES" "$input_file" >/dev/null
        "$ncks_hdf4" -O -4 -L 1 \
            -v "$D3_VARIABLE,$D3_COORDINATES" \
            "${slice_bounds[@]}" \
            "$input_file" "$tmp_output"
        ncks -M "$tmp_output" >/dev/null
        mv -f "$tmp_output" "$output_file"
        cleanup
        tmp_dir=""

        printf 'Wrote %s\n' "$output_file"
        ((processed += 1))
    done

    day="$(date -I -d "$day + 1 day")"
done

((processed > 0)) || die "No MOD08_D3 or MYD08_D3 files were found for the requested range."
