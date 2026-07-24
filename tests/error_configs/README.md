# Error Handling Test Configs

These YAML files intentionally contain errors to test the pipeline's error handling.

## Usage

```bash
cd /Users/fillmore/EarthSystem/DAVINCI
conda activate davinci

# Test individual configs
davinci run tests/error_configs/01_malformed_yaml.yaml
davinci run tests/error_configs/02_bad_indentation.yaml
# ... etc

# Or run them all and capture output
for f in tests/error_configs/*.yaml; do
  echo "=== Testing: $f ==="
  davinci run "$f" 2>&1 | head -20
  echo
done
```

## Test Cases

| File | Error Type | Expected Behavior |
|------|-----------|-------------------|
| 01_malformed_yaml.yaml | Unclosed quote | YAML parse error |
| 02_bad_indentation.yaml | Bad indentation | YAML parse error |
| 03_missing_analysis.yaml | Missing required section | Validation error |
| 04_missing_dataset_files.yaml | Files don't exist | DataNotFoundError |
| 05_invalid_dataset_type.yaml | Unknown type | Registry/validation error |
| 06_invalid_geometry_type.yaml | Unknown type | Registry/validation error |
| 07_invalid_dates.yaml | Unparseable dates | Validation error |
| 08_end_before_start.yaml | End < start date | Validation error |
| 09_type_mismatch.yaml | String where number expected | Validation error |
| 10_invalid_pair_ref.yaml | Geometry to undefined dataset/geometry | Validation error |
| 11_invalid_plot_type.yaml | Unknown plot type | Registry/validation error |
| 12_empty_file.yaml | Empty config | Validation error |
| 13_corrupt_env_var.yaml | Undefined env var | Expansion error or file not found |
| 14_duplicate_keys.yaml | Duplicate YAML keys | Silent overwrite or warning |
| 15_null_values.yaml | Null where string needed | Validation error |

## Expected Output

Each test should produce:
1. A clear, user-friendly error message
2. An error log file in `logs/error_*.log` with full traceback (for I/O errors)
3. Non-zero exit code

## Checking Error Logs

After running tests, check for detailed error logs:
```bash
ls -la logs/error_*.log
cat logs/error_*.log
```
