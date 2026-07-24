#!/usr/bin/env python
"""
Run a DAVINCI analysis pipeline.

This script provides a simple way to run any analysis given a YAML config file.
It uses the full pipeline runner with progress bars and Markdown logging.

Usage:
    python scripts/run_analysis.py path/to/config.yaml

Or make it executable:
    chmod +x scripts/run_analysis.py
    ./scripts/run_analysis.py path/to/config.yaml

You can also use the CLI:
    davinci run path/to/config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    """Run the analysis pipeline."""
    parser = argparse.ArgumentParser(
        description="Run a DAVINCI analysis pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/run_analysis.py analyses/asia-aq/configs/asia-aq-airnow.example.yaml
    python scripts/run_analysis.py my_analysis/config.yaml --quiet

Environment variables in config paths (e.g., ${MY_DATA}/file.nc) are
automatically expanded. Set them before running:
    export MY_DATA=/path/to/data
    python scripts/run_analysis.py config.yaml
        """,
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress bar output.",
    )

    args = parser.parse_args()

    # Validate config file
    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        return 1

    # Print header
    print("=" * 70)
    print("DAVINCI Analysis Pipeline")
    print("=" * 70)
    print(f"\nConfig: {config_path.absolute()}")
    print()

    # Run the pipeline
    from davinci_monet.pipeline.runner import run_analysis

    result = run_analysis(str(config_path), show_progress=not args.quiet)

    # Report results
    print()
    print("=" * 70)
    if result.success:
        print("Analysis completed successfully!")
        print(f"Total time: {result.total_duration_seconds:.1f} seconds")
        print(f"Stages: {', '.join(result.completed_stages)}")
    else:
        print("Analysis failed!")
        for failed in result.failed_stages:
            print(f"  {failed.stage_name}: {failed.error}")
        return 1

    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
