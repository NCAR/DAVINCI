#!/usr/bin/env python
"""Run all DAVINCI plot examples.

This script executes all individual plot examples in sequence,
generating the complete set of demonstration plots.
"""

import subprocess
import sys
import time
from pathlib import Path


def run_example(script_path: Path) -> tuple[bool, float]:
    """Run a single example script.

    Returns
    -------
    tuple[bool, float]
        (success, elapsed_seconds)
    """
    start = time.time()
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=script_path.parent,
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}")
        return False, elapsed

    return True, elapsed


def main():
    """Run all plot examples."""
    print("=" * 60)
    print("DAVINCI: Running All Plot Examples")
    print("=" * 60)

    examples_dir = Path(__file__).parent
    scripts = sorted(examples_dir.glob("plot_*.py"))

    if not scripts:
        print("No plot_*.py scripts found!")
        return

    print(f"\nFound {len(scripts)} example scripts\n")

    total_time = 0.0
    successes = 0
    failures = []

    for script in scripts:
        name = script.stem
        print(f"Running {name}...")

        success, elapsed = run_example(script)
        total_time += elapsed

        if success:
            successes += 1
            print(f"  Completed in {elapsed:.1f}s\n")
        else:
            failures.append(name)
            print(f"  FAILED after {elapsed:.1f}s\n")

    print("=" * 60)
    print(f"Results: {successes}/{len(scripts)} succeeded")
    print(f"Total time: {total_time:.1f}s")

    if failures:
        print(f"\nFailed scripts:")
        for name in failures:
            print(f"  - {name}")

    output_dir = examples_dir / "output" / "plots"
    if output_dir.exists():
        n_plots = len(list(output_dir.glob("*.png")))
        print(f"\nOutput: {n_plots} PNG files in {output_dir}")

    print("=" * 60)


if __name__ == "__main__":
    main()
