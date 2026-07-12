#!/usr/bin/env python3
"""Local entry point for the APS360 progress-report pipeline."""

from pathlib import Path

from pipeline import run_pipeline


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    output = run_pipeline(project_root, max_companies=250)
    stats = output["stats"]
    results = output["results"]
    print("Dataset:", stats)
    print("Results:", results)


if __name__ == "__main__":
    main()
