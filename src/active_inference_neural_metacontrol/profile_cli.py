"""Build a controlled metacontrol timing profile from generated data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .profiling import build_timing_profile, save_timing_profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-parallel-generation",
        action="store_true",
        help="accept noisy latencies collected with multiple instance workers",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = build_timing_profile(
        args.dataset_dir,
        allow_parallel_generation=args.allow_parallel_generation,
    )
    save_timing_profile(profile, args.output)
    print(json.dumps(profile, indent=2))
