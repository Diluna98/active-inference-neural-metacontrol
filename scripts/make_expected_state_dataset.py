"""Build observation-marginalized MOS state labels from a research archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from active_inference_neural_metacontrol.expected_state_labels import (
    build_expected_state_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = build_expected_state_dataset(args.source_dir, args.output_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
