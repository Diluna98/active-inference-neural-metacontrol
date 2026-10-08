"""Compare state-inference accuracy, complexity, and VFE across MOS resolutions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .resolution_vfe_diagnostic import (
    ResolutionVFEDiagnosticConfig,
    evaluate_resolution_vfe_diagnostic,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-resolution", type=int, choices=(2, 5, 10, 20), default=10)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--message-passing-iterations", type=int, default=10)
    parser.add_argument("--policy-workers", type=int, default=1)
    args = parser.parse_args()
    report = evaluate_resolution_vfe_diagnostic(
        args.instance_seed,
        output_dir=args.output_dir,
        config=ResolutionVFEDiagnosticConfig(
            reference_resolution=args.reference_resolution,
            max_steps=args.max_steps,
            message_passing_iterations=args.message_passing_iterations,
            policy_workers=args.policy_workers,
        ),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
