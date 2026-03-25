#!/usr/bin/env python3
"""Run one subject through the scaffold pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipeline.config import load_config
from pipeline.orchestrator import run_subject


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one subject pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--subject", required=True, help="Subject ID")
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
        result = run_subject(args.subject, cfg)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"Pipeline startup failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    print(result)


if __name__ == "__main__":
    main()
