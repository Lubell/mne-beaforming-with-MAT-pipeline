from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


SUPPORTED_STATS_TESTS = {
    "ttest_ind",
    "rank_sum",
    "permutation",
    "t2circ",
    "watson_williams",
}
EXPERIMENTAL_STATS_TESTS = {"t2circ", "watson_williams"}


@dataclass(frozen=True)
class PipelineConfig:
    raw: dict[str, Any]

    @property
    def data_root(self) -> Path:
        return Path(self.raw["data_root"]).expanduser().resolve()

    @property
    def output_root(self) -> Path:
        return Path(self.raw["output_root"]).expanduser().resolve()

    @property
    def subjects(self) -> list[str]:
        return list(self.raw.get("subjects", []))


def validate_config(raw_cfg: dict[str, Any]) -> None:
    required = ["data_root", "output_root", "input", "filtering", "stats"]
    missing = [k for k in required if k not in raw_cfg]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    input_cfg = raw_cfg["input"]
    input_required = ["preprocessed_dir", "mri_dir", "preprocessed_extensions"]
    input_missing = [k for k in input_required if k not in input_cfg]
    if input_missing:
        raise ValueError(f"Missing required input config keys: {input_missing}")

    filtering = raw_cfg["filtering"]
    bands = filtering.get("bands", [])
    if not bands:
        raise ValueError("filtering.bands must define at least one band")
    for band in bands:
        if not all(key in band for key in ["name", "fmin", "fmax"]):
            raise ValueError(f"Invalid band entry: {band}")

    stats_cfg = raw_cfg["stats"]
    tests = stats_cfg.get("tests", [])
    if not isinstance(tests, list):
        raise ValueError("stats.tests must be a list of test names")
    unknown = [t for t in tests if t not in SUPPORTED_STATS_TESTS]
    if unknown:
        raise ValueError(f"Unknown stats tests in config: {unknown}")

    allow_experimental = bool(stats_cfg.get("allow_experimental", False))
    blocked = [t for t in tests if t in EXPERIMENTAL_STATS_TESTS and not allow_experimental]
    if blocked:
        raise ValueError(
            "Experimental stats tests requested but disabled. "
            f"Set stats.allow_experimental=true to enable: {blocked}"
        )


def load_config(path: str | Path) -> PipelineConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)
    if not isinstance(raw_cfg, dict):
        raise ValueError("Config must be a YAML mapping/object")
    validate_config(raw_cfg)
    return PipelineConfig(raw=raw_cfg)


def resolve_subjects(cfg: PipelineConfig, requested_subject: str | None = None) -> list[str]:
    if requested_subject:
        return [requested_subject]
    subjects = cfg.subjects
    if not subjects:
        raise ValueError("No subjects configured and no requested subject provided")
    return subjects
