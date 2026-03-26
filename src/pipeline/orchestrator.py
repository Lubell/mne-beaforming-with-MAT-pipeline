from __future__ import annotations

from pathlib import Path
from typing import Any
from collections import Counter

import numpy as np
import pandas as pd

from .beamformer import apply_lcmv_to_epochs, build_forward, compute_covariances, make_lcmv_filters
from .conditions import build_contrasts
from .config import PipelineConfig
from .filtering import build_band_dataset
from .io import (
    load_preprocessed_subject,
    load_subject_anatomy,
    save_derivative,
    validate_subject_runtime_inputs,
)
from .stats.runner import apply_multiple_comparison, run_stats


def _prep_epochs(epochs, cfg: PipelineConfig):
    prep = cfg.raw.get("preprocessing", {})
    if prep.get("crop") is not None:
        epochs = epochs.copy().crop(tmin=float(prep["crop"][0]), tmax=float(prep["crop"][1]))
    if prep.get("baseline") is not None:
        epochs = epochs.copy().apply_baseline(tuple(prep["baseline"]))
    if prep.get("sfreq") is not None and abs(float(prep["sfreq"]) - epochs.info["sfreq"]) > 1e-6:
        epochs = epochs.copy().resample(float(prep["sfreq"]))
    return epochs


def _flatten_epochs_for_stats(epochs) -> np.ndarray:
    # n_epochs x (n_channels*n_times), used by scaffold stats tests.
    data = epochs.get_data(copy=True)
    return data.reshape(data.shape[0], -1)


def _stats_summary(stats_out: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for contrast_name, test_map in stats_out.items():
        summary[contrast_name] = {}
        for test_name, result in test_map.items():
            mask = np.asarray(result.meta.get("fdr_mask", np.array([], dtype=bool)))
            summary[contrast_name][test_name] = {
                "n_features": int(np.asarray(result.pvalue).size),
                "n_fdr_significant": int(mask.sum()) if mask.size else 0,
            }
    return summary


def _event_code_summary(epochs) -> dict[str, Any]:
    codes = [int(x) for x in epochs.events[:, 2].tolist()]
    counts = Counter(codes)
    sorted_codes = sorted(counts.keys())
    return {
        "n_unique_codes": len(sorted_codes),
        "codes": sorted_codes,
        "counts": {int(code): int(counts[code]) for code in sorted_codes},
    }


def _event_code_metadata_df(epochs) -> pd.DataFrame:
    event_codes = [int(x) for x in epochs.events[:, 2].tolist()]

    category_map: dict[int, str] = {
        210: "nat", 213: "nat", 217: "nat",
        230: "nat", 233: "nat", 237: "nat",
        260: "nat", 263: "nat", 267: "nat",
        520: "lin", 523: "lin", 527: "lin",
        540: "lin", 543: "lin", 547: "lin",
        550: "lin", 553: "lin", 557: "lin",
    }
    filter_map: dict[int, str] = {
        210: "unfilt", 230: "unfilt", 260: "unfilt",
        213: "lp", 233: "lp", 263: "lp",
        217: "hp", 237: "hp", 267: "hp",
        520: "unfilt", 540: "unfilt", 550: "unfilt",
        523: "lp", 543: "lp", 553: "lp",
        527: "hp", 547: "hp", 557: "hp",
    }

    metadata = pd.DataFrame({"event_code": event_codes})
    metadata["category"] = [category_map.get(code, "unknown") for code in event_codes]
    metadata["filter"] = [filter_map.get(code, "unknown") for code in event_codes]
    metadata["is_known_code"] = metadata["category"].ne("unknown") & metadata["filter"].ne("unknown")
    return metadata


def run_subject(subject_id: str, cfg: PipelineConfig, inspect_event_codes: bool = False) -> dict[str, Any]:
    beam_cfg = cfg.raw.get("beamformer", {})
    validate_subject_runtime_inputs(subject_id, cfg, beamformer_enabled=bool(beam_cfg.get("enabled", False)))

    epochs = load_preprocessed_subject(subject_id, cfg)
    epochs = _prep_epochs(epochs, cfg)

    inspection_cfg = cfg.raw.get("inspection", {})
    inspect_only = bool(inspection_cfg.get("event_codes_only", False)) or bool(inspect_event_codes)
    event_codes = _event_code_summary(epochs)

    if inspect_only:
        save_derivative(event_codes, subject_id, "event_codes", cfg)
        return {
            "subject": subject_id,
            "saved": str(Path(cfg.output_root) / subject_id),
            "n_epochs": len(epochs),
            "inspection_only": True,
            "event_codes": event_codes,
        }

    derived_metadata = _event_code_metadata_df(epochs)
    if epochs.metadata is None:
        # Fallback metadata with condition labels derived from event codes.
        epochs.metadata = derived_metadata
    else:
        # Preserve existing metadata while ensuring standard condition columns exist.
        merged = epochs.metadata.reset_index(drop=True).copy()
        for col in derived_metadata.columns:
            if col not in merged.columns:
                merged[col] = derived_metadata[col].values
        epochs.metadata = merged

    band_ds = build_band_dataset(epochs, cfg.raw["filtering"])
    save_derivative(band_ds["unfiltered"], subject_id, "filt_info", cfg)
    for name, ep in band_ds["bands"].items():
        save_derivative(ep, subject_id, f"filt_{name}", cfg)
    for name, ep in band_ds["hilbert"].items():
        save_derivative(ep, subject_id, f"filt_{name}_HB", cfg)

    contrasts = {}
    contrast_defs = cfg.raw.get("conditions", {}).get("contrasts", {})
    if epochs.metadata is not None and contrast_defs:
        contrasts = build_contrasts(epochs, contrast_defs)

    stats_out: dict[str, Any] = {}
    tests = cfg.raw.get("stats", {}).get("tests", [])
    for contrast_name, (a_ep, b_ep) in contrasts.items():
        data_a = _flatten_epochs_for_stats(a_ep)
        data_b = _flatten_epochs_for_stats(b_ep)
        raw_results = run_stats(tests, data_a, data_b, cfg.raw.get("stats", {}))
        stats_out[contrast_name] = {
            test_name: apply_multiple_comparison(res, cfg.raw.get("stats", {}))
            for test_name, res in raw_results.items()
        }

    stats_summary = _stats_summary(stats_out)
    if stats_summary:
        save_derivative(stats_summary, subject_id, "stats_summary", cfg)

    beam_out: dict[str, Any] = {}
    if beam_cfg.get("enabled", False):
        anatomy = load_subject_anatomy(subject_id, cfg)
        forward = build_forward(epochs, anatomy)
        covs = compute_covariances(epochs, beam_cfg)
        filters = make_lcmv_filters(epochs, forward, covs, beam_cfg)
        stcs = apply_lcmv_to_epochs(epochs, filters)
        beam_out["n_stcs"] = len(stcs)
        beam_out["forward_nsource"] = int(forward["nsource"])

    report = {
        "subject": subject_id,
        "saved": str(Path(cfg.output_root) / subject_id),
        "n_epochs": len(epochs),
        "bands": list(band_ds["bands"].keys()),
        "contrast_count": len(contrasts),
        "stats_tests": tests,
        "stats_ready": bool(contrasts and tests),
        "stats": stats_summary,
        "event_codes": event_codes,
        "beamformer": beam_out,
    }
    return report
