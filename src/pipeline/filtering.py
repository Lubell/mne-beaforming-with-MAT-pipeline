from __future__ import annotations

from typing import Any

import mne


def apply_band_filter(epochs: mne.Epochs, band: dict[str, Any], cfg: dict[str, Any]) -> mne.Epochs:
    method = cfg.get("method", "fir")
    fir_design = cfg.get("fir_design", "firwin")

    filtered = epochs.copy().filter(
        l_freq=float(band["fmin"]),
        h_freq=float(band["fmax"]),
        method=method,
        fir_design=fir_design,
        verbose=False,
    )
    return filtered


def compute_hilbert_complex(epochs: mne.Epochs) -> mne.Epochs:
    out = epochs.copy()
    out.apply_hilbert(envelope=False, verbose=False)
    return out


def build_band_dataset(epochs: mne.Epochs, filtering_cfg: dict[str, Any]) -> dict[str, Any]:
    bands = filtering_cfg["bands"]
    apply_hilbert = bool(filtering_cfg.get("apply_hilbert", True))

    dataset: dict[str, Any] = {
        "unfiltered": epochs.copy(),
        "bands": {},
        "hilbert": {},
    }

    for band in bands:
        name = str(band["name"])
        band_epochs = apply_band_filter(epochs, band, filtering_cfg)
        dataset["bands"][name] = band_epochs
        if apply_hilbert:
            dataset["hilbert"][name] = compute_hilbert_complex(band_epochs)

    return dataset
