from __future__ import annotations

from typing import Any

import mne
import pandas as pd


def attach_trial_metadata(epochs: mne.Epochs, metadata_df: pd.DataFrame) -> mne.Epochs:
    if len(metadata_df) != len(epochs):
        raise ValueError("metadata_df row count must match number of epochs")
    out = epochs.copy()
    out.metadata = metadata_df.reset_index(drop=True)
    return out


def select_condition(epochs: mne.Epochs, condition_expr: str) -> mne.Epochs:
    if epochs.metadata is None:
        raise ValueError("epochs.metadata is required for condition expressions")
    idx = epochs.metadata.query(condition_expr).index.to_numpy()
    return epochs[idx]


def build_contrasts(epochs: mne.Epochs, contrast_defs: dict[str, Any]) -> dict[str, tuple[mne.Epochs, mne.Epochs]]:
    out: dict[str, tuple[mne.Epochs, mne.Epochs]] = {}
    for name, spec in contrast_defs.items():
        a = select_condition(epochs, spec["a"])
        b = select_condition(epochs, spec["b"])
        out[name] = (a, b)
    return out
