from __future__ import annotations

from typing import Any

import mne
from mne.beamformer import apply_lcmv_epochs, make_lcmv


def build_forward(epochs: mne.Epochs, anatomy: dict[str, Any]) -> mne.Forward:
    required = ["trans", "src", "bem"]
    missing = [k for k in required if k not in anatomy]
    if missing:
        raise ValueError(f"Anatomy missing required fields: {missing}")

    return mne.make_forward_solution(
        info=epochs.info,
        trans=anatomy["trans"],
        src=anatomy["src"],
        bem=anatomy["bem"],
        meg=True,
        eeg=False,
        verbose=False,
    )


def compute_covariances(epochs: mne.Epochs, beam_cfg: dict[str, Any]) -> dict[str, mne.Covariance]:
    noise_tmin, noise_tmax = beam_cfg.get("noise_cov_window", [-0.2, 0.0])
    data_tmin, data_tmax = beam_cfg.get("data_cov_window", [-0.2, 0.398])

    noise_cov = mne.compute_covariance(epochs, tmin=noise_tmin, tmax=noise_tmax, method="empirical", verbose=False)
    data_cov = mne.compute_covariance(epochs, tmin=data_tmin, tmax=data_tmax, method="empirical", verbose=False)
    return {"noise_cov": noise_cov, "data_cov": data_cov}


def make_lcmv_filters(
    epochs: mne.Epochs,
    forward: mne.Forward,
    covs: dict[str, mne.Covariance],
    beam_cfg: dict[str, Any],
):
    return make_lcmv(
        info=epochs.info,
        forward=forward,
        data_cov=covs["data_cov"],
        noise_cov=covs["noise_cov"],
        reg=float(beam_cfg.get("reg", 0.05)),
        pick_ori=beam_cfg.get("pick_ori", "max-power"),
        weight_norm=beam_cfg.get("weight_norm", "nai"),
        verbose=False,
    )


def apply_lcmv_to_epochs(epochs: mne.Epochs, filters):
    return apply_lcmv_epochs(epochs, filters, return_generator=False, verbose=False)
