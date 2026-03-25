from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mne
import numpy as np
from scipy.io import loadmat

from .config import PipelineConfig


@dataclass(frozen=True)
class SubjectContext:
    subject_id: str
    preprocessed_file: Path
    anatomy_files: dict[str, Path]


def validate_subject_runtime_inputs(subject_id: str, cfg: PipelineConfig, beamformer_enabled: bool = False) -> None:
    input_cfg = cfg.raw["input"]
    pre_dir = cfg.data_root / input_cfg["preprocessed_dir"]
    if not pre_dir.exists():
        raise FileNotFoundError(
            f"Preprocessed directory does not exist: {pre_dir}. "
            "Update data_root/input.preprocessed_dir in config."
        )

    _find_subject_preprocessed(subject_id, cfg)

    if not beamformer_enabled:
        return

    mri_cfg = input_cfg.get("anatomy_files", {})
    required_anatomy = ["trans", "bem", "src"]
    missing_keys = [k for k in required_anatomy if k not in mri_cfg]
    if missing_keys:
        raise ValueError(f"Missing input.anatomy_files keys for beamformer mode: {missing_keys}")

    mri_base = cfg.data_root / input_cfg["mri_dir"] / subject_id
    missing_files = [str(mri_base / mri_cfg[k]) for k in required_anatomy if not (mri_base / mri_cfg[k]).exists()]
    if missing_files:
        raise FileNotFoundError(
            "Beamformer mode is enabled but anatomy files are missing: "
            + ", ".join(missing_files)
        )


def _find_subject_preprocessed(subject_id: str, cfg: PipelineConfig) -> Path:
    input_cfg = cfg.raw["input"]
    pre_dir = cfg.data_root / input_cfg["preprocessed_dir"]
    extensions = input_cfg.get("preprocessed_extensions", [".fif", ".mat"])
    candidates: list[Path] = []
    for ext in extensions:
        candidates.extend(sorted(pre_dir.glob(f"{subject_id}*{ext}")))
    if not candidates:
        raise FileNotFoundError(f"No preprocessed file found for {subject_id} in {pre_dir}")
    return candidates[0]


def discover_subject_inputs(cfg: PipelineConfig) -> list[SubjectContext]:
    out: list[SubjectContext] = []
    mri_cfg = cfg.raw["input"]["anatomy_files"]
    mri_base = cfg.data_root / cfg.raw["input"]["mri_dir"]

    for subject_id in cfg.subjects:
        pre = _find_subject_preprocessed(subject_id, cfg)
        anatomy = {
            "trans": mri_base / subject_id / mri_cfg["trans"],
            "bem": mri_base / subject_id / mri_cfg["bem"],
            "src": mri_base / subject_id / mri_cfg["src"],
        }
        out.append(SubjectContext(subject_id=subject_id, preprocessed_file=pre, anatomy_files=anatomy))
    return out


def load_preprocessed_subject(subject_id: str, cfg: PipelineConfig) -> mne.Epochs:
    pre = _find_subject_preprocessed(subject_id, cfg)
    if pre.suffix == ".fif":
        return mne.read_epochs(pre, preload=True)
    if pre.suffix == ".mat":
        mat_schema = cfg.raw.get("input", {}).get("mat_schema", {})
        return _load_mat_epochs(pre, mat_schema)
    raise ValueError(f"Unsupported preprocessed extension: {pre.suffix}")


def _load_mat_epochs(path: Path, schema: dict[str, Any] | None = None) -> mne.EpochsArray:
    schema = schema or {}
    try:
        mat = loadmat(path, simplify_cells=True)
        return _load_mat_epochs_from_schema(mat, path, schema)
    except NotImplementedError:
        return _load_mat_v73_fieldtrip(path, schema)


def _get_required_field(mat: dict[str, Any], key: str, path: Path) -> Any:
    if key not in mat:
        raise KeyError(f"Missing MAT field '{key}' in {path}")
    return mat[key]


def _coerce_channel_types(ch_types_in: Any, n_channels: int) -> list[str] | str:
    if isinstance(ch_types_in, str):
        return ch_types_in
    if isinstance(ch_types_in, (list, tuple, np.ndarray)):
        ch_types = [str(x) for x in ch_types_in]
        if len(ch_types) != n_channels:
            raise ValueError(
                f"ch_types length ({len(ch_types)}) does not match n_channels ({n_channels})"
            )
        return ch_types
    raise ValueError("ch_types must be a string or sequence of strings")


def _infer_channel_types_from_names(ch_names: list[str], default_meg: str = "mag") -> list[str]:
    out: list[str] = []
    for name in ch_names:
        up = name.upper()
        if up.startswith("EOG"):
            out.append("eog")
            continue
        if up.startswith("ECG"):
            out.append("ecg")
            continue
        if up.startswith("STI"):
            out.append("stim")
            continue
        if up.startswith("MEG"):
            # Neuromag convention: ...1 magnetometer, ...2/3 gradiometer
            if up[-1:] == "1":
                out.append("mag")
            elif up[-1:] in {"2", "3"}:
                out.append("grad")
            else:
                out.append(default_meg)
            continue
        out.append("misc")
    return out


def _decode_h5_char_dataset(ds: Any) -> str:
    arr = np.asarray(ds).astype("uint16").ravel()
    return "".join(chr(int(x)) for x in arr if int(x) != 0)


def _load_mat_v73_fieldtrip(path: Path, schema: dict[str, Any]) -> mne.EpochsArray:
    import h5py

    root_key = schema.get("fieldtrip_root_key", "data")
    default_meg = str(schema.get("default_meg_type", "mag"))
    dtype = np.dtype(schema.get("dtype", "float32"))
    max_trials = schema.get("max_trials")

    with h5py.File(path, "r") as f:
        if root_key not in f:
            raise ValueError(
                f"MAT v7.3 file does not contain root group '{root_key}': {path}"
            )
        grp = f[root_key]
        required = ["trial", "time", "label", "fsample"]
        missing = [k for k in required if k not in grp]
        if missing:
            raise ValueError(
                f"FieldTrip group '{root_key}' missing required datasets {missing} in {path}"
            )

        label_refs = np.asarray(grp["label"]).ravel()
        ch_names = [_decode_h5_char_dataset(f[ref]) for ref in label_refs]
        n_channels = len(ch_names)

        trial_refs = np.asarray(grp["trial"]).ravel()
        time_refs = np.asarray(grp["time"]).ravel()
        if max_trials is not None:
            max_trials = int(max_trials)
            trial_refs = trial_refs[:max_trials]
            time_refs = time_refs[:max_trials]
        if len(trial_refs) == 0:
            raise ValueError(f"No trial data found in {path}")
        if len(time_refs) != len(trial_refs):
            raise ValueError(
                f"Mismatch between number of trial and time cells in {path}: {len(trial_refs)} vs {len(time_refs)}"
            )

        trial_stack = []
        tmin = None
        n_times = None
        for trial_ref, time_ref in zip(trial_refs, time_refs):
            arr = np.asarray(f[trial_ref], dtype=dtype)
            tvec = np.asarray(f[time_ref], dtype=float).ravel()
            if tmin is None:
                tmin = float(tvec[0])
            if n_times is None:
                n_times = int(tvec.shape[0])
            if arr.ndim != 2:
                raise ValueError(f"Trial cell must be 2D, got {arr.shape} in {path}")
            # FieldTrip trial cell is channels x times in MATLAB; in HDF5 export this may appear transposed.
            if arr.shape == (n_times, n_channels):
                arr = arr.T
            elif arr.shape == (n_channels, n_times):
                pass
            else:
                raise ValueError(
                    f"Unexpected trial shape {arr.shape}; expected ({n_channels}, {n_times}) or ({n_times}, {n_channels})"
                )
            trial_stack.append(arr)

        data = np.stack(trial_stack, axis=0).astype(dtype, copy=False)
        sfreq = float(np.asarray(grp["fsample"]).ravel()[0])

        if "ch_types" in schema:
            ch_types = _coerce_channel_types(schema["ch_types"], n_channels)
        else:
            ch_types = _infer_channel_types_from_names(ch_names, default_meg=default_meg)

        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

        if "trialinfo" in grp:
            trialinfo = np.asarray(grp["trialinfo"]).ravel()
            event_codes = trialinfo.astype(int)[: len(trial_stack)]
            events = np.column_stack(
                [
                    np.arange(len(trial_stack), dtype=int),
                    np.zeros(len(trial_stack), dtype=int),
                    event_codes,
                ]
            )
            unique_codes = sorted(set(int(x) for x in event_codes.tolist()))
            event_id = {f"code_{code}": int(code) for code in unique_codes}
        else:
            events = np.column_stack(
                [
                    np.arange(len(trial_stack), dtype=int),
                    np.zeros(len(trial_stack), dtype=int),
                    np.ones(len(trial_stack), dtype=int),
                ]
            )
            event_id = {"default": 1}

        return mne.EpochsArray(data, info, events=events, tmin=float(tmin), event_id=event_id, verbose=False)


def _load_mat_epochs_from_schema(mat: dict[str, Any], path: Path, schema: dict[str, Any]) -> mne.EpochsArray:
    # Default schema:
    # data: (n_epochs, n_channels, n_times)
    # sfreq: float
    # ch_names: sequence[str]
    # optional: tmin, ch_types, events, event_id
    data_key = schema.get("data_key", "data")
    sfreq_key = schema.get("sfreq_key", "sfreq")
    ch_names_key = schema.get("ch_names_key", "ch_names")
    tmin_key = schema.get("tmin_key", "tmin")
    ch_types_key = schema.get("ch_types_key", "ch_types")
    events_key = schema.get("events_key", "events")
    event_id_key = schema.get("event_id_key", "event_id")

    dtype = np.dtype(schema.get("dtype", "float64"))
    data = np.asarray(_get_required_field(mat, data_key, path), dtype=dtype)
    if data.ndim != 3:
        raise ValueError(f"MAT field 'data' must be 3D (epochs, channels, times), got shape {data.shape}")

    sfreq = float(_get_required_field(mat, sfreq_key, path))
    ch_names = [str(x) for x in list(_get_required_field(mat, ch_names_key, path))]
    if len(ch_names) != data.shape[1]:
        raise ValueError(
            f"ch_names length ({len(ch_names)}) does not match n_channels ({data.shape[1]})"
        )

    tmin = float(mat.get(tmin_key, schema.get("tmin_default", -0.2)))
    ch_types = _coerce_channel_types(mat.get(ch_types_key, schema.get("ch_types_default", "mag")), data.shape[1])

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    if events_key in mat:
        events = np.asarray(mat[events_key], dtype=int)
        if events.ndim != 2 or events.shape[1] != 3 or events.shape[0] != data.shape[0]:
            raise ValueError(
                f"MAT field '{events_key}' must have shape (n_epochs, 3) and align with data epochs"
            )
    else:
        events = np.column_stack(
            [np.arange(data.shape[0]), np.zeros(data.shape[0], dtype=int), np.ones(data.shape[0], dtype=int)]
        )

    event_id_in = mat.get(event_id_key, {"default": 1})
    if isinstance(event_id_in, dict):
        event_id = {str(k): int(v) for k, v in event_id_in.items()}
    else:
        event_id = {"default": int(np.max(events[:, 2])) if events.size else 1}

    return mne.EpochsArray(data, info, events=events, tmin=tmin, event_id=event_id, verbose=False)


def load_subject_anatomy(subject_id: str, cfg: PipelineConfig) -> dict[str, Any]:
    input_cfg = cfg.raw["input"]
    mri_cfg = input_cfg["anatomy_files"]
    mri_base = cfg.data_root / input_cfg["mri_dir"] / subject_id

    out: dict[str, Any] = {}
    trans_path = mri_base / mri_cfg["trans"]
    bem_path = mri_base / mri_cfg["bem"]
    src_path = mri_base / mri_cfg["src"]

    if trans_path.exists():
        out["trans"] = mne.read_trans(trans_path)
    if bem_path.exists():
        out["bem"] = mne.read_bem_solution(bem_path)
    if src_path.exists():
        out["src"] = mne.read_source_spaces(src_path)

    return out


def save_derivative(obj: Any, subject_id: str, stage: str, cfg: PipelineConfig) -> Path:
    out_dir = cfg.output_root / subject_id
    out_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(obj, "save"):
        out_path = out_dir / f"{stage}-epo.fif"
        obj.save(out_path, overwrite=True)
        return out_path

    out_path = out_dir / f"{stage}.npy"
    np.save(out_path, obj, allow_pickle=True)
    return out_path
