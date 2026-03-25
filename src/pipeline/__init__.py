"""Config-driven MNE beamforming pipeline."""

from .config import PipelineConfig, load_config
from .orchestrator import run_subject

__all__ = ["PipelineConfig", "load_config", "run_subject"]
