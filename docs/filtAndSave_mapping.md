# filtAndSave behavior mapping

This document maps legacy MATLAB stage outputs to new pipeline stages.

## Legacy outputs

- Banded/filt_info_<participant>_SFNI.mat
- Banded/filt_<low>_<high>_<participant>_SFNI.mat
- Banded/filt_<low>_<high>_<participant>_HB_SFNI.mat

## New pipeline stages

1. Load preprocessed participant data
   - pipeline.io.load_preprocessed_subject
   - Supports FIF directly and a simple MAT schema adapter.

2. Apply configurable band filtering
   - pipeline.filtering.apply_band_filter
   - Driven by config filtering.bands list.

3. Compute complex Hilbert transforms
   - pipeline.filtering.compute_hilbert_complex
   - Stored separately from bandpassed real-valued data.

4. Persist derivatives by stage
   - pipeline.io.save_derivative
   - Stages: filt_info, filt_<band>, filt_<band>_HB

## Key improvements over script style

- No hardcoded subject loops
- No hardcoded frequency constants
- No hardcoded save paths
- Contrast and test lists are config-driven
- Statistical tests run through a plugin registry
