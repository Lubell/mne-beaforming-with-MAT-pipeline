# MNE Beamforming Pipeline (Scaffold)

This package is a modular, config-driven replacement for script-heavy MATLAB workflows.
It starts from preprocessed participant data and supports:

- Loading one preprocessed file per participant (MAT or FIF)
- Band filtering + Hilbert complex transform (filtAndSave-like stage)
- Condition selection and contrasts
- Event-code inspection before metadata/contrast setup
- Optional forward model + LCMV source projection
- Pluggable statistical tests

## Quick start

1. Create and activate a Python environment.
2. Install in editable mode:

   pip install -e .

3. Edit `configs/example_config.yaml`.
4. Run one subject:

   python scripts/run_subject.py --config configs/example_config.yaml --subject 0006_15Y

5. Optional: inspect event codes only (no filtering/stats):

   python scripts/run_subject.py --config configs/example_config.yaml --subject 0006_15Y --inspect-event-codes

## Notes

- This is intentionally a scaffold: stable public interfaces first, internals can evolve.
- Circular tests (`t2circ`, `watson_williams`) are placeholders to be completed during porting.
- Default derived metadata columns include `event_code`, `category`, and `filter` when event-code mapping is available.
