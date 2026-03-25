from setuptools import find_packages, setup

setup(
	name="mne-beam-pipeline",
	version="0.1.0",
	description="Config-driven MNE beamforming pipeline scaffold",
	python_requires=">=3.10",
	package_dir={"": "src"},
	packages=find_packages(where="src"),
	install_requires=[
		"mne>=1.7",
		"numpy>=1.24",
		"scipy>=1.10",
		"pandas>=2.0",
		"PyYAML>=6.0",
		"h5py>=3.8",
	],
)
