import unittest

import numpy as np

from pipeline.config import validate_config
from pipeline.stats.runner import apply_multiple_comparison, run_stats


class ConfigAndStatsTests(unittest.TestCase):
    def test_experimental_stats_blocked_by_default(self):
        cfg = {
            "data_root": "/tmp/data",
            "output_root": "/tmp/out",
            "input": {
                "preprocessed_dir": "pre",
                "mri_dir": "mri",
                "preprocessed_extensions": [".fif", ".mat"],
            },
            "filtering": {"bands": [{"name": "b1", "fmin": 1.0, "fmax": 4.0}]},
            "stats": {"tests": ["t2circ"]},
        }
        with self.assertRaises(ValueError):
            validate_config(cfg)

    def test_stats_runner_and_fdr(self):
        rng = np.random.default_rng(seed=0)
        a = rng.normal(size=(20, 30))
        b = rng.normal(loc=0.2, size=(18, 30))

        results = run_stats(["ttest_ind", "rank_sum", "permutation"], a, b, {"n_perm": 100})
        self.assertEqual(set(results.keys()), {"ttest_ind", "rank_sum", "permutation"})
        for result in results.values():
            self.assertEqual(result.pvalue.shape, (30,))
            adjusted = apply_multiple_comparison(result, {"fdr_alpha": 0.05})
            self.assertEqual(adjusted.meta["fdr_mask"].shape, (30,))


if __name__ == "__main__":
    unittest.main()
