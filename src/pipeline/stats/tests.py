from __future__ import annotations

import numpy as np
from scipy.stats import ranksums, ttest_ind

from .base import BaseStatTest, StatResult


class TTestInd(BaseStatTest):
    name = "ttest_ind"

    def fit(self, data_a: np.ndarray, data_b: np.ndarray, ctx=None) -> StatResult:
        stat, p = ttest_ind(data_a, data_b, axis=0, equal_var=False, nan_policy="omit")
        return StatResult(name=self.name, statistic=np.asarray(stat), pvalue=np.asarray(p), meta={})


class RankSum(BaseStatTest):
    name = "rank_sum"

    def fit(self, data_a: np.ndarray, data_b: np.ndarray, ctx=None) -> StatResult:
        # ranksums is scalar per feature, so iterate over columns
        stats = np.zeros(data_a.shape[1], dtype=float)
        pvals = np.ones(data_a.shape[1], dtype=float)
        for i in range(data_a.shape[1]):
            s, p = ranksums(data_a[:, i], data_b[:, i])
            stats[i] = s
            pvals[i] = p
        return StatResult(name=self.name, statistic=stats, pvalue=pvals, meta={})


class PermutationTest(BaseStatTest):
    name = "permutation"

    def fit(self, data_a: np.ndarray, data_b: np.ndarray, ctx=None) -> StatResult:
        rng = np.random.default_rng(seed=42)
        n_perm = int((ctx or {}).get("n_perm", 1000))

        obs = np.mean(data_a, axis=0) - np.mean(data_b, axis=0)
        pooled = np.vstack([data_a, data_b])
        n_a = data_a.shape[0]
        null = np.zeros((n_perm, data_a.shape[1]))

        for k in range(n_perm):
            idx = rng.permutation(pooled.shape[0])
            a_k = pooled[idx[:n_a], :]
            b_k = pooled[idx[n_a:], :]
            null[k] = np.mean(a_k, axis=0) - np.mean(b_k, axis=0)

        pvals = (np.sum(np.abs(null) >= np.abs(obs), axis=0) + 1) / (n_perm + 1)
        return StatResult(name=self.name, statistic=obs, pvalue=pvals, meta={"n_perm": n_perm})


class T2CircTest(BaseStatTest):
    name = "t2circ"

    def fit(self, data_a: np.ndarray, data_b: np.ndarray, ctx=None) -> StatResult:
        raise NotImplementedError(
            "T2CircTest is a placeholder. Port MATLAB T2circ2 behavior before enabling this test."
        )


class WatsonWilliamsTest(BaseStatTest):
    name = "watson_williams"

    def fit(self, data_a: np.ndarray, data_b: np.ndarray, ctx=None) -> StatResult:
        raise NotImplementedError(
            "WatsonWilliamsTest is a placeholder. Add a circular stats backend before enabling this test."
        )
