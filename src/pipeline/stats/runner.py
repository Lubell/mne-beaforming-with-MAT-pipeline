from __future__ import annotations

from typing import Any

import numpy as np

from .base import StatResult
from .tests import PermutationTest, RankSum, T2CircTest, TTestInd, WatsonWilliamsTest


def _fdr_bh(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    p = np.asarray(pvals).reshape(-1)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    thresh = alpha * (np.arange(1, n + 1) / n)
    passed = ranked <= thresh

    out = np.zeros(n, dtype=bool)
    if np.any(passed):
        k = np.max(np.where(passed)[0])
        cutoff = ranked[k]
        out = p <= cutoff
    return out.reshape(np.asarray(pvals).shape)


REGISTRY = {
    "ttest_ind": TTestInd,
    "rank_sum": RankSum,
    "permutation": PermutationTest,
    "t2circ": T2CircTest,
    "watson_williams": WatsonWilliamsTest,
}


def run_stats(test_names: list[str], data_a: np.ndarray, data_b: np.ndarray, cfg: dict[str, Any]) -> dict[str, StatResult]:
    results: dict[str, StatResult] = {}
    for test_name in test_names:
        if test_name not in REGISTRY:
            raise ValueError(f"Unknown stat test: {test_name}")
        test = REGISTRY[test_name]()
        results[test_name] = test.fit(data_a, data_b, ctx=cfg)
    return results


def apply_multiple_comparison(result: StatResult, cfg: dict[str, Any]) -> StatResult:
    alpha = float(cfg.get("fdr_alpha", 0.05))
    mask = _fdr_bh(result.pvalue, alpha=alpha)
    meta = dict(result.meta)
    meta["fdr_alpha"] = alpha
    meta["fdr_mask"] = mask
    return StatResult(name=result.name, statistic=result.statistic, pvalue=result.pvalue, meta=meta)
