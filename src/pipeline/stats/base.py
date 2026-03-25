from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class StatResult:
    name: str
    statistic: np.ndarray
    pvalue: np.ndarray
    meta: dict[str, Any]


class BaseStatTest(ABC):
    name: str

    @abstractmethod
    def fit(self, data_a: np.ndarray, data_b: np.ndarray, ctx: dict[str, Any] | None = None) -> StatResult:
        pass
