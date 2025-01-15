from typing import Any, Final, TypeAlias, overload

import numpy as np
import numpy.typing as npt

MIN_FLOAT_TOLERANCE: Final[float] = 1e-6

FloatArray: TypeAlias = npt.NDArray[np.floating[Any]]
IntArray: TypeAlias = npt.NDArray[np.integer[Any]]


@overload
def uniform_to_category(u_sample: float, n_categories: int) -> np.int64: ...


@overload
def uniform_to_category(u_sample: FloatArray, n_categories: int) -> IntArray: ...


def uniform_to_category(u_sample: float | FloatArray, n_categories: int) -> np.int64 | IntArray:
    if n_categories <= 0:
        raise ValueError("n_categories must be positive")

    if isinstance(u_sample, (np.ndarray, np.floating)):
        u = np.asarray(u_sample)
        if np.any(u < 0.0) or np.any(u > 1.0):
            raise ValueError(f"u_sample must be in the range [0, 1], got {u_sample}")
    else:
        if u_sample < 0.0 or u_sample > 1.0:
            raise ValueError(f"u_sample must be in the range [0, 1], got {u_sample}")
        u = np.asarray(u_sample)

    indices = np.floor(u * n_categories).astype(np.int64)
    return np.minimum(indices, n_categories - 1)
