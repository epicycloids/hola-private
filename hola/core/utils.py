"""
Core utilities for HOLA (Hyperparameter Optimization, Lightweight Asynchronous).

This module provides fundamental utility functions and type definitions used
throughout the HOLA library for hyperparameter optimization tasks. It includes
tools for handling categorical parameters and defining common type aliases.
"""

from typing import Any, Final, TypeAlias, overload

import numpy as np
import numpy.typing as npt

# Constants
MIN_FLOAT_TOLERANCE: Final[float] = 1e-6
"""Minimum floating point tolerance used for numerical comparisons."""

# Type Aliases
FloatArray: TypeAlias = npt.NDArray[np.floating[Any]]
"""Type alias for NumPy arrays containing floating point values."""

IntArray: TypeAlias = npt.NDArray[np.integer[Any]]
"""Type alias for NumPy arrays containing integer values."""


@overload
def uniform_to_category(u_sample: float, n_categories: int) -> np.int64: ...


@overload
def uniform_to_category(u_sample: FloatArray, n_categories: int) -> IntArray: ...


def uniform_to_category(u_sample: float | FloatArray, n_categories: int) -> np.int64 | IntArray:
    """
    Convert uniform random samples to categorical indices.

    This function maps values from the uniform distribution [0, 1] to integer
    indices representing categories. It's particularly useful for sampling
    categorical parameters in hyperparameter optimization.

    :param u_sample: Uniform random sample(s) in the range [0, 1]
    :type u_sample: float | FloatArray
    :param n_categories: Number of categories to map to (must be positive)
    :type n_categories: int
    :return: Integer indices representing categorical choices
    :rtype: np.int64 | IntArray
    :raises ValueError: If n_categories <= 0 or if u_sample contains values outside [0, 1]

    Example:
        >>> import numpy as np
        >>> uniform_to_category(0.7, 3)  # Single sample
        2
        >>> uniform_to_category(np.array([0.1, 0.5, 0.9]), 3)  # Multiple samples
        array([0, 1, 2])
    """
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
