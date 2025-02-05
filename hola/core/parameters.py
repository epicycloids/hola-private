"""
Parameter configuration and transformation utilities for HOLA.

This module defines the parameter configuration system used to specify the search
space for hyperparameter optimization. It provides classes for different types of
parameters (continuous, categorical, integer, and lattice) and utilities for
transforming between normalized [0,1] space and parameter-native spaces.

Key components:
- Parameter configs that specify the domain and behavior of each parameter type
- A transformer that handles conversion between normalized and native parameter spaces
- Support for handling both scalar and batch parameter transformations

The normalized space is designed such that:
1. `unnormalize` is a left inverse of `normalize`
2. All sampling occurs in the unit hypercube [0,1]^d
3. Discrete parameters map uniformly to their possible values
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, Protocol, TypeAlias, TypeVar, Union, overload, runtime_checkable

import msgspec
import numpy as np
import numpy.typing as npt
from msgspec import Struct

from hola.core.utils import MIN_FLOAT_TOLERANCE, FloatArray, IntArray, uniform_to_category

ParameterName: TypeAlias = str
"""Type alias for parameter names in configuration dictionaries."""

Category = TypeVar("Category")
"""Generic type variable for categorical parameter values."""


class Scale(str, Enum):
    """
    Scale types for continuous parameters.

    :cvar LINEAR: Linear scale between min and max values
    :cvar LOG: Logarithmic scale between min and max values
    """

    LINEAR = "linear"
    LOG = "log"


class ParamInfeasibleError(ValueError):
    """Raised when parameter values fall outside their feasible domains."""

    pass


@runtime_checkable
class ParameterConfig(Protocol):
    """
    Protocol defining the interface for parameter configurations.

    All parameter types must implement methods for:
    - Converting between normalized [0,1] space and parameter-native space
    - Checking value feasibility
    - Comparing parameter domains for expansion
    """

    @overload
    def unnormalize(self, u: float) -> Any: ...

    @overload
    def unnormalize(self, u: FloatArray) -> npt.NDArray[Any]: ...

    def unnormalize(self, u: float | FloatArray) -> Any | npt.NDArray[Any]:
        """
        Convert normalized values to parameter-native space.

        :param u: Values in [0,1] to convert
        :type u: float | FloatArray
        :return: Values in parameter-native space
        :rtype: Any | npt.NDArray[Any]
        :raises ValueError: If u contains values outside [0,1]
        """
        ...

    def normalize(self, x: Any) -> float | FloatArray:
        """
        Convert parameter-native values to normalized [0,1] space.

        :param x: Values in parameter-native space
        :type x: Any
        :return: Values in [0,1]
        :rtype: float | FloatArray
        :raises ParamInfeasibleError: If x contains infeasible values
        """
        ...

    def is_feasible(self, value: Any) -> bool:
        """
        Check if value is feasible for this parameter.

        :param value: Value to check
        :type value: Any
        :return: True if value is feasible, False otherwise
        :rtype: bool
        """
        ...

    def has_expanded_domain(self, old_config: "ParameterConfig") -> bool:
        """
        Check if this config's domain is an expansion of old_config's domain.

        :param old_config: Previous parameter configuration to compare against
        :type old_config: ParameterConfig
        :return: True if this domain is an expansion of the old domain
        :rtype: bool
        :raises TypeError: If configs are of incompatible types
        """
        ...


class BaseParameterConfig(Struct, frozen=True):
    """Base class for parameter configurations with common validation logic."""

    @staticmethod
    def validate_unit_float(u: float | FloatArray) -> None:
        """
        Validate that value(s) lie in the unit interval [0,1].

        :param u: Value(s) to validate
        :type u: float | FloatArray
        :raises ValueError: If any values lie outside [0,1]
        :raises TypeError: If u is neither float nor ndarray
        """
        if isinstance(u, (int, float)):
            if u < 0.0 or u > 1.0:
                raise ValueError(f"Value must be in [0,1], got {u}")
        elif isinstance(u, np.ndarray):
            if np.any(u < 0.0) or np.any(u > 1.0):
                raise ValueError("All array values must be in [0,1].")
        else:
            raise TypeError(f"Value must be a float or ndarray, got {type(u)}")


class ContinuousParameterConfig(BaseParameterConfig, tag="continuous", frozen=True):
    """
    Configuration for continuous-valued parameters.

    Supports both linear and logarithmic scales between minimum and maximum values.
    """

    min: float
    """Minimum value of the parameter."""

    max: float
    """Maximum value of the parameter."""

    scale: Scale = Scale.LINEAR
    """Scale type for the parameter (linear or logarithmic)."""

    def __post_init__(self):
        """
        Validate parameter configuration.

        :raises ValueError: If max <= min or if log scale with non-positive bounds
        """
        if self.max < self.min + MIN_FLOAT_TOLERANCE:
            raise ValueError("max must be strictly greater than min")
        if self.scale == Scale.LOG and (self.min <= 0 or self.max <= 0):
            raise ValueError("log scale requires strictly positive bounds")

    @overload
    def unnormalize(self, u: float) -> float: ...

    @overload
    def unnormalize(self, u: FloatArray) -> FloatArray: ...

    def unnormalize(self, u: float | FloatArray) -> float | FloatArray:
        """
        Convert normalized values to continuous parameter values.

        For linear scale: x = min + u * (max - min)
        For log scale: x = min * exp(u * log(max/min))

        :param u: Value(s) in [0,1] to convert
        :type u: float | FloatArray
        :return: Parameter value(s) in [min, max]
        :rtype: float | FloatArray
        :raises ValueError: If u contains values outside [0,1]
        """
        self.validate_unit_float(u)
        if self.scale == Scale.LINEAR:
            result = self.min + u * (self.max - self.min)
        else:
            ratio = np.log(self.max / self.min)
            result = self.min * np.exp(u * ratio)

        if isinstance(u, np.ndarray):
            return result
        return float(result)

    @overload
    def normalize(self, x: float) -> float: ...

    @overload
    def normalize(self, x: FloatArray) -> FloatArray: ...

    def normalize(self, x: float | FloatArray) -> float | FloatArray:
        """
        Convert parameter values to normalized space.

        For linear scale: u = (x - min) / (max - min)
        For log scale: u = log(x/min) / log(max/min)

        :param x: Parameter value(s) to normalize
        :type x: float | FloatArray
        :return: Normalized value(s) in [0,1]
        :rtype: float | FloatArray
        :raises ParamInfeasibleError: If x contains values outside [min, max]
        """
        if not self.is_feasible(x):
            raise ParamInfeasibleError(f"Values outside bounds [{self.min}, {self.max}]")

        arr = np.asarray(x)
        if self.scale == Scale.LINEAR:
            out = (arr - self.min) / (self.max - self.min)
        else:
            out = np.log(arr / self.min) / np.log(self.max / self.min)

        if isinstance(x, np.ndarray):
            return out
        return float(out)

    def is_feasible(self, x: float | FloatArray) -> bool:
        """
        Check if value(s) lie within parameter bounds.

        :param x: Value(s) to check
        :type x: float | FloatArray
        :return: True if all values are in [min, max]
        :rtype: bool
        """
        arr = np.asarray(x)
        return bool(np.all(arr >= self.min) and np.all(arr <= self.max))

    def has_expanded_domain(self, old_config: "ContinuousParameterConfig") -> bool:
        """
        Check if this config's domain is an expansion of old_config's domain.

        A domain is considered expanded if:
        - The scale type has changed (requiring resampling)
        - The minimum value has decreased
        - The maximum value has increased

        :param old_config: Previous configuration to compare against
        :type old_config: ContinuousParameterConfig
        :return: True if domain has expanded
        :rtype: bool
        :raises TypeError: If old_config is not a ContinuousParameterConfig
        """
        if not isinstance(old_config, ContinuousParameterConfig):
            raise TypeError(f"Cannot compare with config of type {type(old_config)}")

        # Changing scale is considered an expansion (or at least requires a reset).
        if self.scale != old_config.scale:
            return True

        return (self.min < old_config.min) or (self.max > old_config.max)


class CategoricalParameterConfig(
    BaseParameterConfig, Generic[Category], tag="categorical", frozen=True
):
    """
    Configuration for categorical parameters.

    Maps normalized values uniformly to a finite set of categories. Normalized values
    are mapped to indices using a consistent binning strategy to ensure uniform
    sampling across categories.
    """

    categories: tuple[Category]
    """Tuple of valid categories for this parameter."""

    def __post_init__(self):
        """
        Validate parameter configuration.

        :raises ValueError: If categories is empty
        """
        if not self.categories:
            raise ValueError("`categories` cannot be empty")

    @property
    def n_categories(self) -> int:
        """
        :return: Number of possible categories
        :rtype: int
        """
        return len(self.categories)

    @overload
    def unnormalize(self, u: float) -> Category: ...

    @overload
    def unnormalize(self, u: FloatArray) -> npt.NDArray: ...

    def unnormalize(self, u: float | FloatArray) -> Category | npt.NDArray:
        """
        Convert normalized values to categories.

        The unit interval is divided into n_categories equal-sized bins,
        and each normalized value is mapped to the category corresponding
        to its bin.

        :param u: Value(s) in [0,1] to convert
        :type u: float | FloatArray
        :return: Category or array of categories
        :rtype: Category | npt.NDArray
        :raises ValueError: If u contains values outside [0,1]
        """
        self.validate_unit_float(u)
        idx = uniform_to_category(u, self.n_categories)
        if isinstance(u, np.ndarray) and u.ndim > 0:
            return np.array([self.categories[int(i)] for i in idx.flatten()]).reshape(u.shape)
        else:
            return self.categories[int(idx)]

    @overload
    def normalize(self, x: Category) -> float: ...

    @overload
    def normalize(self, x: npt.NDArray) -> FloatArray: ...

    def normalize(self, x: Category | npt.NDArray) -> float | FloatArray:
        """
        Convert categories to normalized space.

        Each category is mapped to the center of its corresponding bin
        in the unit interval: (index + 0.5) / n_categories

        :param x: Category or array of categories to normalize
        :type x: Category | npt.NDArray
        :return: Normalized value(s) in [0,1]
        :rtype: float | FloatArray
        :raises ParamInfeasibleError: If x contains invalid categories
        """
        if not self.is_feasible(x):
            valid_cats = ", ".join(repr(c) for c in self.categories)
            if isinstance(x, np.ndarray):
                raise ParamInfeasibleError(f"Values not in categories: [{valid_cats}]")
            else:
                raise ParamInfeasibleError(f"Value {repr(x)} not in categories: [{valid_cats}]")

        if isinstance(x, np.ndarray) and x.ndim > 0:
            indices = np.array([self.categories.index(xi) for xi in x.flat])
            return ((indices + 0.5) / self.n_categories).reshape(x.shape)
        else:
            idx = self.categories.index(x)
            return (idx + 0.5) / self.n_categories

    def is_feasible(self, x: Category | npt.NDArray) -> bool:
        """
        Check if value(s) are valid categories.

        :param x: Value(s) to check
        :type x: Category | npt.NDArray
        :return: True if all values are valid categories
        :rtype: bool
        """
        if isinstance(x, np.ndarray):
            return all(xi in self.categories for xi in x.flat)
        else:
            return x in self.categories

    def has_expanded_domain(self, old_config: "CategoricalParameterConfig") -> bool:
        """
        Check if this config's domain is an expansion of old_config's domain.

        A categorical domain is expanded if it contains any categories
        not present in the old config.

        :param old_config: Previous configuration to compare against
        :type old_config: CategoricalParameterConfig
        :return: True if domain has expanded
        :rtype: bool
        :raises TypeError: If old_config is not a CategoricalParameterConfig
        """
        if not isinstance(old_config, CategoricalParameterConfig):
            raise TypeError(f"Cannot compare with config of type {type(old_config)}")
        return not set(old_config.categories).issuperset(self.categories)


class IntegerParameterConfig(BaseParameterConfig, tag="integer", frozen=True):
    """
    Configuration for integer-valued parameters.

    Maps normalized values uniformly to integers in [min, max]. Uses the same
    binning strategy as categorical parameters to ensure uniform sampling.
    """

    min: int
    """Minimum value of the parameter (inclusive)."""

    max: int
    """Maximum value of the parameter (inclusive)."""

    def __post_init__(self):
        """
        Validate parameter configuration.

        :raises ValueError: If min > max
        """
        if self.min > self.max:
            raise ValueError(f"min ({self.min}) must be <= max ({self.max})")

    @property
    def num_values(self) -> int:
        """
        :return: Number of possible integer values
        :rtype: int
        """
        return self.max - self.min + 1

    @overload
    def unnormalize(self, u: float) -> int: ...

    @overload
    def unnormalize(self, u: FloatArray) -> IntArray: ...

    def unnormalize(self, u: float | FloatArray) -> int | IntArray:
        """
        Convert normalized values to integers.

        The unit interval is divided into num_values equal-sized bins,
        and each normalized value is mapped to min + bin_index.

        :param u: Value(s) in [0,1] to convert
        :type u: float | FloatArray
        :return: Integer value(s) in [min, max]
        :rtype: int | IntArray
        :raises ValueError: If u contains values outside [0,1]
        """
        self.validate_unit_float(u)
        idx = uniform_to_category(u, self.num_values)
        if isinstance(u, np.ndarray) and u.ndim > 0:
            return (self.min + idx).astype(int).reshape(u.shape)
        return int(self.min + idx)

    @overload
    def normalize(self, x: int) -> float: ...

    @overload
    def normalize(self, x: IntArray) -> FloatArray: ...

    def normalize(self, x: int | IntArray) -> float | FloatArray:
        """
        Convert integers to normalized space.

        Each integer is mapped to the center of its corresponding bin
        in the unit interval: (value - min + 0.5) / num_values

        :param x: Integer value(s) to normalize
        :type x: int | IntArray
        :return: Normalized value(s) in [0,1]
        :rtype: float | FloatArray
        :raises ParamInfeasibleError: If x contains values outside [min, max]
        """
        if not self.is_feasible(x):
            raise ParamInfeasibleError(f"Values outside bounds [{self.min}, {self.max}]")

        arr = np.asarray(x)
        arr_int = np.round(arr).astype(int)
        indices = arr_int - self.min
        out = (indices + 0.5) / self.num_values

        if isinstance(x, np.ndarray):
            return out
        return float(out)

    def is_feasible(self, x: int | IntArray) -> bool:
        """
        Check if value(s) are valid integers within bounds.

        Values are rounded to nearest integer before checking bounds.

        :param x: Value(s) to check
        :type x: int | IntArray
        :return: True if all rounded values are in [min, max]
        :rtype: bool
        """
        arr = np.asarray(x)
        arr_int = np.round(arr).astype(int)
        return bool(np.all(arr_int >= self.min) and np.all(arr_int <= self.max))

    def has_expanded_domain(self, old_config: "IntegerParameterConfig") -> bool:
        """
        Check if this config's domain is an expansion of old_config's domain.

        The domain is expanded if either bound has moved outward.

        :param old_config: Previous configuration to compare against
        :type old_config: IntegerParameterConfig
        :return: True if domain has expanded
        :rtype: bool
        :raises TypeError: If old_config is not an IntegerParameterConfig
        """
        if not isinstance(old_config, IntegerParameterConfig):
            raise TypeError(f"Cannot compare with config of type {type(old_config)}")
        return (self.min < old_config.min) or (self.max > old_config.max)


class LatticeParameterConfig(BaseParameterConfig, tag="lattice", frozen=True):
    """
    Configuration for lattice-valued parameters.

    Maps normalized values to a uniform grid of floating-point values. Useful for
    parameters that should take on a fixed set of evenly-spaced values.
    """

    min: float
    """Minimum value of the parameter."""

    max: float
    """Maximum value of the parameter."""

    num_values: int
    """Number of points in the lattice."""

    def __post_init__(self):
        """
        Validate parameter configuration.

        :raises ValueError: If max <= min or num_values <= 1
        """
        if self.max < self.min + MIN_FLOAT_TOLERANCE:
            raise ValueError("max must be strictly greater than min")
        if self.num_values <= 1:
            raise ValueError("num_values must be greater than 1")

    @property
    def step_size(self) -> float:
        """
        :return: Distance between consecutive lattice points
        :rtype: float
        """
        return (self.max - self.min) / (self.num_values - 1)

    @overload
    def unnormalize(self, u: float) -> float: ...

    @overload
    def unnormalize(self, u: FloatArray) -> FloatArray: ...

    def unnormalize(self, u: float | FloatArray) -> float | FloatArray:
        """
        Convert normalized values to lattice points.

        The unit interval is divided into num_values equal-sized bins,
        and each normalized value is mapped to min + bin_index * step_size.

        :param u: Value(s) in [0,1] to convert
        :type u: float | FloatArray
        :return: Lattice point value(s)
        :rtype: float | FloatArray
        :raises ValueError: If u contains values outside [0,1]
        """
        self.validate_unit_float(u)
        idx = uniform_to_category(u, self.num_values)
        if isinstance(u, np.ndarray) and u.ndim > 0:
            return (self.min + idx * self.step_size).astype(float).reshape(u.shape)
        return float(self.min + idx * self.step_size)

    @overload
    def normalize(self, x: float) -> float: ...

    @overload
    def normalize(self, x: FloatArray) -> FloatArray: ...

    def normalize(self, x: float | FloatArray) -> float | FloatArray:
        """
        Convert lattice points to normalized space.

        Values are first rounded to the nearest lattice point, then mapped
        to the center of their corresponding bin: (index + 0.5) / num_values

        :param x: Value(s) to normalize
        :type x: float | FloatArray
        :return: Normalized value(s) in [0,1]
        :rtype: float | FloatArray
        :raises ParamInfeasibleError: If x is not close to a lattice point
                                    or outside [min, max]
        """
        if not self.is_feasible(x):
            raise ParamInfeasibleError(
                f"Values not on lattice or outside [{self.min}, {self.max}]."
            )

        arr = np.asarray(x)
        indices = np.round((arr - self.min) / self.step_size)
        out = (indices + 0.5) / self.num_values

        if isinstance(x, np.ndarray):
            return out
        return float(out)

    def is_feasible(self, x: float | FloatArray) -> bool:
        """
        Check if value(s) are valid lattice points.

        A value is valid if it is sufficiently close to a lattice point
        and within the parameter bounds.

        :param x: Value(s) to check
        :type x: float | FloatArray
        :return: True if all values are valid lattice points
        :rtype: bool
        """
        arr = np.asarray(x)
        if np.any(arr < self.min) or np.any(arr > self.max):
            return False
        indices = np.round((arr - self.min) / self.step_size)
        lattice_vals = self.min + indices * self.step_size
        return np.allclose(arr, lattice_vals)

    def has_expanded_domain(self, old_config: "LatticeParameterConfig") -> bool:
        """
        Check if this config's domain is an expansion of old_config's domain.

        The domain is expanded if:
        - The bounds have moved outward
        - The number of lattice points has increased

        :param old_config: Previous configuration to compare against
        :type old_config: LatticeParameterConfig
        :return: True if domain has expanded
        :rtype: bool
        :raises TypeError: If old_config is not a LatticeParameterConfig
        """
        if not isinstance(old_config, LatticeParameterConfig):
            raise TypeError(f"Cannot compare with config of type {type(old_config)}")
        return (
            self.min < old_config.min
            or self.max > old_config.max
            or self.num_values > old_config.num_values
        )


PredefinedParameterConfig: TypeAlias = Union[
    ContinuousParameterConfig,
    CategoricalParameterConfig,
    IntegerParameterConfig,
    LatticeParameterConfig,
]
"""Type alias for all concrete parameter configuration types."""


@dataclass
class ParameterTransformer:
    """
    Handles conversion between normalized and parameter-native spaces.

    This class manages a collection of parameter configurations and provides methods
    for batch conversion of parameter values between the normalized unit hypercube
    used for sampling and the native parameter spaces.

    The transformer ensures consistent handling of parameters across the optimization
    process and provides validation of parameter values and configurations.
    """

    parameters: dict[ParameterName, ParameterConfig]
    """Dictionary mapping parameter names to their configurations."""

    @classmethod
    def from_dict(cls, params_dict: dict[ParameterName, dict[str, Any]]) -> "ParameterTransformer":
        """
        Create transformer from a dictionary of parameter specifications.

        :param params_dict: Dictionary mapping names to parameter configurations
        :type params_dict: dict[ParameterName, dict[str, Any]]
        :return: Configured parameter transformer
        :rtype: ParameterTransformer
        """
        configs = msgspec.convert(params_dict, dict[ParameterName, PredefinedParameterConfig])
        return cls(configs)

    @classmethod
    def from_json(cls, json_str: str) -> "ParameterTransformer":
        """
        Create transformer from a JSON string of parameter specifications.

        :param json_str: JSON string containing parameter configurations
        :type json_str: str
        :return: Configured parameter transformer
        :rtype: ParameterTransformer
        """
        configs = msgspec.json.decode(json_str, type=dict[ParameterName, PredefinedParameterConfig])
        return cls(configs)

    @property
    def param_names(self) -> list[ParameterName]:
        """
        :return: List of parameter names in consistent order
        :rtype: list[ParameterName]
        """
        return list(self.parameters.keys())

    @property
    def num_params(self) -> int:
        """
        :return: Number of parameters in the transformer
        :rtype: int
        """
        return len(self.parameters)

    def unnormalize(self, normalized_params: npt.NDArray) -> list[dict[ParameterName, Any]]:
        """
        Convert batch of normalized parameters to native parameter dictionaries.

        :param normalized_params: Array of shape (n_samples, num_params) in [0,1]
        :type normalized_params: npt.NDArray
        :return: List of parameter dictionaries in native space
        :rtype: list[dict[ParameterName, Any]]
        :raises ValueError: If normalized_params has incorrect shape
        """
        if normalized_params.ndim != 2 or normalized_params.shape[1] != self.num_params:
            raise ValueError(
                f"Expected (n_samples, {self.num_params}) array, got {normalized_params.shape}"
            )

        n_samples = normalized_params.shape[0]
        results = []
        for i in range(n_samples):
            row_dict = {
                name: self.parameters[name].unnormalize(normalized_params[i, j])
                for j, name in enumerate(self.param_names)
            }
            results.append(row_dict)
        return results

    def normalize(self, param_dicts: list[dict[ParameterName, Any]]) -> npt.NDArray:
        """
        Convert batch of parameter dictionaries to normalized space.

        :param param_dicts: List of parameter dictionaries in native space
        :type param_dicts: list[dict[ParameterName, Any]]
        :return: Array of shape (n_samples, num_params) in [0,1]
        :rtype: npt.NDArray
        :raises ValueError: If param_dicts is empty or has incorrect parameters
        """
        if not param_dicts:
            raise ValueError("Empty parameter list.")

        n_samples = len(param_dicts)
        arr = np.zeros((n_samples, self.num_params), dtype=float)

        # Check first sample's keys
        if set(param_dicts[0].keys()) != set(self.param_names):
            raise ValueError("Parameter names do not match configuration.")

        for i, pdict in enumerate(param_dicts):
            if set(pdict.keys()) != set(self.param_names):
                raise ValueError(f"Param names in sample {i} do not match transformer config.")
            for j, name in enumerate(self.param_names):
                arr[i, j] = self.parameters[name].normalize(pdict[name])

        return arr

    def is_feasible(self, param_dict: dict[ParameterName, Any]) -> bool:
        """
        Check if parameter dictionary contains feasible values.

        :param param_dict: Parameter dictionary to check
        :type param_dict: dict[ParameterName, Any]
        :return: True if all parameters are feasible, False otherwise
        :rtype: bool
        """
        if set(param_dict.keys()) != set(self.param_names):
            return False
        return all(self.parameters[name].is_feasible(val) for name, val in param_dict.items())

    def has_expanded_domain(self, old_transformer: "ParameterTransformer") -> bool:
        """
        Check if this transformer's domain is an expansion of old_transformer's domain.

        :param old_transformer: Previous transformer to compare against
        :type old_transformer: ParameterTransformer
        :return: True if this domain is an expansion of the old domain
        :rtype: bool
        :raises ValueError: If parameter sets don't match
        :raises TypeError: If parameter types don't match
        """
        if set(self.parameters) != set(old_transformer.parameters):
            raise ValueError("Parameter sets do not match between transformers.")

        for name, new_cfg in self.parameters.items():
            old_cfg = old_transformer.parameters[name]
            if not isinstance(old_cfg, type(new_cfg)):
                raise TypeError(f"Param {name} type mismatch: {type(new_cfg)} vs {type(old_cfg)}")
            if new_cfg.has_expanded_domain(old_cfg):  # type: ignore
                return True

        return False
