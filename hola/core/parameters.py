from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, Protocol, TypeAlias, TypeVar, Union, overload, runtime_checkable

import msgspec
import numpy as np
import numpy.typing as npt
from msgspec import Struct

from hola.core.utils import MIN_FLOAT_TOLERANCE, FloatArray, IntArray, uniform_to_category

# Type definitions
ParameterName = str
Category = TypeVar("Category")


class Scale(str, Enum):
    """Type of scaling to apply to continuous parameters."""

    LINEAR = "linear"
    LOG = "log"


class ParamInfeasibleError(ValueError):
    """Raised when a parameter value is infeasible under the current configuration."""

    pass


@runtime_checkable
class ParameterConfig(Protocol):
    """Base class for all parameter configurations."""

    @overload
    def unnormalize(self, u: float) -> Any: ...

    @overload
    def unnormalize(self, u: FloatArray) -> npt.NDArray[Any]: ...

    def unnormalize(self, u: float | FloatArray) -> Any | npt.NDArray[Any]:
        """Transform normalized parameter values to actual parameter space."""
        ...

    def normalize(self, x: Any) -> float | FloatArray:
        """Transform parameter values from actual space to normalized [0, 1] space."""
        ...

    def is_feasible(self, value: Any) -> bool: ...

    def has_expanded_domain(self, old_config: "ParameterConfig") -> bool: ...


class BaseParameterConfig(Struct, frozen=True):
    @staticmethod
    def validate_unit_float(u: float | FloatArray) -> None:
        """Validate that values are in [0, 1]."""
        if isinstance(u, (int, float)):
            if u < 0.0 or u > 1.0:
                raise ValueError(f"Value must be in [0, 1], got {u}")
        elif isinstance(u, np.ndarray):
            if np.any(u < 0.0) or np.any(u > 1.0):
                raise ValueError("All values must be in [0, 1]")
        else:
            raise TypeError(f"Value must be a number or numpy array, got {type(u)}")


class ContinuousParameterConfig(BaseParameterConfig, tag="continuous", frozen=True):
    """Configuration for continuous numerical parameters."""

    min: float
    max: float
    scale: Scale = Scale.LINEAR

    def __post_init__(self):
        """Validate configuration bounds and scaling compatibility."""
        if self.max < self.min + MIN_FLOAT_TOLERANCE:
            raise ValueError("max must be strictly greater than min")
        if self.scale == Scale.LOG and (self.min <= 0 or self.max <= 0):
            raise ValueError("log scale requires strictly positive bounds")

    @overload
    def unnormalize(self, u: float) -> float: ...

    @overload
    def unnormalize(self, u: FloatArray) -> FloatArray: ...

    def unnormalize(self, u: float | FloatArray) -> float | FloatArray:
        """Transform normalized values to continuous parameter values."""
        self.validate_unit_float(u)
        if self.scale == Scale.LINEAR:
            result = self.min + u * (self.max - self.min)
        else:
            result = self.min * np.exp(u * np.log(self.max / self.min))
        if isinstance(u, np.ndarray):
            return result
        return float(result)

    @overload
    def normalize(self, x: float) -> float: ...

    @overload
    def normalize(self, x: FloatArray) -> FloatArray: ...

    def normalize(self, x: float | FloatArray) -> float | FloatArray:
        if not self.is_feasible(x):
            raise ParamInfeasibleError(f"Values outside bounds [{self.min}, {self.max}]")

        x = np.asarray(x)
        if self.scale == Scale.LINEAR:
            return (x - self.min) / (self.max - self.min)
        else:
            return np.log(x / self.min) / np.log(self.max / self.min)

    def is_feasible(self, x: float | FloatArray) -> bool:
        x = np.asarray(x)
        return bool(np.all(x >= self.min) and np.all(x <= self.max))

    def has_expanded_domain(self, old_config: "ContinuousParameterConfig") -> bool:
        if not isinstance(old_config, ContinuousParameterConfig):
            raise TypeError(f"Cannot compare with config of type {type(old_config)}")

        if self.scale != old_config.scale:
            # Scale change requires sampler reset to be safe
            return True

        return self.min < old_config.min or self.max > old_config.max


class CategoricalParameterConfig(
    BaseParameterConfig, Generic[Category], tag="categorical", frozen=True
):
    """Configuration for categorical parameters with uniform selection."""

    categories: tuple[Category]

    def __post_init__(self):
        """Validate category list."""
        if not self.categories:
            raise ValueError("`categories` cannot be empty")

    @property
    def n_categories(self) -> int:
        """Get number of categories."""
        return len(self.categories)

    @overload
    def unnormalize(self, u: float) -> Category: ...

    @overload
    def unnormalize(self, u: FloatArray) -> npt.NDArray: ...

    def unnormalize(self, u: float | FloatArray) -> Category | npt.NDArray:
        """Transform normalized values to categorical parameter values."""
        self.validate_unit_float(u)
        idx = uniform_to_category(u, self.n_categories)
        if isinstance(u, np.ndarray) and u.ndim > 0:
            return np.array([self.categories[int(i)] for i in idx.flatten()]).reshape(u.shape)
        return self.categories[int(idx)]

    @overload
    def normalize(self, x: Category) -> float: ...

    @overload
    def normalize(self, x: npt.NDArray) -> FloatArray: ...

    def normalize(self, x: Category | npt.NDArray) -> float | FloatArray:
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
        if isinstance(x, np.ndarray):
            return all(xi in self.categories for xi in x.flat)
        else:
            return x in self.categories

    def has_expanded_domain(self, old_config: "CategoricalParameterConfig") -> bool:
        if not isinstance(old_config, CategoricalParameterConfig):
            raise TypeError(f"Cannot compare with config of type {type(old_config)}")

        return not set(old_config.categories).issuperset(self.categories)


class IntegerParameterConfig(BaseParameterConfig, tag="integer", frozen=True):
    """Configuration for integer parameters with uniform steps."""

    min: int
    max: int

    def __post_init__(self):
        """Validate integer bounds."""
        if self.min > self.max:
            raise ValueError(f"min ({self.min}) must be <= max ({self.max})")

    @property
    def num_values(self) -> int:
        """Get number of possible integer values."""
        return self.max - self.min + 1

    @overload
    def unnormalize(self, u: float) -> int: ...

    @overload
    def unnormalize(self, u: FloatArray) -> IntArray: ...

    def unnormalize(self, u: float | FloatArray) -> int | IntArray:
        """Transform normalized values to integer parameter values."""
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
        if not self.is_feasible(x):
            raise ParamInfeasibleError(f"Values outside bounds [{self.min}, {self.max}]")

        x = np.asarray(x)
        x_int = np.round(x).astype(int)
        indices = x_int - self.min
        return (indices + 0.5) / self.num_values

    def is_feasible(self, x: int | IntArray) -> bool:
        x = np.asarray(x)
        x_int = np.round(x).astype(int)
        return bool(np.all(x_int >= self.min) and np.all(x_int <= self.max))

    def has_expanded_domain(self, old_config: "IntegerParameterConfig") -> bool:
        if not isinstance(old_config, IntegerParameterConfig):
            raise TypeError(f"Cannot compare with config of type {type(old_config)}")

        return self.min < old_config.min or self.max > old_config.max


class LatticeParameterConfig(BaseParameterConfig, tag="lattice", frozen=True):
    """Configuration for lattice (evenly-spaced grid) parameters."""

    min: float
    max: float
    num_values: int

    def __post_init__(self):
        """Validate lattice configuration."""
        if self.max < self.min + MIN_FLOAT_TOLERANCE:
            raise ValueError("max must be strictly greater than min")
        if self.num_values <= 1:
            raise ValueError("num_values must be greater than 1")

    @property
    def step_size(self) -> float:
        """Get size of steps between lattice points."""
        return (self.max - self.min) / (self.num_values - 1)

    @overload
    def unnormalize(self, u: float) -> float: ...

    @overload
    def unnormalize(self, u: FloatArray) -> FloatArray: ...

    def unnormalize(self, u: float | FloatArray) -> float | FloatArray:
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
        if not self.is_feasible(x):
            raise ParamInfeasibleError(
                f"Values not on lattice or outside bounds [{self.min}, {self.max}]"
            )

        x = np.asarray(x)
        indices = np.round((x - self.min) / self.step_size)
        return (indices + 0.5) / self.num_values

    def is_feasible(self, x: float | FloatArray) -> bool:
        x = np.asarray(x)
        if np.any(x < self.min) or np.any(x > self.max):
            return False

        # Check if values are close to lattice points
        indices = np.round((x - self.min) / self.step_size)
        lattice_values = self.min + indices * self.step_size
        return np.allclose(x, lattice_values)

    def has_expanded_domain(self, old_config: "LatticeParameterConfig") -> bool:
        if not isinstance(old_config, LatticeParameterConfig):
            raise TypeError(f"Cannot compare with config of type {type(old_config)}")

        # Consider expansion if bounds expanded or grid got finer
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


@dataclass
class ParameterTransformer:
    """Transforms parameter values between normalized and actual spaces."""

    parameters: dict[ParameterName, ParameterConfig]

    @classmethod
    def from_dict(cls, params_dict: dict[ParameterName, dict[str, Any]]) -> "ParameterTransformer":
        parameters = msgspec.convert(params_dict, dict[ParameterName, PredefinedParameterConfig])
        return cls(parameters)

    @classmethod
    def from_json(cls, json_str: str) -> "ParameterTransformer":
        parameters = msgspec.json.decode(
            json_str, type=dict[ParameterName, PredefinedParameterConfig]
        )
        return cls(parameters)

    @property
    def param_names(self) -> list[ParameterName]:
        return list(self.parameters.keys())

    @property
    def num_params(self) -> int:
        return len(self.parameters)

    def unnormalize(self, normalized_params: npt.NDArray) -> list[dict[ParameterName, Any]]:
        if normalized_params.ndim != 2 or normalized_params.shape[1] != self.num_params:
            raise ValueError(
                f"Expected array of shape (n_samples, {self.num_params}), "
                f"got {normalized_params.shape}"
            )

        n_samples = normalized_params.shape[0]
        param_dicts = []

        for i in range(n_samples):
            param_dict = {
                name: self.parameters[name].unnormalize(normalized_params[i, j])
                for j, name in enumerate(self.param_names)
            }
            param_dicts.append(param_dict)

        return param_dicts

    def normalize(self, param_dicts: list[dict[ParameterName, Any]]) -> npt.NDArray:
        if not param_dicts:
            raise ValueError("Empty parameter list")

        n_samples = len(param_dicts)
        normalized = np.zeros((n_samples, self.num_params))

        # Validate first set of parameters
        if set(param_dicts[0].keys()) != set(self.param_names):
            raise ValueError("Parameter names do not match configuration")

        # Process each sample
        for i, param_dict in enumerate(param_dicts):
            if set(param_dict.keys()) != set(self.param_names):
                raise ValueError(f"Parameter names in sample {i} do not match configuration")

            for j, name in enumerate(self.param_names):
                normalized[i, j] = self.parameters[name].normalize(param_dict[name])

        return normalized

    def is_feasible(self, param_dict: dict[ParameterName, Any]) -> bool:
        if set(param_dict.keys()) != set(self.param_names):
            return False

        return all(self.parameters[name].is_feasible(value) for name, value in param_dict.items())

    def has_expanded_domain(self, old_transformer: "ParameterTransformer") -> bool:
        # Parameter sets must match
        if set(self.parameters) != set(old_transformer.parameters):
            raise ValueError("Parameter sets don't match")

        for name, new_config in self.parameters.items():
            old_config = old_transformer.parameters[name]
            if not isinstance(old_config, type(new_config)):
                raise TypeError(
                    f"Parameter {name} has incompatible types: {type(new_config)} vs {type(old_config)}"
                )
            if new_config.has_expanded_domain(old_config):  # type: ignore
                return True
        return False
