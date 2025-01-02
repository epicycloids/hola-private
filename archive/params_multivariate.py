"""Parameter configuration and transformation utilities.

This module provides a flexible system for defining and transforming hyperparameters
between normalized [0, 1] space and their actual parameter spaces. It supports:

- Continuous parameters (linear or log scale)
- Integer parameters (uniform steps)
- Categorical parameters (uniform selection)
- Lattice parameters (evenly spaced grid points)

Example:
    >>> # Define parameter configurations
    >>> params = {
    ...     "learning_rate": create_param_config(
    ...         "continuous",
    ...         min=1e-4,
    ...         max=1e-1,
    ...         scale="log"
    ...     ),
    ...     "batch_size": create_param_config(
    ...         "integer",
    ...         min=16,
    ...         max=256
    ...     ),
    ...     "optimizer": create_param_config(
    ...         "categorical",
    ...         categories=["adam", "sgd", "rmsprop"]
    ...     )
    ... }
    >>> # Create transformer
    >>> transformer = ParameterTransformer(params)
    >>> # Transform normalized values to actual parameter values
    >>> actual = transformer.transform_normalized_params([0.5, 0.7, 0.3])
    >>> print(actual)
    {'learning_rate': np.float64(0.003162277660168379), 'batch_size': 184, 'optimizer': 'adam'}
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Final, Generic, Literal, NewType, Protocol, TypeVar, Union, cast

import numpy as np
import numpy.typing as npt
from pydantic import Field, model_validator

from hola.core.utils import MIN_FLOAT_TOLERANCE, BaseConfig, uniform_to_category

# Type definitions
ParamName = NewType("ParamName", str)
Category = TypeVar("Category")


class Scale(str, Enum):
    """Type of scaling to apply to continuous parameters.

    :cvar LINEAR: Linear scaling between min and max
    :cvar LOG: Logarithmic scaling between min and max
    """

    LINEAR = "linear"
    LOG = "log"


class ParamConfig(Protocol):
    """Protocol defining the interface for parameter configurations."""

    def transform_param(self, u: float) -> Any:
        """Transform normalized parameter value to actual parameter space.

        :param u: Normalized value in [0, 1]
        :type u: float
        :return: Parameter value in its actual space
        :rtype: Any
        :raises ValueError: If u is not in [0, 1]
        """
        ...

    def back_transform_param(self, x: Any) -> float:
        """Transform parameter value from actual space to normalized [0, 1] space.

        :param x: Parameter value in its actual space
        :type x: Any
        :return: Normalized value in [0, 1]
        :rtype: float
        :raises ValueError: If x is not a valid parameter value
        """
        ...


class ParamInfeasibleError(ValueError):
    """Raised when a parameter value is infeasible under the current configuration."""

    pass


class ContinuousParamConfig(BaseConfig):
    """Configuration for continuous numerical parameters.

    Supports both linear and logarithmic scaling between minimum and maximum
    values. This class can handle both scalar and vectorized (NumPy array)
    inputs for transforming and back-transforming parameter values.

    :param min: Minimum parameter value
    :type min: float
    :param max: Maximum parameter value
    :type max: float
    :param scale: Scaling type (linear or logarithmic)
    :type scale: Scale
    :raises ValueError: If max is not strictly greater than min, or if log
        scale is used with non-positive bounds.
    :raises TypeError: If inputs are not of the correct type.

    Example:
        >>> # Linear scale learning rate from 0.0 to 0.1
        >>> linear_param = ContinuousParamConfig(min=0.0, max=0.1)
        >>> linear_param.transform_param(0.5)
        0.05
        >>> linear_param.transform_param(np.array([0.0, 0.5, 1.0]))
        array([0.  , 0.05, 0.1 ])
        >>> # Log scale learning rate from 1e-4 to 1e-1
        >>> log_param = ContinuousParamConfig(min=1e-4, max=1e-1, scale=Scale.LOG)
        >>> log_param.transform_param(0.5)
        0.0031622776601683794
        >>> log_param.transform_param(np.array([0.0, 0.5, 1.0]))
        array([1.e-04, 3.16227766e-03, 1.e-01])
    """

    min: float = Field(..., description="Minimum parameter value")
    max: float = Field(..., description="Maximum parameter value")
    scale: Scale = Field(default=Scale.LINEAR, description="Scaling type")

    @model_validator(mode="after")
    def validate_bounds(self) -> "ContinuousParamConfig":
        """Validate parameter bounds and scaling."""
        if self.max < self.min + MIN_FLOAT_TOLERANCE:
            raise ValueError("max must be strictly greater than min")
        if self.scale == Scale.LOG and (self.min <= 0 or self.max <= 0):
            raise ValueError("log scale requires strictly positive bounds")
        return self

    @classmethod
    def parse(cls, data: str | bytes | dict[str, Any] | BaseConfig) -> "ContinuousParamConfig":
        """Parse input data into an ContinuousParamConfig.

        Overrides BaseConfig.parse to provide correct return type annotation.
        """
        return super().parse(data)  # type: ignore

    def transform_param(self, u: npt.ArrayLike) -> npt.NDArray[np.floating]:
        """Transform normalized value(s) to continuous parameter value(s).

        :param u: Normalized value(s) in the range [0, 1]. Can be a scalar
            (float) or a NumPy array.
        :type u: ArrayLike
        :raises ValueError: If any value in u is outside the range [0, 1].
        :return: Transformed parameter value(s). Returns a float if the input
            was a float, a NumPy array otherwise.
        :rtype: npt.NDArray[np.floating]
        """
        u = np.asarray(u)

        # Type validation
        if not np.issubdtype(u.dtype, np.number):
            raise TypeError(f"u must contain numeric values, got {u.dtype}")

        if np.any(u < 0.0) or np.any(u > 1.0):
            raise ValueError(f"Normalized values must be in [0, 1], got {u[(u < 0.0) | (u > 1.0)]}")

        if self.scale == Scale.LINEAR:
            return self.min + u * (self.max - self.min)
        else:  # log scale
            return self.min * np.exp(u * np.log(self.max / self.min))

    def back_transform_param(self, x: npt.ArrayLike) -> npt.NDArray[np.floating]:
        """Transform continuous parameter value(s) to normalized value(s).

        :param x: Parameter value(s) within the bounds [min, max]. Can be a
            scalar (float) or a NumPy array.
        :type x: ArrayLike
        :raises ParamInfeasibleError: If any value in x is outside the bounds
            [min, max].
        :return: Normalized value(s) in the range [0, 1]. Returns a float if
            the input was a float, a NumPy array otherwise.
        :rtype: npt.NDArray[np.floating]
        """
        x = np.asarray(x)

        # Type validation
        if not np.issubdtype(x.dtype, np.number):
            raise TypeError(f"x must contain numeric values, got {x.dtype}")

        if np.any(x < self.min) or np.any(x > self.max):
            raise ParamInfeasibleError(
                f"Values outside bounds [{self.min}, {self.max}]: {x[(x < self.min) | (x > self.max)]}"
            )

        if self.scale == Scale.LINEAR:
            return (x - self.min) / (self.max - self.min)
        else:  # log scale
            return np.log(x / self.min) / np.log(self.max / self.min)


class CategoricalParamConfig(BaseConfig, Generic[Category]):
    """Configuration for categorical parameters with uniform selection.

    This class supports both scalar and vectorized (NumPy array) inputs for
    transforming and back-transforming categorical parameter values.

    :param categories: List of valid values.
    :type categories: list[T]
    :raises ValueError: If categories is an empty list.

    Example:
        >>> # Define optimizer selection
        >>> opt_param = CategoricalParamConfig(categories=["adam", "sgd"])
        >>> opt_param.transform_param(0.7)
        'sgd'
        >>> opt_param.transform_param(np.array([0.2, 0.8]))
        array(['adam', 'sgd'], dtype='<U4')
        >>> opt_param.back_transform_param("adam")
        0.25
        >>> opt_param.back_transform_param(np.array(["adam", "sgd"]))
        array([0.25, 0.75])
    """

    categories: list[Category] = Field(..., min_length=1, description="List of valid values")

    @model_validator(mode="after")
    def validate_categories(self) -> "CategoricalParamConfig":
        """Validate that categories is not empty."""
        if not self.categories:
            raise ValueError("`categories` cannot be empty")
        return self

    @classmethod
    def parse(cls, data: str | bytes | dict[str, Any] | BaseConfig) -> "CategoricalParamConfig":
        """Parse input data into an CategoricalParamConfig.

        Overrides BaseConfig.parse to provide correct return type annotation.
        """
        return super().parse(data)  # type: ignore

    @property
    def n_categories(self) -> int:
        """Get number of categories."""
        return len(self.categories)

    def transform_param(self, u: npt.ArrayLike) -> Category | npt.NDArray[Any]:
        """Transform normalized value(s) to categorical parameter value(s).

        :param u: Normalized value(s) in the range [0, 1]. Can be a scalar
            (float) or a NumPy array.
        :type u: npt.ArrayLike
        :raises ValueError: If any value in u is outside the range [0, 1].
        :raises TypeError: If u does not contain numeric values.
        :return: Transformed categorical parameter value(s). The return type
            matches the input type - if a scalar is passed, a single category
            is returned. If a NumPy array is passed, an array of the same shape
            is returned, containing the corresponding categories.
        :rtype: T | np.ndarray
        """
        u = np.asarray(u)

        # Type validation using np.issubdtype for numeric types
        if not np.issubdtype(u.dtype, np.number):
            raise TypeError(f"u must contain numeric values, got {u.dtype}")

        if np.any(u < 0.0) or np.any(u > 1.0):
            raise ValueError(f"Normalized values must be in [0, 1], got {u[(u < 0.0) | (u > 1.0)]}")

        indices = uniform_to_category(u, self.n_categories)

        if indices.ndim == 0:  # Scalar input
            return self.categories[indices]
        else:  # Array input
            return np.array([self.categories[i] for i in indices])

    def back_transform_param(self, x: Any | npt.ArrayLike) -> npt.NDArray[np.floating]:
        """Transform categorical parameter value(s) to normalized value(s).

        :param x: Categorical parameter value(s). Can be a single value or a
            NumPy array of values.
        :type x: Any | npt.ArrayLike
        :raises ParamInfeasibleError: If any value in x is not found in the
            list of valid categories.
        :return: Normalized value(s) in the range [0, 1]. Returns a NumPy array
            of floats.
        :rtype: npt.NDArray[np.floating]
        """
        if isinstance(x, np.ndarray):
            # Handle NumPy array input
            transformed_values = []
            for val in x:
                try:
                    idx = self.categories.index(val)
                except ValueError:
                    valid_cats = ", ".join(repr(c) for c in self.categories)
                    raise ParamInfeasibleError(
                        f"Value {repr(val)} not in categories: [{valid_cats}]"
                    ) from None
                transformed_values.append((idx + 0.5) / self.n_categories)
            return np.array(transformed_values)
        else:
            # Handle single value input
            try:
                idx = self.categories.index(x)
            except ValueError:
                valid_cats = ", ".join(repr(c) for c in self.categories)
                raise ParamInfeasibleError(
                    f"Value {repr(x)} not in categories: [{valid_cats}]"
                ) from None
            return np.array([(idx + 0.5) / self.n_categories])


class IntegerParamConfig(BaseConfig):
    """Configuration for integer parameters with uniform steps.

    This class supports both scalar and vectorized (NumPy array) inputs for
    transforming and back-transforming integer parameter values.

    :param min: Minimum integer value
    :type min: int
    :param max: Maximum integer value
    :type max: int
    :raises ValueError: If min is greater than max.

    Example:
        >>> # Define batch size parameter
        >>> batch_param = IntegerParamConfig(min=16, max=256)
        >>> batch_param.transform_param(0.5)
        136
        >>> batch_param.transform_param(np.array([0.0, 0.5, 1.0]))
        array([ 16, 136, 256])
        >>> batch_param.back_transform_param(16)
        0.0020491803278688524
        >>> batch_param.back_transform_param(np.array([16, 136, 256]))
        array([0.00204918, 0.5       , 0.99795082])
    """

    min: int = Field(..., description="Minimum integer value")
    max: int = Field(..., description="Maximum integer value")

    @model_validator(mode="after")
    def validate_bounds(self) -> "IntegerParamConfig":
        """Validate that min <= max."""
        if self.min > self.max:
            raise ValueError(f"min ({self.min}) must be <= max ({self.max})")
        return self

    @classmethod
    def parse(cls, data: str | bytes | dict[str, Any] | BaseConfig) -> "IntegerParamConfig":
        """Parse input data into an IntegerParamConfig.

        Overrides BaseConfig.parse to provide correct return type annotation.
        """
        return super().parse(data)  # type: ignore

    @property
    def num_values(self) -> int:
        """Get number of possible integer values."""
        return self.max - self.min + 1

    def transform_param(self, u: npt.ArrayLike) -> npt.NDArray[np.int_]:
        """Transform normalized value(s) to integer parameter value(s).

        :param u: Normalized value(s) in the range [0, 1]. Can be a scalar
            (float) or a NumPy array.
        :type u: ArrayLike
        :raises ValueError: If any value in u is outside the range [0, 1].
        :raises TypeError: If u does not contain numeric values.
        :return: Transformed integer parameter value(s). Returns an int if the
            input was a float, a NumPy array otherwise.
        :rtype: npt.NDArray[np.int_]
        """
        u = np.asarray(u)

        # Type validation using np.issubdtype for numeric types
        if not np.issubdtype(u.dtype, np.number):
            raise TypeError(f"u must contain numeric values, got {u.dtype}")

        if np.any(u < 0.0) or np.any(u > 1.0):
            raise ValueError(f"Normalized values must be in [0, 1], got {u[(u < 0.0) | (u > 1.0)]}")

        indices = uniform_to_category(u, self.num_values)
        return self.min + indices

    def back_transform_param(self, x: npt.ArrayLike) -> npt.NDArray[np.floating]:
        """Transform integer parameter value(s) to normalized value(s).

        Handles various numeric types by converting to nearest integer.

        :param x: Parameter value(s). Can be a scalar (float, int) or a NumPy
            array.
        :type x: ArrayLike
        :raises ValueError: If any value cannot be converted to an integer or
            is out of bounds.
        :return: Normalized value(s) in the range [0, 1]. Returns a NumPy array
            of floats.
        :rtype: npt.NDArray[np.floating]
        """
        x = np.asarray(x)

        # Convert to integer values
        try:
            x_int = np.round(x).astype(int)
        except ValueError:
            raise ValueError(f"Cannot convert values to integers: {x}")

        # Type validation
        if not np.all(np.equal(np.mod(x_int, 1), 0)):
            raise TypeError(f"x must contain integer or floating values, got {x.dtype}")

        if np.any(x_int < self.min) or np.any(x_int > self.max):
            raise ParamInfeasibleError(
                f"Values outside bounds [{self.min}, {self.max}]: {x_int[(x_int < self.min) | (x_int > self.max)]}"
            )

        indices = x_int - self.min
        return (indices + 0.5) / self.num_values


class LatticeParamConfig(BaseConfig):
    """Configuration for lattice (evenly-spaced grid) parameters.

    This class supports both scalar and vectorized (NumPy array) inputs for
    transforming and back-transforming lattice parameter values.

    :param min: Minimum value
    :type min: float
    :param max: Maximum value
    :type max: float
    :param num_values: Number of lattice points
    :type num_values: int
    :raises ValueError: If max is not strictly greater than min or if
        num_values is not greater than 1.

    Example:
        >>> # Define dropout rate with 5 possible values
        >>> dropout_param = LatticeParamConfig(min=0.0, max=0.5, num_values=5)
        >>> dropout_param.transform_param(0.5)
        0.25
        >>> dropout_param.transform_param(np.array([0.0, 0.5, 1.0]))
        array([0.  , 0.25, 0.5 ])
        >>> dropout_param.back_transform_param(0.25)
        0.5
        >>> dropout_param.back_transform_param(np.array([0.0, 0.25, 0.5]))
        array([0.1, 0.5, 0.9])
    """

    min: float = Field(..., description="Minimum value")
    max: float = Field(..., description="Maximum value")
    num_values: int = Field(gt=1, description="Number of lattice points")

    @model_validator(mode="after")
    def validate_bounds(self) -> "LatticeParamConfig":
        """Validate parameter bounds."""
        if self.max < self.min + MIN_FLOAT_TOLERANCE:
            raise ValueError("max must be strictly greater than min")
        return self

    @classmethod
    def parse(cls, data: str | bytes | dict[str, Any] | BaseConfig) -> "LatticeParamConfig":
        """Parse input data into an LatticeParamConfig.

        Overrides BaseConfig.parse to provide correct return type annotation.
        """
        return super().parse(data)  # type: ignore

    @property
    def step_size(self) -> float:
        """Get size of steps between lattice points."""
        return (self.max - self.min) / (self.num_values - 1)

    def transform_param(self, u: npt.ArrayLike) -> npt.NDArray[np.floating]:
        """Transform normalized value(s) to lattice parameter value(s).

        :param u: Normalized value(s) in the range [0, 1]. Can be a scalar
            (float) or a NumPy array.
        :type u: npt.ArrayLike
        :raises ValueError: If any value in u is outside the range [0, 1].
        :raises TypeError: If u does not contain numeric values.
        :return: Transformed lattice parameter value(s). Returns a NumPy array
            of floats.
        :rtype: npt.NDArray[np.floating]
        """
        u = np.asarray(u)

        # Type validation using np.issubdtype for numeric types
        if not np.issubdtype(u.dtype, np.number):
            raise TypeError(f"u must contain numeric values, got {u.dtype}")

        if np.any(u < 0.0) or np.any(u > 1.0):
            raise ValueError(f"Normalized values must be in [0, 1], got {u[(u < 0.0) | (u > 1.0)]}")

        indices = uniform_to_category(u, self.num_values)
        return self.min + indices * self.step_size

    def back_transform_param(self, x: npt.ArrayLike) -> npt.NDArray[np.floating]:
        """Transform lattice parameter value(s) to normalized value(s).

        :param x: Lattice parameter value(s) within the bounds [min, max]. Can
            be a scalar (float) or a NumPy array.
        :type x: npt.ArrayLike
        :raises ParamInfeasibleError: If any value in x is outside the bounds
            [min, max].
        :raises TypeError: If x does not contain numeric values
        :return: Normalized value(s) in the range [0, 1]. Returns a NumPy array
            of floats.
        :rtype: npt.NDArray[np.floating]
        """
        x = np.asarray(x)

        # Type validation
        if not np.issubdtype(x.dtype, np.number):
            raise TypeError(f"x must contain numeric values, got {x.dtype}")

        if np.any(x < self.min) or np.any(x > self.max):
            raise ParamInfeasibleError(
                f"Values outside bounds [{self.min}, {self.max}]: {x[(x < self.min) | (x > self.max)]}"
            )

        indices = np.round((x - self.min) / self.step_size)
        return (indices + 0.5) / self.num_values


def create_param_config(
    param_type: Literal["continuous", "integer", "categorical", "lattice"], **kwargs: Any
) -> ParamConfig:
    """Create a parameter configuration from a specification.

    Example:
        >>> # Create a continuous parameter
        >>> lr_param = create_param_config(
        ...     "continuous",
        ...     min=1e-4,
        ...     max=1e-1,
        ...     scale=Scale.LOG
        ... )
        >>>
        >>> # Create a categorical parameter
        >>> opt_param = create_param_config(
        ...     "categorical",
        ...     categories=["adam", "sgd", "rmsprop"]
        ... )

    :param param_type: Type of parameter configuration to create
    :type param_type: str
    :param kwargs: Configuration parameters specific to the parameter type
    :type kwargs: Any
    :return: Configured parameter
    :rtype: ParamConfigProtocol
    :raises ValueError: If param_type is invalid or if required kwargs are missing
    """
    config_classes = {
        "continuous": ContinuousParamConfig,
        "integer": IntegerParamConfig,
        "categorical": CategoricalParamConfig,
        "lattice": LatticeParamConfig,
    }

    if param_type not in config_classes:
        valid_types = ", ".join(config_classes.keys())
        raise ValueError(f"param_type must be one of: {valid_types}")

    try:
        return cast(ParamConfig, config_classes[param_type](**kwargs))
    except Exception as e:
        raise ValueError(f"Failed to create {param_type} parameter: {str(e)}") from e


class ParameterTransformer:
    """Transforms parameter values between normalized and actual spaces.

    This class now supports vectorized operations, allowing you to transform
    multiple sets of parameters at once. It distinguishes between scalar and
    vectorized inputs based on the presence of at least one NumPy array with
    more than one element.

    Example:
        >>> params = {
        ...     "learning_rate": ContinuousParamConfig(
        ...         min=1e-4, max=1e-1, scale="log"
        ...     ),
        ...     "batch_size": IntegerParamConfig(
        ...         min=16, max=256
        ...     ),
        ...     "optimizer": CategoricalParamConfig(
        ...         categories=["adam", "sgd", "rmsprop"]
        ...     )
        ... }
        >>> transformer = ParameterTransformer(params)
        >>> normalized_params = np.array([[0.5, 0.7, 0.3], [0.1, 0.9, 0.6]])
        >>> transformed_params = transformer.transform_normalized_params(normalized_params)
        >>> print(transformed_params)
        {'learning_rate': array([0.00316228, 0.00025119]), 'batch_size': array([184, 232]), 'optimizer': array(['adam', 'rmsprop'], dtype='<U7')}

        >>> param_values = {"learning_rate": 0.01, "batch_size": 32, "optimizer": "sgd"}
        >>> normalized_values = transformer.back_transform_param_dict(param_values)
        >>> print(normalized_values)
        [0.66847896 0.06967213 0.75      ]

        >>> param_values_batch = {
        ...     "learning_rate": np.array([0.01, 0.001]),
        ...     "batch_size": np.array([32, 64]),
        ...     "optimizer": np.array(["sgd", "adam"]),
        ... }
        >>> normalized_values_batch = transformer.back_transform_param_dict(param_values_batch)
        >>> print(normalized_values_batch)
        [[0.66847896 0.06967213 0.75      ]
         [0.48377225 0.20081967 0.25      ]]

        >>> param_values_batch_mixed = {
        ...     "learning_rate": np.array([0.01]),
        ...     "batch_size": 64,
        ...     "optimizer": "adam",
        ... }
        >>> normalized_values_batch_mixed = transformer.back_transform_param_dict(param_values_batch_mixed)
        >>> print(normalized_values_batch_mixed)
        [[0.66847896 0.20081967 0.25      ]]
    """

    def __init__(self, param_configs: dict[ParamName, ParamConfig]):
        """Initialize transformer with parameter configurations.

        :param param_configs: Mapping of parameter names to configurations
        :type param_configs: dict[str, ParamConfigProtocol]
        """
        self.param_configs = param_configs
        self.param_names = list(param_configs.keys())
        self.num_params = len(param_configs)

    def transform_normalized_params(
        self, normalized_params: npt.NDArray[np.floating]
    ) -> dict[ParamName, Any]:
        """Transform normalized parameter values to actual parameter values.

        :param normalized_params: Normalized parameter values, where each row
            represents a different set of parameters.
        :type normalized_params: npt.NDArray[np.floating]
        :raises ValueError: If the number of parameters in `normalized_params`
            does not match the expected number.
        :return: A dictionary where keys are parameter names and values are
            NumPy arrays containing the transformed parameter values.
        :rtype: dict[str, Any]
        """
        if normalized_params.ndim == 1:
            normalized_params = normalized_params.reshape(1, -1)

        if normalized_params.shape[1] != self.num_params:
            raise ValueError(
                f"Expected {self.num_params} parameters, got {normalized_params.shape[1]}"
            )

        transformed_params = {}
        for i, name in enumerate(self.param_names):
            transformed_params[name] = self.param_configs[name].transform_param(
                normalized_params[:, i]
            )

        return transformed_params

    def back_transform_param_dict(
        self, param_values: dict[ParamName, Any]
    ) -> npt.NDArray[np.floating]:
        """Transform actual parameter values to normalized values.

        :param param_values: A dictionary where keys are parameter names and
            values are either single parameter values or NumPy arrays of
            parameter values.
        :type param_values: dict[str, Any]
        :raises ValueError: If the parameter names in `param_values` do not
            match the configuration or if all parameter value arrays do not
            have the same shape.
        :return: A NumPy array where each row represents a set of normalized
            parameter values.
        :rtype: npt.NDArray[np.floating]
        """
        if set(param_values.keys()) != set(self.param_configs.keys()):
            raise ValueError("Parameter names in `param_values` do not match configuration")

        # Check if any values are NumPy arrays with more than one element
        has_multielement_arrays = any(
            isinstance(val, np.ndarray) and val.size > 1 for val in param_values.values()
        )

        if has_multielement_arrays:
            # Vectorized back-transformation
            first_array_shape = None
            for name, val in param_values.items():
                if isinstance(val, np.ndarray):
                    if first_array_shape is None:
                        first_array_shape = val.shape
                    elif val.shape != first_array_shape:
                        raise ValueError(
                            "All parameter value arrays must have the same shape. "
                            f"Parameter '{name}' has shape {val.shape}, "
                            f"but expected {first_array_shape}."
                        )
                # Ensure all values are arrays of the same shape
                else:
                    raise ValueError(
                            "All parameter value arrays must have the same shape. "
                            f"Parameter '{name}' is not a NumPy array but , "
                            f"at least one parameter is a NumPy array of size >= 2."
                        )

            num_sets = first_array_shape[0]
            normalized_params = np.zeros((num_sets, self.num_params))
            for i, name in enumerate(self.param_names):
                normalized_params[:, i] = self.param_configs[name].back_transform_param(
                    param_values[name]
                )

        else:
            # Scalar back-transformation
            normalized_params = np.zeros((1, self.num_params))
            for i, name in enumerate(self.param_names):
                normalized_params[0, i] = self.param_configs[name].back_transform_param(
                    param_values[name]
                )

        return normalized_params.squeeze()
