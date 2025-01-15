"""Parameter configuration and transformation utilities.

This module provides a flexible system for defining and transforming hyperparameters
between normalized [0, 1] space and their actual parameter spaces. It supports:

- Continuous parameters (linear or log scale)
- Integer parameters (uniform steps)
- Categorical parameters (uniform selection)
- Lattice parameters (evenly spaced grid points)

The module implements bidirectional transformations between normalized space and
actual parameter values, with robust error handling and validation.

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
from typing import Annotated, Any, Generic, Literal, NewType, TypeAlias, TypeVar, Union

import numpy as np
from pydantic import Field, FiniteFloat, model_validator

from hola.core.utils import MIN_FLOAT_TOLERANCE, BaseConfig, uniform_to_category

# Type definitions
ParamName: TypeAlias = str
Category = TypeVar("Category")


class Scale(str, Enum):
    """Type of scaling to apply to continuous parameters.

    :cvar LINEAR: Use linear scaling between min and max values
    :cvar LOG: Use logarithmic scaling between min and max values
    """

    LINEAR = "linear"
    LOG = "log"


class BaseParamConfig(BaseConfig):
    """Base class for all parameter configurations."""

    type: str = Field(..., frozen=True)


    @staticmethod
    def validate_unit_float(u: float) -> None:
        """Validate that a value is a float in [0, 1]."""
        if not isinstance(u, (int, float)):
            raise TypeError(f"Value must be a number, got {type(u)}")
        if u < 0.0 or u > 1.0:
            raise ValueError(f"Value must be in [0, 1], got {u}")

    def transform_param(self, u: float) -> Any:
        """Transform normalized parameter value to actual parameter space."""
        raise NotImplementedError

    def back_transform_param(self, x: Any) -> float:
        """Transform parameter value from actual space to normalized [0, 1] space."""
        raise NotImplementedError


class ParamInfeasibleError(ValueError):
    """Raised when a parameter value is infeasible under the current configuration.

    This exception indicates that a parameter value cannot be represented within
    the constraints of its configuration (e.g., outside bounds, invalid category).
    """

    pass


class ContinuousParamConfig(BaseParamConfig):
    """Configuration for continuous numerical parameters.

    Handles continuous parameters with either linear or logarithmic scaling
    between minimum and maximum bounds. Logarithmic scaling is particularly
    useful for parameters that vary over multiple orders of magnitude.

    Example:
        >>> # Linear scaling for learning rate
        >>> config = ContinuousParamConfig(min=0.0, max=1.0, scale="linear")
        >>> config.transform_param(0.5)
        0.5
        >>> # Log scaling for regularization strength
        >>> config = ContinuousParamConfig(min=1e-4, max=1e-1, scale="log")
        >>> config.transform_param(0.5)
        np.float64(0.003162277660168379)
    """

    min: FiniteFloat = Field(..., description="Minimum parameter value")
    max: FiniteFloat = Field(..., description="Maximum parameter value")
    scale: Scale = Field(default=Scale.LINEAR, description="Scaling type")

    type: Literal["continuous"] = Field("continuous", frozen=True)

    @model_validator(mode="after")
    def validate_bounds(self) -> "ContinuousParamConfig":
        """Validate configuration bounds and scaling compatibility.

        :return: Validated configuration
        :rtype: ContinuousParamConfig
        :raises ValueError: If bounds are invalid or log scale is used with
            non-positive bounds
        """
        if self.max < self.min + MIN_FLOAT_TOLERANCE:
            raise ValueError("max must be strictly greater than min")
        if self.scale == Scale.LOG and (self.min <= 0 or self.max <= 0):
            raise ValueError("log scale requires strictly positive bounds")
        return self

    def transform_param(self, u: float) -> float:
        """Transform normalized value to continuous parameter value.

        :param u: Normalized value in [0, 1]
        :type u: float
        :return: Continuous parameter value
        :rtype: float
        :raises TypeError: If u is not a number
        :raises ValueError: If u is not in [0, 1]
        """
        self.validate_unit_float(u)
        if self.scale == Scale.LINEAR:
            return self.min + u * (self.max - self.min)
        else:
            return float(self.min * np.exp(u * np.log(self.max / self.min)))

    def back_transform_param(self, x: float) -> float:
        """Transform continuous parameter value to normalized value.

        :param x: Parameter value in actual space
        :type x: float
        :return: Normalized value in [0, 1]
        :rtype: float
        :raises TypeError: If x is not a number
        :raises ParamInfeasibleError: If value is outside bounds
        """
        if not isinstance(x, (int, float)):
            raise TypeError(f"x must be a number, got {type(x)}")
        if x < self.min or x > self.max:
            raise ParamInfeasibleError(f"Value {x} outside bounds [{self.min}, {self.max}]")

        if self.scale == Scale.LINEAR:
            return (x - self.min) / (self.max - self.min)
        else:
            return float(np.log(x / self.min) / np.log(self.max / self.min))


class CategoricalParamConfig(BaseParamConfig, Generic[Category]):
    """Configuration for categorical parameters with uniform selection.

    Handles categorical parameters by mapping normalized values to categories
    using uniform intervals. Categories are assigned equal probability mass
    in the normalized space.

    Example:
        >>> config = CategoricalParamConfig(categories=["adam", "sgd", "rmsprop"])
        >>> config.transform_param(0.7)
        'rmsprop'
        >>> config.back_transform_param("adam")
        0.16666666666666666

    """

    categories: list[Category] = Field(..., min_length=1, description="List of valid values")

    type: Literal["categorical"] = Field("categorical", frozen=True)

    @model_validator(mode="after")
    def validate_categories(self) -> "CategoricalParamConfig":
        """Validate category list.

        :return: Validated configuration
        :rtype: CategoricalParamConfig
        :raises ValueError: If category list is empty
        """
        if not self.categories:
            raise ValueError("`categories` cannot be empty")
        return self

    @property
    def n_categories(self) -> int:
        """Get number of categories.

        :return: Number of valid categories
        :rtype: int
        """
        return len(self.categories)

    def transform_param(self, u: float) -> Category:
        """Transform normalized value to categorical parameter value.

        :param u: Normalized value in [0, 1]
        :type u: float
        :return: Selected category value
        :rtype: Category
        :raises TypeError: If u is not a number
        :raises ValueError: If u is not in [0, 1]
        """
        self.validate_unit_float(u)
        index = uniform_to_category(u, self.n_categories)
        return self.categories[index]

    def back_transform_param(self, x: Category) -> float:
        """Transform categorical parameter value to normalized value.

        :param x: Category value
        :type x: Category
        :return: Normalized value (category midpoint) in [0, 1]
        :rtype: float
        :raises ParamInfeasibleError: If category is not in valid set
        """
        try:
            idx = self.categories.index(x)
        except ValueError:
            valid_cats = ", ".join(repr(c) for c in self.categories)
            raise ParamInfeasibleError(
                f"Value {repr(x)} not in categories: [{valid_cats}]"
            ) from None
        return (idx + 0.5) / self.n_categories


class IntegerParamConfig(BaseParamConfig):
    """Configuration for integer parameters with uniform steps.

    Handles integer parameters with equally spaced steps between minimum and
    maximum values. Normalized values are mapped to the nearest integer within
    bounds.

    Example:
        >>> config = IntegerParamConfig(min=16, max=256)
        >>> config.transform_param(0.5)
        136
        >>> config.back_transform_param(64)
        0.2012448132780083
    """

    min: int = Field(..., description="Minimum integer value")
    max: int = Field(..., description="Maximum integer value")

    type: Literal["integer"] = Field("integer", frozen=True)

    @model_validator(mode="after")
    def validate_bounds(self) -> "IntegerParamConfig":
        """Validate integer bounds.

        :return: Validated configuration
        :rtype: IntegerParamConfig
        :raises ValueError: If min > max
        """
        if self.min > self.max:
            raise ValueError(f"min ({self.min}) must be <= max ({self.max})")
        return self

    @property
    def num_values(self) -> int:
        """Get number of possible integer values.

        :return: Number of integers between min and max (inclusive)
        :rtype: int
        """
        return self.max - self.min + 1

    def transform_param(self, u: float) -> int:
        """Transform normalized value to integer parameter value.

        :param u: Normalized value in [0, 1]
        :type u: float
        :return: Integer value
        :rtype: int
        :raises TypeError: If u is not a number
        :raises ValueError: If u is not in [0, 1]
        """
        self.validate_unit_float(u)
        index = uniform_to_category(u, self.num_values)
        return self.min + index

    def back_transform_param(self, x: int) -> float:
        """Transform integer parameter value to normalized value.

        :param x: Integer value
        :type x: int
        :return: Normalized value (category midpoint) in [0, 1]
        :rtype: float
        :raises TypeError: If x is not a number
        :raises ParamInfeasibleError: If value is outside bounds
        """
        if not isinstance(x, (int, float)):
            raise TypeError(f"x must be a number, got {type(x)}")
        x_int = int(round(x))
        if x_int < self.min or x_int > self.max:
            raise ParamInfeasibleError(f"Value {x_int} outside bounds [{self.min}, {self.max}]")

        index = x_int - self.min
        return (index + 0.5) / self.num_values


class LatticeParamConfig(BaseParamConfig):
    """Configuration for lattice (evenly-spaced grid) parameters.

    Handles continuous parameters that are constrained to a regular grid of
    values. This is useful for parameters that should only take on specific,
    evenly spaced values.

    Example:
        >>> config = LatticeParamConfig(min=0.0, max=1.0, num_values=5)
        >>> config.transform_param(0.5)
        0.5
        >>> config.back_transform_param(0.75)
        0.7
    """

    min: FiniteFloat = Field(..., description="Minimum value")
    max: FiniteFloat = Field(..., description="Maximum value")
    num_values: int = Field(gt=1, description="Number of lattice points")

    type: Literal["lattice"] = Field("lattice", frozen=True)

    @model_validator(mode="after")
    def validate_bounds(self) -> "LatticeParamConfig":
        """Validate lattice configuration.

        :return: Validated configuration
        :rtype: LatticeParamConfig
        :raises ValueError: If max <= min
        """
        if self.max < self.min + MIN_FLOAT_TOLERANCE:
            raise ValueError("max must be strictly greater than min")
        return self

    @property
    def step_size(self) -> float:
        """Get size of steps between lattice points.

        :return: Distance between adjacent lattice points
        :rtype: float
        """
        return (self.max - self.min) / (self.num_values - 1)

    def transform_param(self, u: float) -> float:
        """Transform normalized value to lattice parameter value.

        :param u: Normalized value in [0, 1]
        :type u: float
        :return: Value on lattice
        :rtype: float
        :raises TypeError: If u is not a number
        :raises ValueError: If u is not in [0, 1]
        """
        self.validate_unit_float(u)
        index = uniform_to_category(u, self.num_values)
        return self.min + index * self.step_size

    def back_transform_param(self, x: float) -> float:
        """Transform lattice parameter value to normalized value.

        Maps value to nearest lattice point and returns its normalized position.

        :param x: Value to transform
        :type x: float
        :return: Normalized value in [0, 1]
        :rtype: float
        :raises TypeError: If x is not a number
        :raises ParamInfeasibleError: If value is outside bounds
        """
        if not isinstance(x, (int, float)):
            raise TypeError(f"x must be a number, got {type(x)}")
        if x < self.min or x > self.max:
            raise ParamInfeasibleError(f"Value {x} outside bounds [{self.min}, {self.max}]")

        index = round((x - self.min) / self.step_size)
        return (index + 0.5) / self.num_values


ParamConfig = Annotated[
    Union[ContinuousParamConfig, CategoricalParamConfig, IntegerParamConfig, LatticeParamConfig],
    Field(discriminator="type"),
]


class ParameterTransformer(BaseConfig):
    """Transforms parameter values between normalized and actual spaces.

    This class handles bidirectional transformation between normalized [0, 1]
    space and actual parameter values for multiple parameters. It maintains
    the ordering of parameters and ensures consistent transformations.

    Example:
        >>> # Create transformer with multiple parameters
        >>> transformer = ParameterTransformer({
        ...     "learning_rate": ContinuousParamConfig(min=1e-4, max=1e-1),
        ...     "batch_size": IntegerParamConfig(min=16, max=256),
        ...     "optimizer": CategoricalParamConfig(categories=["adam", "sgd"])
        ... })
        >>> # Transform normalized values to parameter space
        >>> params = transformer.transform_normalized_params([0.5, 0.7, 0.3])
        >>> params
        {'learning_rate': 0.050050000000000004, 'batch_size': 184, 'optimizer': 'adam'}
        >>> # Transform back to normalized space
        >>> transformer.back_transform_param_dict(params)
        [0.5, 0.6991701244813278, 0.25]
    """

    params: dict[ParamName, ParamConfig]

    @property
    def param_names(self) -> list[ParamName]:
        return list(self.params.keys())

    @property
    def num_params(self) -> int:
        return len(self.params)

    def transform_normalized_params(self, normalized_params: list[float]) -> dict[ParamName, Any]:
        """Transform normalized parameter values to actual parameter values.

        :param normalized_params: List of normalized parameter values in [0, 1]
        :type normalized_params: list[float]
        :return: Dictionary mapping parameter names to their actual values
        :rtype: dict[ParamName, Any]
        :raises ValueError: If number of parameters doesn't match configuration
        """
        if len(normalized_params) != self.num_params:
            raise ValueError(
                f"Expected {self.num_params} parameters, " f"got {len(normalized_params)}"
            )

        transformed_params = {}
        for i, name in enumerate(self.param_names):
            transformed_params[name] = self.params[name].transform_param(normalized_params[i])

        return transformed_params

    def back_transform_param_dict(self, param_values: dict[ParamName, Any]) -> list[float]:
        """Transform actual parameter values to normalized values.

        :param param_values: Dictionary mapping parameter names to their values
        :type param_values: dict[ParamName, Any]
        :return: List of normalized parameter values in [0, 1]
        :rtype: list[float]
        :raises ValueError: If parameter names don't match configuration
        """
        if set(param_values.keys()) != set(self.param_names):
            raise ValueError("Parameter names in `param_values` do not match configuration")

        normalized_params = []
        for name in self.param_names:
            normalized_params.append(self.params[name].back_transform_param(param_values[name]))

        return normalized_params
