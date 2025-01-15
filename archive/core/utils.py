"""Utility functions and base classes for the project.

This module provides commonly used utility functions and a base configuration
class to ensure consistency and reusability throughout the codebase. It
includes:

- A base configuration class with Pydantic integration
- Type definitions and constants
- Utility functions for numerical operations

The module focuses on providing robust error handling, type safety, and
consistent interfaces across the project.
"""

from __future__ import annotations

from typing import Any, ClassVar, Final, TypeVar

import numpy as np
from pydantic import BaseModel, ConfigDict

T = TypeVar("T", bound="BaseConfig")

# Constants
MIN_FLOAT_TOLERANCE: Final[float] = 1e-10
"""Minimum tolerance for floating point comparisons."""


class BaseConfig(BaseModel):
    """Base class providing a generic parse method for configuration objects.

    This base class extends Pydantic's BaseModel to provide a consistent
    parsing interface across all configuration objects in the codebase.
    Configuration objects are immutable after creation but will attempt to
    coerce types when possible during instantiation.

    Configuration settings enforce:
    - Objects are frozen (immutable) after creation
    - Extra fields are forbidden
    - Whitespace is stripped from strings
    - Arbitrary types are not allowed
    - Field aliases are supported

    Example:
        >>> class MyConfig(BaseConfig):
        ...     name: str
        ...     count: int
        ...
        >>> config = MyConfig.parse('{"name": "test", "count": "42"}')
        >>> config.count
        42
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
        arbitrary_types_allowed=False,
        populate_by_name=True,
    )

    @classmethod
    def parse(cls: type[T], data: str | bytes | dict[str, Any] | T) -> T:
        """Parse input data into the configuration class.

        This method provides a unified interface for creating configuration
        objects from various input types, including JSON strings, dictionaries,
        and existing BaseConfig instances.

        :param data: Input data to parse. Can be:
            - JSON string
            - bytes containing JSON
            - dictionary
            - instance of this configuration class
        :type data: str | bytes | dict[str, Any] | BaseConfig
        :return: An instance of this configuration class
        :rtype: BaseConfig
        :raises TypeError: If the input data type is not supported
        :raises ValueError: If bytes data cannot be decoded as UTF-8
        :raises pydantic.ValidationError: If the data fails validation

        Example:
            >>> class MyConfig(BaseConfig):
            ...     name: str
            ...     count: int
            ...
            >>> config = MyConfig.parse('{"name": "test", "count": 42}')
            >>> config.name
            'test'
            >>> config = MyConfig.parse({"name": "test", "count": "42"})
            >>> isinstance(config, MyConfig)
            True
        """
        try:
            if isinstance(data, cls):
                return data
            elif isinstance(data, dict):
                return cls.model_validate(data)
            elif isinstance(data, str):
                return cls.model_validate_json(data)
            elif isinstance(data, bytes):
                return cls.model_validate_json(data.decode())
            else:
                raise TypeError(
                    f"Cannot parse data of type {type(data).__name__} into {cls.__name__}. "
                    f"Expected instance of {cls.__name__}, dict, JSON string, or bytes."
                )
        except UnicodeDecodeError as e:
            raise ValueError(f"Failed to decode bytes as UTF-8: {str(e)}") from e


def uniform_to_category(u_sample: float, n_categories: int) -> int:
    """Map a uniform sample to a category index.

    This function takes a uniform random value in [0, 1] and maps it to an
    integer index in the range [0, n_categories-1]. The mapping is uniform,
    meaning each category has an equal probability of being selected if
    u_sample is uniformly distributed.

    :param u_sample: A numeric value sampled from [0, 1]
    :type u_sample: float
    :param n_categories: The total number of categories
    :type n_categories: int
    :return: An integer index in [0, n_categories-1]
    :rtype: int
    :raises TypeError: If inputs are not of the correct type
    :raises ValueError: If n_categories is not positive or u_sample is
        outside [0, 1]

    Example:
        >>> uniform_to_category(0.7, 5)  # 70% -> category 3
        3
        >>> uniform_to_category(0.2, 5)  # 20% -> category 1
        1

    Note:
        The function assigns 1.0 to category n_categories-1. This does not
        affect uniform sampling since sampling 1.0 from a uniform distribution
        on [0, 1] occurs with probability 0.
    """
    # Type validation
    if not isinstance(u_sample, (int, float)):
        raise TypeError(f"u_sample must be a number, got {type(u_sample)}")

    if not isinstance(n_categories, int):
        raise TypeError(f"n_categories must be an integer, got {type(n_categories).__name__}")

    # Value validation
    if n_categories <= 0:
        raise ValueError("n_categories must be positive")

    if u_sample < 0.0 or u_sample > 1.0:
        raise ValueError(f"u_sample must be in the range [0, 1], got {u_sample}")

    # Calculate index
    index = int(np.floor(u_sample * n_categories))
    # Handle edge case where u_sample = 1.0
    return min(index, n_categories - 1)
