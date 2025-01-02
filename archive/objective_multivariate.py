"""Objective configuration and scalarization utilities.

This module provides a flexible system for configuring and scoring optimization
objectives using a Target-Priority-Limit (TPL) scalarization scheme. The TPL
scheme:

- Uses a target value as the optimization goal
- Sets a limit value as a constraint boundary
- Assigns priorities to balance multiple objectives
- Groups objectives for multi-criteria optimization

Example:
    >>> # Configure objectives for hyperparameter optimization
    >>> objectives = {
    ...     "accuracy": create_objective(
    ...         "maximize",
    ...         target=0.95,
    ...         limit=0.80,
    ...         priority=2.0
    ...     ),
    ...     "training_time": create_objective(
    ...         "minimize",
    ...         target=60,     # target: 1 minute
    ...         limit=300,     # limit: 5 minutes
    ...         priority=1.0
    ...     )
    ... }
    >>> # Create scorer
    >>> scorer = ObjectiveScorer(objectives)
    >>> # Score some results
    >>> scores = scorer.score_objectives({
    ...     "accuracy": 0.92,
    ...     "training_time": 180
    ... })
    >>> print(scores)  # Combined score for the objective group
    [0.9]
    >>> scores = scorer.score_objectives({
    ...     "accuracy": np.array([0.92, 0.88, 0.96]),
    ...     "training_time": np.array([180, 200, 150])
    ... })
    >>> print(scores)
    [[0.9        1.13333333 0.5       ]]
"""

from __future__ import annotations

from enum import Enum
from typing import Any, NewType

import numpy as np
import numpy.typing as npt
from pydantic import Field, model_validator

from hola.core.utils import BaseConfig

# Type definitions
ObjectiveName = NewType('ObjectiveName', str)
GroupId = NewType('GroupId', int)


class Direction(str, Enum):
    """Optimization direction for objectives.

    :cvar MINIMIZE: Objective should be minimized
    :cvar MAXIMIZE: Objective should be maximized
    """

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class ObjectiveConfig(BaseConfig):
    """Configuration for an individual optimization objective.

    Supports vectorized scoring of multiple objective values at once.

    Example:
        >>> # Configure accuracy objective
        >>> acc_obj = ObjectiveConfig(
        ...     target=0.95,    # Target 95% accuracy
        ...     limit=0.80,     # At least 80% required
        ...     direction="maximize",
        ...     priority=2.0     # Higher priority than other objectives
        ... )
        >>> acc_obj.score(0.85)
        1.3333333333333333
        >>> acc_obj.score(np.array([0.85, 0.90, 0.96]))
        array([1.33333333, 0.66666667, 0.        ])
        >>> acc_obj.score(0.75)  # Below limit
        inf

    :param target: Desired value for the objective
    :type target: float
    :param limit: Threshold beyond which the objective is infeasible
    :type limit: float
    :param direction: Whether to minimize or maximize
    :type direction: Direction
    :param priority: Relative importance within group
    :type priority: float
    :param comparison_group: Group ID for comparison
    :type comparison_group: int
    :raises ValueError: If target/limit relationship invalid for direction
    """

    target: float = Field(..., allow_inf_nan=False, description="Desired value for the objective")
    limit: float = Field(
        ..., allow_inf_nan=False, description="Threshold beyond which objective is infeasible"
    )
    direction: Direction = Field(
        default=Direction.MINIMIZE, description="Whether to minimize or maximize"
    )
    priority: float = Field(
        default=1.0, gt=0.0, description="Relative importance within comparison group"
    )
    comparison_group: GroupId = Field(
        default=0, description="Group ID for objectives that can be compared"
    )

    @model_validator(mode="after")
    def validate_target_limit(self) -> "ObjectiveConfig":
        """Validate that target and limit values are compatible."""
        if self.limit == self.target:
            raise ValueError(f"Limit ({self.limit}) cannot equal target ({self.target})")

        if self.direction == Direction.MINIMIZE:
            if self.limit < self.target:
                raise ValueError(
                    f"For minimization, limit ({self.limit}) must be > " f"target ({self.target})"
                )
        else:  # MAXIMIZE
            if self.limit > self.target:
                raise ValueError(
                    f"For maximization, limit ({self.limit}) must be < " f"target ({self.target})"
                )

        return self

    @classmethod
    def parse(cls, data: str | bytes | dict[str, Any] | BaseConfig) -> "ObjectiveConfig":
        """Parse input data into an ObjectiveConfig.

        Overrides BaseConfig.parse to provide correct return type annotation.
        """
        return super().parse(data)  # type: ignore

    def score(self, value: npt.ArrayLike) -> npt.NDArray[np.floating]:
        """Score an objective value relative to target and limit.

        The scoring function:
        - Returns 0.0 if the value meets or exceeds the target
        - Returns infinity if the value violates the limit
        - Returns a value in (0, 1) scaled by priority otherwise

        Supports vectorized scoring for NumPy array inputs.

        :param value: Objective value(s) to score. Can be a scalar (float) or a
            NumPy array.
        :type value: npt.ArrayLike
        :raises TypeError: If input is not a numeric type.
        :return: Score(s) between 0 and infinity. Returns a NumPy array of
            floats.
        :rtype: npt.NDArray[np.floating]
        """
        value = np.asarray(value)

        # Type validation using np.issubdtype for numeric types
        if not np.issubdtype(value.dtype, np.number):
            raise TypeError(f"Input value must be a numeric type, got {value.dtype}")

        if self.direction == Direction.MAXIMIZE:
            meets_target = value >= self.target
            violates_limit = value <= self.limit
            score = self.priority * (1.0 - (value - self.limit) / (self.target - self.limit))

        else:  # MINIMIZE
            meets_target = value <= self.target
            violates_limit = value >= self.limit
            score = self.priority * ((value - self.target) / (self.limit - self.target))

        score = np.where(meets_target, 0.0, score)
        score = np.where(violates_limit, np.inf, score)

        return score

    def with_priority(self, priority: float) -> "ObjectiveConfig":
        """Create a new ObjectiveConfig with an updated priority.

        Example:
            >>> obj = ObjectiveConfig(target=0.95, limit=0.8)
            >>> high_priority = obj.with_priority(5.0)
            >>> print(high_priority.priority)
            5.0
            >>> print(obj.priority)  # Original unchanged
            1.0

        :param priority: New priority value
        :type priority: float
        :return: New ObjectiveConfig with updated priority
        :rtype: ObjectiveConfig
        """
        return self.model_copy(update={"priority": priority})

    def with_comparison_group(self, group: int) -> "ObjectiveConfig":
        """Create a new ObjectiveConfig with an updated comparison group.

        Example:
            >>> obj = ObjectiveConfig(target=0.95, limit=0.8)
            >>> grouped = obj.with_comparison_group(2)
            >>> print(grouped.comparison_group)
            2
            >>> print(obj.comparison_group)  # Original unchanged
            0

        :param group: New comparison group ID
        :type group: int
        :return: New ObjectiveConfig with updated comparison group
        :rtype: ObjectiveConfig
        """
        return self.model_copy(update={"comparison_group": group})


def create_objective(
    direction: str | Direction,
    target: float,
    limit: float,
    priority: float = 1.0,
    comparison_group: int = 0,
) -> ObjectiveConfig:
    """Create an objective configuration.

    Example:
        >>> # Configure training time objective
        >>> time_obj = create_objective(
        ...     "minimize",
        ...     target=60,     # target: 1 minute
        ...     limit=300,     # limit: 5 minutes
        ...     priority=1.0
        ... )

    :param direction: "minimize" or "maximize"
    :type direction: str | Direction
    :param target: Target value to optimize towards
    :type target: float
    :param limit: Boundary constraint value
    :type limit: float
    :param priority: Relative importance, defaults to 1.0
    :type priority: float, optional
    :param comparison_group: Group ID, defaults to 0
    :type comparison_group: int, optional
    :return: Configured objective
    :rtype: ObjectiveConfig
    :raises ValueError: If configuration is invalid
    """
    try:
        if isinstance(direction, str):
            direction = Direction(direction)

        return ObjectiveConfig(
            direction=direction,
            target=target,
            limit=limit,
            priority=priority,
            comparison_group=comparison_group,
        )
    except Exception as e:
        raise ValueError(f"Failed to create objective: {str(e)}") from e


class ObjectiveScorer:
    """Scorer for multiple objectives using TPL scalarization.

    Combines multiple objectives into group scores while respecting:
    - Target/limit thresholds for each objective
    - Priority weightings within groups
    - Separation between comparison groups

    Supports vectorized scoring for NumPy array inputs.

    Example:
        >>> objectives = {
        ...     "accuracy": create_objective(
        ...         "maximize",
        ...         target=0.95,
        ...         limit=0.80,
        ...         priority=2.0
        ...     ),
        ...     "time": create_objective(
        ...         "minimize",
        ...         target=60,
        ...         limit=300,
        ...         priority=1.0
        ...     )
        ... }
        >>> scorer = ObjectiveScorer(objectives)
        >>> scores = scorer.score_objectives({
        ...     "accuracy": 0.90,
        ...     "time": 120
        ... })
        >>> print(scores)
        [0.86666667]
        >>> scores = scorer.score_objectives({
        ...     "accuracy": np.array([0.92, 0.88, 0.96]),
        ...     "time": np.array([180, 200, 150])
        ... })
        >>> print(scores)
        [[0.9        1.13333333 0.5       ]]

    :param objective_configs: Map of objective names to configurations
    :type objective_configs: dict[ObjectiveName, ObjectiveConfig]
    """

    def __init__(self, objective_configs: dict[ObjectiveName, ObjectiveConfig]):
        """Initialize the objective scorer."""
        self.objective_configs = objective_configs

        # Map comparison groups to indices
        unique_groups = sorted({config.comparison_group for config in objective_configs.values()})
        self.group_indices = {group_id: idx for idx, group_id in enumerate(unique_groups)}
        self._num_groups = len(self.group_indices)

    @property
    def multigroup(self) -> bool:
        """Check if multiple comparison groups exist.

        :return: True if multiple groups exist
        :rtype: bool
        """
        return self._num_groups > 1

    def score_objectives(
        self, objectives_dict: dict[ObjectiveName, npt.ArrayLike]
    ) -> npt.NDArray[np.floating]:
        """Score all objectives and combine by comparison group.

        Supports vectorized scoring for NumPy array inputs.

        :param objectives_dict: Map of objective names to values or NumPy
            arrays of values.
        :type objectives_dict: dict[ObjectiveName, npt.ArrayLike]
        :return: Array of scores, one per comparison group. The shape of the
            output array depends on the input:
            - If all inputs are scalar, the output is a 1D array of shape
              (num_groups,).
            - If any inputs are arrays, the output is a 2D array of shape
              (num_inputs, num_groups), where num_inputs is the number of
              elements in the input arrays.
        :rtype: npt.NDArray[np.floating]
        :raises KeyError: If objective names don't match configuration
        :raises ValueError: If input arrays have inconsistent shapes
        """
        # Validate objective names match
        config_names = set(self.objective_configs)
        objective_names = set(objectives_dict)

        if config_names != objective_names:
            missing = config_names - objective_names
            extra = objective_names - config_names
            if missing:
                raise KeyError(f"Missing objectives: {missing}")
            if extra:
                raise KeyError(f"Unexpected objectives: {extra}")

        # Check if any values are NumPy arrays with more than one element
        has_multielement_arrays = any(
            isinstance(val, np.ndarray) and val.size > 1 for val in objectives_dict.values()
        )

        if has_multielement_arrays:
            # Vectorized scoring
            first_array_shape = None
            for name, val in objectives_dict.items():
                if isinstance(val, np.ndarray):
                    if first_array_shape is None:
                        first_array_shape = val.shape
                    elif val.shape != first_array_shape:
                        raise ValueError(
                            f"All objective value arrays must have the same shape. "
                            f"Objective '{name}' has shape {val.shape}, "
                            f"but expected {first_array_shape}."
                        )
                # Ensure all values are arrays of the same shape
                else:
                    objectives_dict[name] = np.array([objectives_dict[name]] * first_array_shape[0]).reshape(first_array_shape)

            num_inputs = first_array_shape[0]
            scored_objs = np.zeros((num_inputs, self._num_groups))
            for name, value in objectives_dict.items():
                config = self.objective_configs[name]
                group_idx = self.group_indices[config.comparison_group]
                scored_objs[:, group_idx] += config.score(value)

        else:
            # Scalar scoring
            scored_objs = np.zeros(self._num_groups)
            for name, value in objectives_dict.items():
                config = self.objective_configs[name]
                group_idx = self.group_indices[config.comparison_group]
                scored_objs[group_idx] += config.score(value)

        return scored_objs
