"""Objective configuration and scalarization utilities.

This module provides a flexible system for configuring and scoring optimization
objectives using a Target-Priority-Limit (TPL) scalarization scheme. Key features:

- Target values specify desired optimization goals
- Limit values enforce constraint boundaries
- Priorities balance multiple objectives
- Groups support multi-criteria optimization
- Scaling ensures objectives are commensurable

The TPL scheme allows fine-grained control over optimization behavior while
maintaining clear semantics and numerical stability.

Example:
    >>> # Configure objectives for hyperparameter optimization
    >>> objectives = {
    ...     "accuracy": create_objective(
    ...         "maximize",
    ...         target=0.95,  # Target 95% accuracy
    ...         limit=0.80,   # Require at least 80%
    ...         priority=2.0  # Higher priority than training time
    ...     ),
    ...     "training_time": create_objective(
    ...         "minimize",
    ...         target=60,    # Target: 1 minute
    ...         limit=300,    # Limit: 5 minutes
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
"""

from __future__ import annotations

from enum import Enum
from typing import NewType

import numpy as np
import numpy.typing as npt
from pydantic import Field, model_validator

from hola.core.utils import BaseConfig

# Type definitions
ObjectiveName = NewType("ObjectiveName", str)
GroupId = NewType("GroupId", int)


class Direction(str, Enum):
    """Optimization direction for objectives.

    :cvar MINIMIZE: Objective should be minimized, lower values are better
    :cvar MAXIMIZE: Objective should be maximized, higher values are better
    """

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class ObjectiveConfig(BaseConfig):
    """Configuration for an individual optimization objective.

    Defines a single objective with target value, limit value, optimization
    direction, and priority level. The configuration supports both minimization
    and maximization objectives, with targets representing ideal values and
    limits representing constraint boundaries.

    The objective's score is computed as:
    - 0.0 if target is reached or exceeded
    - infinity if limit is violated
    - A value in (0, infinity) scaled by priority otherwise

    Example:
        >>> # Configure accuracy objective
        >>> acc_obj = ObjectiveConfig(
        ...     target=0.95,    # Target 95% accuracy
        ...     limit=0.80,     # At least 80% required
        ...     direction="maximize",
        ...     priority=2.0     # Higher priority than other objectives
        ... )
        >>> # Score meets target
        >>> acc_obj.score(0.96)  # Returns 0.0
        >>> # Score in feasible range
        >>> acc_obj.score(0.85)  # Returns ~1.33
        >>> # Score violates limit
        >>> acc_obj.score(0.75)  # Returns infinity
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
        """Validate that target and limit values are compatible.

        For minimization, limit must be greater than target.
        For maximization, limit must be less than target.

        :return: Validated configuration
        :rtype: ObjectiveConfig
        :raises ValueError: If target and limit values are incompatible
            with optimization direction
        """
        if self.limit == self.target:
            raise ValueError(f"Limit ({self.limit}) cannot equal target ({self.target})")

        if self.direction == Direction.MINIMIZE:
            if self.limit < self.target:
                raise ValueError(
                    f"For minimization, limit ({self.limit}) must be > target ({self.target})"
                )
        else:  # MAXIMIZE
            if self.limit > self.target:
                raise ValueError(
                    f"For maximization, limit ({self.limit}) must be < target ({self.target})"
                )

        return self

    def score(self, value: float) -> float:
        """Score an objective value relative to target and limit.

        The scoring function:
        - Returns 0.0 if the value meets or exceeds the target
        - Returns infinity if the value violates the limit
        - Returns a value in (0, infinity) scaled by priority otherwise

        For minimization objectives:
        - score = 0.0 if value <= target
        - score = infinity if value >= limit
        - score = priority * (value - target)/(limit - target) otherwise

        For maximization objectives:
        - score = 0.0 if value >= target
        - score = infinity if value <= limit
        - score = priority * (1 - (value - limit)/(target - limit)) otherwise

        :param value: Objective value to score
        :type value: float
        :return: Score between 0 and infinity
        :rtype: float
        :raises TypeError: If input is not a numeric type
        """
        if not isinstance(value, (int, float)):
            raise TypeError(f"Input value must be a number, got {type(value)}")

        if self.direction == Direction.MAXIMIZE:
            if value >= self.target:
                return 0.0
            if value <= self.limit:
                return float("inf")
            return self.priority * (1.0 - (value - self.limit) / (self.target - self.limit))
        else:  # MINIMIZE
            if value <= self.target:
                return 0.0
            if value >= self.limit:
                return float("inf")
            return self.priority * ((value - self.target) / (self.limit - self.target))

    def with_priority(self, priority: float) -> "ObjectiveConfig":
        """Create a new ObjectiveConfig with an updated priority.

        :param priority: New priority value
        :type priority: float
        :return: New ObjectiveConfig with updated priority
        :rtype: ObjectiveConfig
        :raises ValueError: If priority is not positive
        """
        return self.model_copy(update={"priority": priority})

    def with_comparison_group(self, group: int) -> "ObjectiveConfig":
        """Create a new ObjectiveConfig with an updated comparison group.

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

    Convenience function for creating properly configured objectives with
    validation of all parameters.

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

    Example:
        >>> # Create accuracy objective
        >>> acc_obj = create_objective(
        ...     "maximize",
        ...     target=0.95,
        ...     limit=0.80,
        ...     priority=2.0
        ... )
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

    The scorer maintains proper scaling and weighting of objectives while
    ensuring that constraint violations (limit violations) always dominate
    non-constraint factors.

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
        >>> print(scores)  # [0.86666667]
    """

    def __init__(self, objective_configs: dict[ObjectiveName, ObjectiveConfig]):
        """Initialize the objective scorer.

        :param objective_configs: Dictionary mapping objective names to their
            configurations
        :type objective_configs: dict[ObjectiveName, ObjectiveConfig]
        """
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

    def score_objectives(self, objectives_dict: dict[ObjectiveName, float]) -> npt.NDArray[np.float64]:
        """Score all objectives and combine by comparison group.

        Scores each objective individually and combines scores within comparison
        groups. Scores are combined by summing within each group, with infinity
        propagating through sums to ensure constraint violations are respected.

        :param objectives_dict: Map of objective names to values
        :type objectives_dict: dict[ObjectiveName, float]
        :return: Array of scores, one per comparison group
        :rtype: npt.NDArray[np.float64]
        :raises KeyError: If objective names don't match configuration

        Example:
            >>> scores = scorer.score_objectives({
            ...     "accuracy": 0.92,
            ...     "time": 180
            ... })
            >>> # Check if any group violates constraints
            >>> np.any(np.isinf(scores))  # Returns False
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

        # Initialize scores for each group
        scored_objs = np.zeros(self._num_groups)

        # Score each objective and add to appropriate group
        for name, value in objectives_dict.items():
            config = self.objective_configs[name]
            group_idx = self.group_indices[config.comparison_group]
            scored_objs[group_idx] += config.score(value)

        return scored_objs