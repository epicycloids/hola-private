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
    >>> print(scores)
    [0.9]
"""

from __future__ import annotations

from enum import Enum
from typing import NewType, TypeVar

from pydantic import (
    Field,
    FiniteFloat,
    PositiveFloat,
    computed_field,
    field_validator,
    model_validator,
)

from hola.core.utils import BaseConfig

# Type definitions
OC = TypeVar("OC", bound="ObjectiveConfig")
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
        >>> acc_obj.score(0.96)
        0.0
        >>> # Score in feasible range
        >>> acc_obj.score(0.85)
        1.333333333333334
        >>> # Score violates limit
        >>> acc_obj.score(0.75)
        inf
    """

    target: FiniteFloat = Field(..., description="Desired value for the objective")
    limit: FiniteFloat = Field(..., description="Threshold beyond which objective is infeasible")
    direction: Direction = Field(
        default=Direction.MINIMIZE, description="Whether to minimize or maximize"
    )
    priority: PositiveFloat = Field(
        default=1.0, description="Relative importance within comparison group"
    )
    comparison_group: GroupId = Field(
        default=GroupId(0), description="Group ID for objectives that can be compared"
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


class ObjectiveScorer(BaseConfig):
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
        >>> print(scores)
        [0.9]
    """

    objectives: dict[ObjectiveName, ObjectiveConfig]

    @field_validator("objectives")
    @classmethod
    def validate_objectives_not_empty(
        cls, v: dict[ObjectiveName, ObjectiveConfig]
    ) -> dict[ObjectiveName, ObjectiveConfig]:
        """Ensure at least one objective is provided."""
        if not v:
            raise ValueError("At least one objective must be provided")
        return v

    @computed_field  # type: ignore[prop-decorator]
    @property
    def group_ids(self) -> set[GroupId]:
        """Unique group IDs."""
        return {config.comparison_group for config in self.objectives.values()}

    @computed_field  # type: ignore[prop-decorator]
    @property
    def num_groups(self) -> int:
        """Number of comparison groups."""
        return len(self.group_ids)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def multigroup(self) -> bool:
        """Check if multiple comparison groups exist."""
        return self.num_groups > 1

    def score_objectives(self, objectives_dict: dict[ObjectiveName, float]) -> dict[GroupId, float]:
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
            >>> np.any(np.isinf(scores))
            np.False_
        """
        # Validate objective names match
        config_names = set(self.objectives)
        objective_names = set(objectives_dict)

        if config_names != objective_names:
            missing = config_names - objective_names
            extra = objective_names - config_names
            if missing:
                raise KeyError(f"Missing objectives: {missing}")
            if extra:
                raise KeyError(f"Unexpected objectives: {extra}")

        # Initialize scores for each group
        scored_objs = {group_id: 0.0 for group_id in self.group_ids}

        # Score each objective and add to appropriate group
        for name, value in objectives_dict.items():
            config = self.objectives[name]
            scored_objs[config.comparison_group] += config.score(value)

        return scored_objs
