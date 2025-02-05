"""
Objective configuration and scoring system for HOLA.

This module implements the multi-objective scoring system used in HOLA to evaluate
trial parameters. The scoring system has these key features:

1. Each objective has a target value (where the user is satisfied) and a limit
   value (beyond which the solution is rejected)
2. Objectives can be configured for either maximization or minimization
3. Objectives are grouped into comparison groups, with scores combined within
   each group using priority-weighted sums
4. Scores are normalized to [0, inf) where:
   - 0 means the objective meets or exceeds its target
   - 1 means the objective is halfway between target and limit
   - inf means the objective exceeds its limit

Multiple comparison groups allow for handling incomparable objectives, where
the optimizer must consider each group's score independently.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeAlias

import msgspec
import numpy as np
from msgspec import Struct

from hola.core.utils import FloatArray

ObjectiveName: TypeAlias = str
"""Type alias for objective names in configuration dictionaries."""

GroupId: TypeAlias = int
"""Type alias for comparison group identifiers."""


class Direction(str, Enum):
    """
    Optimization direction for objectives.

    :cvar MAXIMIZE: Objective value should be increased
    :cvar MINIMIZE: Objective value should be decreased
    """

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class ObjectiveConfig(Struct, frozen=True):
    """
    Configuration for a single objective in the optimization problem.

    Each objective is defined by target and limit values, with scores
    computed based on where the actual value falls between these bounds.
    """

    target: float
    """Value at or beyond which the objective is considered satisfied."""

    limit: float
    """Value beyond which the objective is considered infeasible."""

    direction: Direction = Direction.MINIMIZE
    """Whether to maximize or minimize the objective."""

    priority: float = 1.0
    """Weight given to this objective within its comparison group."""

    comparison_group: GroupId = 0
    """Group ID for combining comparable objectives."""

    def __post_init__(self):
        """
        Validate objective configuration.

        :raises ValueError: If:
            - target equals limit
            - limit and target are incorrectly ordered for the direction
            - priority is not positive
            - comparison group is negative
        """
        if self.limit == self.target:
            raise ValueError(f"Limit ({self.limit}) cannot equal target ({self.target})")

        if self.direction == Direction.MINIMIZE:
            if self.limit < self.target:
                raise ValueError(
                    f"For minimization, limit ({self.limit}) must be > target ({self.target})"
                )
        else:  # maximize
            if self.limit > self.target:
                raise ValueError(
                    f"For maximization, limit ({self.limit}) must be < target ({self.target})"
                )

        if self.priority <= 0:
            raise ValueError("Priority should be positive")

        if self.comparison_group < 0:
            raise ValueError("Comparison group ID must be a positive integer")

    def score(self, value: float) -> float:
        """
        Calculate the normalized score for an objective value.

        For minimization:
        - Returns 0 if value <= target
        - Returns inf if value >= limit
        - Otherwise returns priority * (value - target)/(limit - target)

        For maximization:
        - Returns 0 if value >= target
        - Returns inf if value <= limit
        - Otherwise returns priority * (1 - (value - limit)/(target - limit))

        :param value: Objective value to score
        :type value: float
        :return: Normalized score in [0, inf]
        :rtype: float
        """
        if self.direction == Direction.MAXIMIZE:
            if value >= self.target:
                return 0.0
            if value <= self.limit:
                return float("inf")
            return self.priority * (1.0 - (value - self.limit) / (self.target - self.limit))
        else:  # minimize
            if value <= self.target:
                return 0.0
            if value >= self.limit:
                return float("inf")
            return self.priority * ((value - self.target) / (self.limit - self.target))


@dataclass
class ObjectiveScorer:
    """
    Scores multiple objectives and combines them into comparison groups.

    This class handles the scoring of multiple objectives according to their
    configurations and combines scores within comparison groups using
    priority-weighted sums.
    """

    objectives: dict[ObjectiveName, ObjectiveConfig]
    """Dictionary mapping objective names to their configurations."""

    _group_id_map: dict[GroupId, int] = field(default_factory=dict, init=False)
    """Internal mapping from group IDs to contiguous indices."""

    @classmethod
    def from_dict(cls, objectives_dict: dict[ObjectiveName, dict[str, Any]]) -> "ObjectiveScorer":
        """
        Create scorer from a dictionary of objective specifications.

        :param objectives_dict: Dictionary mapping names to objective configurations
        :type objectives_dict: dict[ObjectiveName, dict[str, Any]]
        :return: Configured objective scorer
        :rtype: ObjectiveScorer
        """
        return cls(
            objectives=msgspec.convert(objectives_dict, dict[ObjectiveName, ObjectiveConfig])
        )

    @classmethod
    def from_json(cls, json_str: str) -> "ObjectiveScorer":
        """
        Create scorer from a JSON string of objective specifications.

        :param json_str: JSON string containing objective configurations
        :type json_str: str
        :return: Configured objective scorer
        :rtype: ObjectiveScorer
        """
        return cls(
            objectives=msgspec.json.decode(json_str, type=dict[ObjectiveName, ObjectiveConfig])
        )

    def __post_init__(self):
        """
        Initialize group ID mapping and validate configuration.

        :raises ValueError: If no objectives are provided
        """
        if not self.objectives:
            raise ValueError("At least one objective must be provided")

        for config in self.objectives.values():
            group_id = config.comparison_group
            if group_id not in self._group_id_map:
                self._group_id_map[group_id] = len(self._group_id_map)

    @property
    def group_ids(self) -> set[GroupId]:
        """
        :return: Set of all comparison group IDs
        :rtype: set[GroupId]
        """
        return set(self._group_id_map.keys())

    @property
    def num_groups(self) -> int:
        """
        :return: Number of comparison groups
        :rtype: int
        """
        return len(self.group_ids)

    @property
    def is_multigroup(self) -> bool:
        """
        :return: True if there are multiple comparison groups
        :rtype: bool
        """
        return self.num_groups > 1

    def score(self, objectives_dict: dict[ObjectiveName, float]) -> float | FloatArray:
        """
        Score a set of objective values.

        For each comparison group, combines the priority-weighted scores of its
        objectives using addition. Returns either a single score (if there is
        only one comparison group) or an array of scores (if there are multiple
        groups).

        :param objectives_dict: Dictionary mapping objective names to values
        :type objectives_dict: dict[ObjectiveName, float]
        :return: Combined score(s) for comparison group(s)
        :rtype: float | FloatArray
        :raises KeyError: If objective names don't match configuration
        """
        config_names = set(self.objectives)
        objective_names = set(objectives_dict)

        if config_names != objective_names:
            missing = config_names - objective_names
            extra = objective_names - config_names
            if missing:
                raise KeyError(f"Missing objectives: {missing}")
            if extra:
                raise KeyError(f"Unexpected objectives: {extra}")

        scored_objs = np.zeros(self.num_groups)

        for name, value in objectives_dict.items():
            config = self.objectives[name]
            group_index = self._group_id_map[config.comparison_group]
            scored_objs[group_index] += config.score(value)

        if self.is_multigroup:
            return scored_objs
        return scored_objs.item()
