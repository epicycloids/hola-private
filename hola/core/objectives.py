from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeAlias

import msgspec
import numpy as np
from msgspec import Struct

from hola.core.utils import FloatArray

ObjectiveName: TypeAlias = str
GroupId: TypeAlias = int


class Direction(str, Enum):

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class ObjectiveConfig(Struct, frozen=True):
    target: float
    limit: float
    direction: Direction = Direction.MINIMIZE
    priority: float = 1.0
    comparison_group: GroupId = 0

    def __post_init__(self):
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
    objectives: dict[ObjectiveName, ObjectiveConfig]
    _group_id_map: dict[GroupId, int] = field(default_factory=dict, init=False)

    @classmethod
    def from_dict(cls, objectives_dict: dict[ObjectiveName, dict[str, Any]]) -> "ObjectiveScorer":
        return cls(
            objectives=msgspec.convert(objectives_dict, dict[ObjectiveName, ObjectiveConfig])
        )

    @classmethod
    def from_json(cls, json_str: str) -> "ObjectiveScorer":
        return cls(
            objectives=msgspec.json.decode(json_str, type=dict[ObjectiveName, ObjectiveConfig])
        )

    def __post_init__(self):
        if not self.objectives:
            raise ValueError("At least one objective must be provided")

        for config in self.objectives.values():
            group_id = config.comparison_group
            if group_id not in self._group_id_map:
                self._group_id_map[group_id] = len(self._group_id_map)

    @property
    def group_ids(self) -> set[GroupId]:
        return set(self._group_id_map.keys())

    @property
    def num_groups(self) -> int:
        return len(self.group_ids)

    @property
    def is_multigroup(self) -> bool:
        return self.num_groups > 1

    def score(self, objectives_dict: dict[ObjectiveName, float]) -> float | FloatArray:
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
