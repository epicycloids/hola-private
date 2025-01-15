from itertools import groupby, islice
from typing import Generic, Hashable, Iterable, Protocol, TypeVar

import numpy as np
from sortedcontainers import SortedKeyList

from hola.core.utils import MIN_FLOAT_TOLERANCE, FloatArray

Key = TypeVar("Key", bound=Hashable)
DType = TypeVar("DType", bound=np.generic)


class Poset(Protocol[Key]):
    """Protocol defining operations for partially ordered sets."""

    def __len__(self) -> int: ...
    def __getitem__(self, key: Key) -> float | FloatArray: ...
    def add(self, key: Key, value: float | FloatArray) -> None: ...
    def pop(self, key: Key) -> float | FloatArray | None: ...
    def popitem(self) -> tuple[Key, float | FloatArray] | None: ...
    def keys(self) -> Iterable[Key]: ...
    def values(self) -> Iterable[float | FloatArray]: ...
    def items(self) -> Iterable[tuple[Key, float | FloatArray]]: ...
    def fronts(self) -> Iterable[list[tuple[Key, float | FloatArray]]]: ...
    def peek(self, k: int = 1) -> list[tuple[Key, float | FloatArray]]: ...


class ScalarPoset(Generic[Key]):
    """Implementation for totally ordered scalar values."""

    def __init__(self) -> None:
        self._scores: dict[Key, float] = {}
        self.sorted_keys = SortedKeyList(key=self._score_key)

    def __len__(self) -> int:
        return len(self.sorted_keys)

    def __getitem__(self, key: Key) -> float:
        if key not in self._scores:
            raise KeyError(f"Entry {key} does not exist")
        return self._scores[key]

    def _score_key(self, k: Key) -> float:
        return self._scores[k]

    def add(self, key: Key, value: float | FloatArray) -> None:
        if isinstance(value, np.ndarray):
            raise ValueError("Only scalar values permitted.")
        if key in self._scores:
            raise ValueError(f"Entry {key} already exists.")
        self._scores[key] = float(value)
        self.sorted_keys.add(key)

    def pop(self, key: Key) -> float | None:
        if key not in self._scores:
            return None
        self.sorted_keys.remove(key)
        return self._scores.pop(key)

    def popitem(self) -> tuple[Key, float] | None:
        if not self.sorted_keys:
            return None
        key = self.sorted_keys.pop()
        return key, self._scores.pop(key)

    def keys(self) -> Iterable[Key]:
        return self.sorted_keys

    def values(self) -> Iterable[float]:
        return (self[key] for key in self.keys())

    def items(self) -> Iterable[tuple[Key, float]]:
        return ((key, self[key]) for key in self.keys())

    def fronts(self) -> Iterable[list[tuple[Key, float]]]:
        return (list(group) for _, group in groupby(self.items(), key=lambda item: item[1]))

    def peek(self, k: int = 1) -> list[tuple[Key, float]]:
        if k < 1:
            raise ValueError("k must be positive")
        return list(islice(self.items(), k))


class VectorPoset(Generic[Key]):
    """Implementation for partially ordered vector values."""

    def __init__(self) -> None:
        self._scores: dict[Key, FloatArray] = {}
        self._dominance_ranks: dict[Key, int] = {}
        self._crowding_distances: dict[Key, float] = {}
        self._ranks_dirty: bool = True
        self._distances_dirty: bool = True

    def __len__(self) -> int:
        return len(self._scores)

    def __getitem__(self, key: Key) -> FloatArray:
        if key not in self._scores:
            raise KeyError(f"Entry {key} does not exist")
        return self._scores[key]

    def add(self, key: Key, value: float | FloatArray) -> None:
        value = np.asarray(value)
        if key in self._scores:
            raise ValueError(f"Entry {key} already exists.")
        self._scores[key] = value
        self._ranks_dirty = True
        self._distances_dirty = True

    def pop(self, key: Key) -> FloatArray | None:
        if key not in self._scores:
            return None
        value = self._scores.pop(key)
        self._ranks_dirty = True
        self._distances_dirty = True
        return value

    def popitem(self) -> tuple[Key, FloatArray] | None:
        if not self._scores:
            return None

        self._ensure_ranks_updated()
        min_rank = min(self._dominance_ranks.values(), default=None)
        if min_rank is None:
            return None
        key = next(k for k, r in self._dominance_ranks.items() if r == min_rank)
        value = self.pop(key)
        if value is None:
            return None
        return key, value

    def keys(self) -> Iterable[Key]:
        self._ensure_ranks_updated()
        self._ensure_distances_updated()
        return self._scores.keys()

    def values(self) -> Iterable[FloatArray]:
        self._ensure_ranks_updated()
        self._ensure_distances_updated()
        return self._scores.values()

    def items(self) -> Iterable[tuple[Key, FloatArray]]:
        self._ensure_ranks_updated()
        self._ensure_distances_updated()
        return self._scores.items()

    def fronts(self) -> Iterable[list[tuple[Key, FloatArray]]]:
        """Returns items grouped by their non-domination rank."""
        if not self._scores:
            return iter([])

        self._ensure_ranks_updated()
        self._ensure_distances_updated()

        ranked_items = [(k, v) for k, v in self.items()]
        ranked_items.sort(key=lambda x: self._dominance_ranks[x[0]])
        return (
            list(g) for _, g in groupby(ranked_items, key=lambda x: self._dominance_ranks[x[0]])
        )

    def peek(self, k: int = 1) -> list[tuple[Key, FloatArray]]:
        if k < 1:
            raise ValueError("k must be positive")

        sorted_keys = self._get_sorted_keys()
        return [(key, self._scores[key]) for key in sorted_keys[:k]]

    def _ensure_ranks_updated(self, maximal_meaningful_rank: int | None = None) -> None:
        if not self._ranks_dirty:
            return
        self._update_dominance(maximal_meaningful_rank)

    def _ensure_distances_updated(self) -> None:
        """Updates crowding distances if needed."""
        if not self._distances_dirty:
            return

        self._ensure_ranks_updated()
        self._crowding_distances.clear()

        # Calculate crowding distances for each front
        for front in self.fronts():
            front_distances = self._calculate_crowding_distances(front)
            self._crowding_distances.update(front_distances)

        self._distances_dirty = False

    def _update_dominance(self, maximal_meaningful_rank: int | None = None) -> None:
        """Updates the dominance ranks for all points."""
        if not self._scores:
            self._dominance_ranks.clear()
            return

        points = np.array([self._scores[k] for k in self._scores.keys()])
        keys = list(self._scores.keys())
        n = len(points)
        ranks = np.zeros(n, dtype=int)
        lex_order = np.lexsort(points.T)

        for i in range(n):
            curr_idx = lex_order[i]
            curr_point = points[curr_idx]

            # For each point, check against all previously tested points
            dominators = points[lex_order[:i]]
            if len(dominators) > 0:
                dominated_by = np.all(dominators <= curr_point, axis=1) & np.any(
                    dominators < curr_point, axis=1
                )
                if np.any(dominated_by):
                    ranks[curr_idx] = np.max(ranks[lex_order[:i]][dominated_by]) + 1
                else:
                    ranks[curr_idx] = 0
            else:
                ranks[curr_idx] = 0

        if maximal_meaningful_rank is not None:
            ranks[ranks > maximal_meaningful_rank] = maximal_meaningful_rank + 1

        self._dominance_ranks = dict(zip(keys, ranks))
        self._ranks_dirty = False

    def _calculate_crowding_distances(
        self, front_items: list[tuple[Key, FloatArray]]
    ) -> dict[Key, float]:
        if not front_items:
            return {}

        n_points = len(front_items)
        if n_points <= 2:
            # For 1 or 2 points, assign maximum crowding distance
            return {key: float("inf") for key, _ in front_items}

        keys = [item[0] for item in front_items]
        points = np.array([item[1] for item in front_items])

        n_objectives = points.shape[1]
        crowding_distances = np.zeros(n_points)

        # Calculate crowding distance for each objective
        for obj_idx in range(n_objectives):
            sorted_indices = np.argsort(points[:, obj_idx])
            sorted_points = points[sorted_indices]
            obj_range = sorted_points[-1, obj_idx] - sorted_points[0, obj_idx]

            # Set infinite distance for boundary points
            crowding_distances[sorted_indices[0]] = float("inf")
            crowding_distances[sorted_indices[-1]] = float("inf")

            if obj_range > MIN_FLOAT_TOLERANCE:  # Only calculate if points differ in this objective
                for i in range(1, n_points - 1):
                    idx = sorted_indices[i]
                    prev_val = sorted_points[i - 1, obj_idx]
                    next_val = sorted_points[i + 1, obj_idx]
                    crowding_distances[idx] += (next_val - prev_val) / obj_range

        return dict(zip(keys, crowding_distances))

    def _get_sorted_keys(self) -> list[Key]:
        # Note: Caller must ensure ranks and distances are updated
        # Sort by rank first, then by crowding distance (descending)
        return sorted(
            self._scores.keys(),
            key=lambda k: (self._dominance_ranks[k], -self._crowding_distances[k]),
        )
