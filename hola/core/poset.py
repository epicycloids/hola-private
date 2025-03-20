"""
Partially ordered set (poset) implementations for maintaining trial
leaderboards.

This module provides data structures for maintaining sorted collections of
trials in hyperparameter optimization. It supports both scalar-valued trials
(sorted by a single objective) and vector-valued trials (sorted by multiple
objectives using non-dominated sorting and crowding distance).

The posets maintain their elements in sorted order and provide efficient access
to the best-performing trials. The VectorPoset implements the NSGA-II
algorithm's sorting mechanism for multi-objective optimization.
"""

from itertools import groupby, islice
from typing import Generic, Hashable, Iterable, Protocol, TypeVar

import numpy as np
from sortedcontainers import SortedKeyList

from hola.core.utils import MIN_FLOAT_TOLERANCE, FloatArray

Key = TypeVar("Key", bound=Hashable)
"""Type variable for hashable keys used to identify trials in posets."""


class Poset(Protocol[Key]):
    """
    Protocol defining the interface for partially ordered collections of trials.

    This protocol specifies the methods that both scalar and vector posets must
    implement to maintain sorted collections of trials.
    """

    def __len__(self) -> int:
        """
        :return: Number of trials in the poset
        :rtype: int
        """
        ...

    def __getitem__(self, key: Key) -> float | FloatArray:
        """
        Retrieve the score(s) associated with a trial.

        :param key: Trial identifier
        :type key: Key
        :return: Score(s) for the trial
        :rtype: float | FloatArray
        :raises KeyError: If key doesn't exist in the poset
        """
        ...

    def add(self, key: Key, value: float | FloatArray) -> None:
        """
        Add a new trial to the poset.

        :param key: Trial identifier
        :type key: Key
        :param value: Score(s) for the trial
        :type value: float | FloatArray
        :raises ValueError: If key already exists or value has invalid format
        """
        ...

    def pop(self, key: Key) -> float | FloatArray | None:
        """
        Remove and return a trial's score(s).

        :param key: Trial identifier
        :type key: Key
        :return: Score(s) for the trial, or None if key doesn't exist
        :rtype: float | FloatArray | None
        """
        ...

    def poptopitem(self) -> tuple[Key, float | FloatArray] | None:
        """
        Remove and return the best-performing trial.

        :return: Tuple of (key, score(s)) for the best trial, or None if empty
        :rtype: tuple[Key, float | FloatArray] | None
        """
        ...

    def keys(self) -> Iterable[Key]:
        """
        :return: Iterator over trial identifiers in sorted order
        :rtype: Iterable[Key]
        """
        ...

    def values(self) -> Iterable[float | FloatArray]:
        """
        :return: Iterator over trial scores in sorted order
        :rtype: Iterable[float | FloatArray]
        """
        ...

    def items(self) -> Iterable[tuple[Key, float | FloatArray]]:
        """
        :return: Iterator over (key, score(s)) pairs in sorted order
        :rtype: Iterable[tuple[Key, float | FloatArray]]
        """
        ...

    def fronts(self) -> Iterable[list[tuple[Key, float | FloatArray]]]:
        """
        :return: Iterator over Pareto fronts, each containing (key, score(s)) pairs
        :rtype: Iterable[list[tuple[Key, float | FloatArray]]]
        """
        ...

    def peek(self, k: int = 1) -> list[tuple[Key, float | FloatArray]]:
        """
        View the top k trials without removing them.

        :param k: Number of trials to peek at
        :type k: int
        :return: List of top k (key, score(s)) pairs
        :rtype: list[tuple[Key, float | FloatArray]]
        :raises ValueError: If k < 1
        """
        ...


class ScalarPoset(Generic[Key]):
    """
    Poset implementation for scalar-valued trials.

    Maintains trials sorted by a single objective value in ascending order.
    Implemented using a sorted key list for efficient insertion and retrieval.
    """

    def __init__(self) -> None:
        """Initialize an empty scalar poset."""
        self._scores: dict[Key, float] = {}
        self.sorted_keys = SortedKeyList(key=self._score_key)

    def __len__(self) -> int:
        """
        :return: Number of trials in the poset
        :rtype: int
        """
        return len(self.sorted_keys)

    def __getitem__(self, key: Key) -> float:
        """
        Retrieve the score associated with a trial.

        :param key: Trial identifier
        :type key: Key
        :return: Score for the trial
        :rtype: float
        :raises KeyError: If key doesn't exist in the poset
        """
        if key not in self._scores:
            raise KeyError(f"Entry {key} does not exist.")
        return self._scores[key]

    def add(self, key: Key, value: float | FloatArray) -> None:
        """
        Add a new trial to the poset.

        :param key: Trial identifier
        :type key: Key
        :param value: Score for the trial
        :type value: float | FloatArray
        :raises ValueError: If value is not a scalar float or if key already exists
        """
        if not isinstance(value, float):
            raise ValueError("Only scalar float values are permitted.")
        if key in self._scores:
            raise ValueError(f"Entry {key} already exists.")
        self._scores[key] = float(value)
        self.sorted_keys.add(key)

    def pop(self, key: Key) -> float | None:
        """
        Remove and return a trial's score.

        :param key: Trial identifier
        :type key: Key
        :return: Score for the trial, or None if key doesn't exist
        :rtype: float | None
        """
        if key not in self._scores:
            return None
        self.sorted_keys.remove(key)
        return self._scores.pop(key)

    def poptopitem(self) -> tuple[Key, float] | None:
        """
        Remove and return the trial with lowest score.

        :return: Tuple of (key, score) for the best trial, or None if empty
        :rtype: tuple[Key, float] | None
        """
        if not self.sorted_keys:
            return None
        key = self.sorted_keys.pop(index=0)
        return key, self._scores.pop(key)

    def keys(self) -> Iterable[Key]:
        """
        :return: Iterator over trial identifiers in ascending score order
        :rtype: Iterable[Key]
        """
        return self.sorted_keys

    def values(self) -> Iterable[float]:
        """
        :return: Iterator over trial scores in ascending order
        :rtype: Iterable[float]
        """
        return (self[key] for key in self.keys())

    def items(self) -> Iterable[tuple[Key, float]]:
        """
        :return: Iterator over (key, score) pairs in ascending score order
        :rtype: Iterable[tuple[Key, float]]
        """
        return ((key, self[key]) for key in self.keys())

    def fronts(self) -> Iterable[list[tuple[Key, float]]]:
        """
        Group trials by score value.

        For scalar posets, each front contains trials with exactly equal scores.

        :return: Iterator over groups of (key, score) pairs with equal scores
        :rtype: Iterable[list[tuple[Key, float]]]
        """
        return (list(group) for _, group in groupby(self.items(), key=lambda item: item[1]))

    def peek(self, k: int = 1) -> list[tuple[Key, float]]:
        """
        View the k trials with lowest scores without removing them.

        :param k: Number of trials to peek at
        :type k: int
        :return: List of k (key, score) pairs with lowest scores
        :rtype: list[tuple[Key, float]]
        :raises ValueError: If k < 1
        """
        if k < 1:
            raise ValueError("k must be positive.")
        return list(islice(self.items(), k))

    def get_crowding_distance(self, key: Key) -> float:
        """
        Get the crowding distance for a trial.

        For scalar posets, all crowding distances are infinity since
        each trial forms its own front.

        :param key: Trial identifier
        :type key: Key
        :return: Crowding distance (always infinity for scalar posets)
        :rtype: float
        :raises KeyError: If key doesn't exist in the poset
        """
        if key not in self._scores:
            raise KeyError(f"Entry {key} does not exist.")
        return float("inf")  # All elements are by themselves; infinite crowding distance.

    def get_indices(self) -> set[Key]:
        """
        Get the set of all keys in the poset.

        :return: Set of all trial identifiers in the poset
        :rtype: set[Key]
        """
        return set(self.sorted_keys)

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------
    def _score_key(self, k: Key) -> float:
        return self._scores[k]


class VectorPoset(Generic[Key]):
    """
    Poset implementation for vector-valued trials using non-dominated sorting.

    Maintains trials in Pareto fronts using the NSGA-II algorithm's sorting
    mechanism. Within each front, trials are sorted by crowding distance to
    maintain diversity. The first front contains non-dominated solutions, with
    subsequent fronts containing solutions dominated by increasing numbers of
    other solutions.
    """

    def __init__(self):
        """Initialize an empty vector poset."""
        self._vectors: dict[Key, FloatArray] = {}
        self._fronts: list[list[Key]] = []
        self._crowding_distances: dict[Key, float] = {}

    def __len__(self) -> int:
        """
        :return: Number of trials in the poset
        :rtype: int
        """
        return len(self._vectors)

    def __getitem__(self, key: Key) -> FloatArray:
        """
        Retrieve the score vector associated with a trial.

        :param key: Trial identifier
        :type key: Key
        :return: Score vector for the trial
        :rtype: FloatArray
        :raises KeyError: If key doesn't exist in the poset
        """
        if key not in self._vectors:
            raise KeyError(f"Entry {key} does not exist.")
        return self._vectors[key]

    def add(self, key: Key, value: float | FloatArray) -> None:
        """
        Add a new trial to the poset and update Pareto fronts.

        :param key: Trial identifier
        :type key: Key
        :param value: Score vector for the trial
        :type value: float | FloatArray
        :raises ValueError: If key already exists, or if value is not a 1D array
        """
        if key in self._vectors:
            raise ValueError(f"Key {key} already exists in this poset.")
        arr = np.atleast_1d(value)
        if arr.ndim != 1:
            raise ValueError("VectorPoset supports only 1D vectors.")
        self._vectors[key] = arr
        # Insert into the first feasible front
        self._insert_into_front(0, key)

    def pop(self, key: Key) -> FloatArray | None:
        """
        Remove and return a trial's score vector, updating Pareto fronts.

        :param key: Trial identifier
        :type key: Key
        :return: Score vector for the trial, or None if key doesn't exist
        :rtype: FloatArray | None
        """
        if key not in self._vectors:
            return None
        front_index = self._find_front_index_of_key(key)
        if front_index is None:
            return None
        val = self._vectors.pop(key)
        front = self._fronts[front_index]
        front.remove(key)

        # Remove empty fronts; otherwise recalc crowding distances
        if not front:
            self._fronts.pop(front_index)
        else:
            self._compute_crowding_distance(front)
            self._pull_back_from_front(front_index)
        return val

    def poptopitem(self) -> tuple[Key, FloatArray] | None:
        """
        Remove and return a trial from the first Pareto front.

        Returns the first trial in the best front, which will have maximum
        crowding distance among non-dominated solutions.

        :return: Tuple of (key, score_vector) for the selected trial, or None if empty
        :rtype: tuple[Key, FloatArray] | None
        """
        if not self._vectors:
            return None
        best_front = self._fronts[0]
        key = best_front[0]
        val = self.pop(key)
        return (key, val) if val is not None else None

    def keys(self) -> Iterable[Key]:
        """
        :return: Iterator over trial identifiers in Pareto front order
        :rtype: Iterable[Key]
        """
        for k, _ in self.items():
            yield k

    def values(self) -> Iterable[FloatArray]:
        """
        :return: Iterator over trial score vectors in Pareto front order
        :rtype: Iterable[FloatArray]
        """
        for _, v in self.items():
            yield v

    def items(self) -> Iterable[tuple[Key, FloatArray]]:
        """
        :return: Iterator over (key, score_vector) pairs in Pareto front order
        :rtype: Iterable[tuple[Key, FloatArray]]
        """
        for front in self._fronts:
            for k in front:
                yield (k, self._vectors[k])

    def fronts(self) -> Iterable[list[tuple[Key, FloatArray]]]:
        """
        :return: Iterator over Pareto fronts, each containing (key,
            score_vector) pairs sorted by crowding distance
        :rtype: Iterable[list[tuple[Key, FloatArray]]]
        """
        for front in self._fronts:
            yield [(k, self._vectors[k]) for k in front]

    def peek(self, k: int = 1) -> list[tuple[Key, FloatArray]]:
        """
        View k trials in Pareto front order without removing them.

        Returns trials in order of front membership, with ties broken by
        crowding distance.

        :param k: Number of trials to peek at
        :type k: int
        :return: List of k (key, score_vector) pairs
        :rtype: list[tuple[Key, FloatArray]]
        :raises ValueError: If k < 1
        """
        if k < 1:
            raise ValueError("k must be positive.")
        out = []
        for fr in self._fronts:
            for key in fr:
                out.append((key, self._vectors[key]))
                if len(out) == k:
                    return out
        return out

    def get_crowding_distance(self, key: Key) -> float:
        """
        Get the crowding distance for a trial.

        :param key: Trial identifier
        :type key: Key
        :return: Crowding distance for the trial
        :rtype: float
        :raises KeyError: If key doesn't exist in the poset
        """
        if key not in self._vectors:
            raise KeyError(f"Entry {key} does not exist.")
        return self._crowding_distances.get(key, float("inf"))

    def get_indices(self) -> set[Key]:
        """
        Get the set of all keys in the poset.

        :return: Set of all trial identifiers in the poset
        :rtype: set[Key]
        """
        return set(self._vectors.keys())

    def items_with_distances(self) -> Iterable[tuple[Key, FloatArray, float]]:
        """
        :return: Iterator over (key, scores, crowding_distance) triples
        :rtype: Iterable[tuple[Key, FloatArray, float]]
        """
        for front in self._fronts:
            for k in front:
                yield k, self._vectors[k], self._crowding_distances.get(k, 0.0)

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------
    def _insert_into_front(self, front_index: int, key: Key) -> None:
        """
        Insert a trial into the appropriate Pareto front.

        Uses recursive insertion to maintain proper front ordering. A trial is
        inserted into the first front where it is not dominated by any existing
        trial.

        :param front_index: Index of the front to attempt insertion into
        :type front_index: int
        :param key: Trial identifier to insert
        :type key: Key
        """
        # If we have no more fronts to check, create a new one.
        if front_index == len(self._fronts):
            self._fronts.append([key])
            return

        current_front = self._fronts[front_index]

        # If any element in the current front dominates 'key', insert into the next front.
        for existing_key in current_front:
            if dominates(self._vectors[existing_key], self._vectors[key]):
                self._insert_into_front(front_index + 1, key)
                return

        # Otherwise, 'key' might dominate some current_front elements.
        dominated_here = []
        for existing_key in current_front:
            if dominates(self._vectors[key], self._vectors[existing_key]):
                dominated_here.append(existing_key)

        # Remove dominated elements from this front.
        for dk in dominated_here:
            current_front.remove(dk)

        # Add 'key' to the current front.
        current_front.append(key)
        self._compute_crowding_distance(current_front)

        # Move any dominated elements to the next front.
        for dk in dominated_here:
            self._insert_into_front(front_index + 1, dk)

    def _pull_back_from_front(self, front_index: int) -> None:
        """
        Check if any trials from later fronts can move to earlier fronts.

        Called after removing trials to maintain proper front ordering.

        :param front_index: Index of the front to check for pull-backs
        :type front_index: int
        """
        if front_index >= len(self._fronts) - 1:
            return
        current_front = self._fronts[front_index]
        next_front = self._fronts[front_index + 1]

        to_pull = []
        for key_next in next_front:
            # If 'key_next' is not dominated by anything in 'current_front,' move it up.
            if not any(
                dominates(self._vectors[kc], self._vectors[key_next]) for kc in current_front
            ):
                to_pull.append(key_next)

        if not to_pull:
            return

        # Remove them from the next front, insert them into current_front
        for pull_key in to_pull:
            next_front.remove(pull_key)
            self._insert_into_front(front_index, pull_key)

        # If the next front is now empty, remove it.
        if not next_front:
            self._fronts.pop(front_index + 1)
        else:
            self._compute_crowding_distance(next_front)

        # Recursively check deeper fronts.
        self._pull_back_from_front(front_index + 1)

    def _compute_crowding_distance(self, front: list[Key]) -> None:
        """
        Compute crowding distances for trials in a Pareto front.

        Implements the NSGA-II crowding distance calculation to maintain
        diversity within fronts. Boundary points receive infinite distance.

        :param front: List of trial identifiers in a single front
        :type front: list[Key]
        """
        if len(front) == 0:
            return

        # If up to 2 elements, everyone gets infinite distance (boundaries).
        if len(front) <= 2:
            for k in front:
                self._crowding_distances[k] = float("inf")
            return

        distances = {k: 0.0 for k in front}
        d = len(self._vectors[front[0]])
        for dim in range(d):
            # Sort the front by the current dimension
            front.sort(key=lambda k: self._vectors[k][dim])
            min_val = float(self._vectors[front[0]][dim])
            max_val = float(self._vectors[front[-1]][dim])
            denom = max_val - min_val

            if abs(denom) < MIN_FLOAT_TOLERANCE:
                # Degenerate dimension, skip.
                continue

            # Boundary points get infinite distance in this dimension.
            distances[front[0]] = float("inf")
            distances[front[-1]] = float("inf")

            # For interior points, accumulate normalized distance.
            for i in range(1, len(front) - 1):
                if distances[front[i]] != float("inf"):
                    prev_k = front[i - 1]
                    next_k = front[i + 1]
                    dist = (self._vectors[next_k][dim] - self._vectors[prev_k][dim]) / denom
                    distances[front[i]] += dist

        # Update global dictionary.
        self._crowding_distances.update(distances)

        # Sort the front in descending order of crowding distance.
        front.sort(key=lambda k: self._crowding_distances[k], reverse=True)

    def _find_front_index_of_key(self, key: Key) -> int | None:
        """
        Find which front contains a given trial.

        :param key: Trial identifier to locate
        :type key: Key
        :return: Index of the containing front, or None if not found
        :rtype: int | None
        """
        for i, fr in enumerate(self._fronts):
            if key in fr:
                return i
        return None


def dominates(a: FloatArray, b: FloatArray) -> bool:
    """
    Check if vector a dominates vector b in the Pareto sense.

    A vector a dominates b if it is at least as good in all dimensions and
    strictly better in at least one dimension.

    :param a: First vector
    :type a: FloatArray
    :param b: Second vector
    :type b: FloatArray
    :return: True if a dominates b, False otherwise
    :rtype: bool
    """
    return np.all(a <= b) and np.any(a < b)
