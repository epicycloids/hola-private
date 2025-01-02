"""Sorted population data structures for optimization.

This module provides data structures for maintaining sorted populations of
solutions, supporting both scalar (single-objective) and vector
(multi-objective) sorting strategies. Key features include:

- Scalar sorting for single-group optimization
- Vector sorting with Pareto fronts for multi-group optimization
- Crowding distance calculation for diversity preservation
- Efficient updates and queries of population statistics

The module implements both single-objective total ordering and multi-objective
partial ordering with Pareto dominance relationships.

Example:
    >>> # Single-group (scalar) sorting
    >>> pop = ScalarSortedPopulation()
    >>> pop.add("sample1", np.array([0.5]))
    >>> pop.add("sample2", np.array([0.3]))
    >>> pop.get_sorted_labels()
    ["sample2", "sample1"]

    >>> # Multi-group (vector) sorting
    >>> pop = VectorSortedPopulation()
    >>> pop.add("sample1", np.array([0.5, 0.3]))
    >>> pop.add("sample2", np.array([0.3, 0.4]))
    >>> pop.get_front(0)  # Get Pareto front
    ["sample2", "sample1"]
"""

from __future__ import annotations

import warnings
from typing import Final, NewType, Protocol, TypeVar

import numpy as np
import numpy.typing as npt
from sortedcontainers import SortedKeyList

# Type definitions
Score = NewType('Score', npt.NDArray[np.float64])
Label = TypeVar("Label")

# Constants
EPSILON: Final[float] = 1e-10  # Threshold for float comparisons


class SortedPopulation(Protocol[Label]):
    """Protocol for sorted population data structures.

    This protocol defines the interface for tracking and sorting solutions,
    whether using scalar or vector comparisons. Single-group (scalar) sorting
    is treated as a special case of multi-group sorting.

    The protocol ensures consistent behavior across different sorting
    strategies while allowing specialized implementations for different
    comparison types.
    """

    def add(self, label: Label, score: Score) -> None:
        """Add a new sample to the population.

        :param label: Unique identifier for the sample
        :type label: Label
        :param score: Score value(s) for the sample
        :type score: npt.NDArray
        :raises ValueError: If label already exists
        """
        ...

    def get_sorted_labels(self) -> list[Label]:
        """Get labels sorted by score(s).

        For multi-group sorting, orders by non-domination rank and
        crowding distance.

        :return: List of labels in sorted order
        :rtype: list[Label]
        """
        ...

    def num_samples(self) -> int:
        """Get number of samples in population.

        :return: Number of samples
        :rtype: int
        """
        ...

    def get_score(self, label: Label) -> Score:
        """Get score for a sample.

        :param label: Sample identifier
        :type label: Label
        :return: Score value(s)
        :rtype: npt.NDArray
        :raises KeyError: If label not found
        """
        ...

    def update_score(self, label: Label, new_score: Score) -> None:
        """Update score for existing sample.

        :param label: Sample identifier
        :type label: Label
        :param new_score: New score value(s)
        :type new_score: npt.NDArray
        :raises KeyError: If label not found
        """
        ...

    def get_front(self, level: int) -> list[Label]:
        """Get samples in specified Pareto front level.

        For single-group case, each front contains one sample.
        For multi-group case, fronts contain non-dominated samples.

        :param level: Front level (0 is best front)
        :type level: int
        :return: Labels of samples in that front
        :rtype: list[Label]
        :raises ValueError: If level is invalid
        """
        ...

    def get_front_index(self, label: Label) -> int:
        """Get Pareto front level for a sample.

        :param label: Sample identifier
        :type label: Label
        :return: Front level (0 is best front)
        :rtype: int
        :raises KeyError: If label not found
        """
        ...

    def get_crowding_distance(self, label: Label) -> float:
        """Get crowding distance for a sample.

        For single-group case, returns infinity.
        For multi-group case, returns distance to neighboring points.

        :param label: Sample identifier
        :type label: Label
        :return: Crowding distance
        :rtype: float
        :raises KeyError: If label not found
        """
        ...

    def get_num_fronts(self) -> int:
        """Get total number of Pareto front levels.

        :return: Number of front levels
        :rtype: int
        """
        ...

    def get_top_samples(self, num_samples: int) -> list[Label]:
        """Get best samples by rank and crowding distance.

        :param num_samples: Number of samples to return
        :type num_samples: int
        :return: List of top sample labels
        :rtype: list[Label]
        """
        ...


class ScalarSortedPopulation(SortedPopulation[Label]):
    """Population sorted by scalar scores (single-group case).

    In the single-group case, scores form a total order, so each front contains
    exactly one point and all points have infinite crowding distance. This
    implementation provides efficient updates and queries for scalar
    objectives.

    Example:
        >>> # Create population for minimizing training time
        >>> pop = ScalarSortedPopulation[str]()
        >>> pop.add("fast_config", np.array([60.0]))
        >>> pop.add("slow_config", np.array([120.0]))
        >>> best = pop.get_top_samples(1)[0]
        >>> print(best)
        fast_config
    """

    def __init__(self) -> None:
        """Initialize scalar sorted population.

        The population maintains an internal sorted data structure for
        efficient ranking and retrieval of samples.
        """
        self.scores: dict[Label, Score] = {}
        self.sorted_labels = SortedKeyList(key=lambda label: self.scores[label][0])

    def add(self, label: Label, score: Score) -> None:
        """Add a new sample to the population.

        Example:
            >>> pop = ScalarSortedPopulation[str]()
            >>> # Add sample with training time score
            >>> pop.add("config_1", np.array([120.0]))

        :param label: Unique identifier for the sample
        :type label: Label
        :param score: Single score value as 1D numpy array
        :type score: npt.NDArray
        :raises ValueError: If label already exists in population or
            score is not 1-dimensional
        """
        if label in self.scores:
            raise ValueError(f"Sample {label} already exists in population")

        if score.ndim != 1 or len(score) != 1:
            raise ValueError("Score must be 1-dimensional with a single value")

        self.scores[label] = score
        self.sorted_labels.add(label)

    def get_sorted_labels(self) -> list[Label]:
        """Get labels sorted by score value from best to worst.

        For scalar populations, samples are sorted in ascending order of
        their score values, treating lower scores as better.

        :return: List of labels sorted by score
        :rtype: list[Label]
        """
        return list(self.sorted_labels)

    def num_samples(self) -> int:
        """Get number of samples in the population.

        :return: Total number of samples
        :rtype: int
        """
        return len(self.sorted_labels)

    def get_score(self, label: Label) -> Score:
        """Get score for a sample.

        Example:
            >>> score = pop.get_score("config_1")
            >>> print(f"Training time: {score[0]:.1f}s")

        :param label: Sample identifier
        :type label: Label
        :return: Score value as 1D numpy array
        :rtype: npt.NDArray
        :raises KeyError: If label not found in population
        """
        if label not in self.scores:
            raise KeyError(f"Label {label} not found")
        return self.scores[label]

    def update_score(self, label: Label, new_score: Score) -> None:
        """Update score for existing sample and resort population.

        Example:
            >>> # Update training time after optimization
            >>> pop.update_score("config_1", np.array([90.0]))

        :param label: Sample identifier
        :type label: Label
        :param new_score: New score value as 1D numpy array
        :type new_score: npt.NDArray
        :raises KeyError: If label not found in population
        """
        if label not in self.scores:
            raise KeyError(f"Label {label} not found")
        self.sorted_labels.remove(label)
        self.scores[label] = new_score
        self.sorted_labels.add(label)

    def get_front(self, level: int) -> list[Label]:
        """Get sample in specified front level.

        For scalar populations, each front contains exactly one sample,
        with front level corresponding to position in sorted order.

        :param level: Front level (0 is best front)
        :type level: int
        :return: List containing single label at that level
        :rtype: list[Label]
        :raises ValueError: If level >= number of samples
        """
        if level >= len(self.sorted_labels):
            raise ValueError(f"Invalid front level: {level}")
        return [self.sorted_labels[level]]

    def get_front_index(self, label: Label) -> int:
        """Get front level (position) for a sample.

        For scalar populations, front index is equivalent to position
        in the sorted order of samples.

        :param label: Sample identifier
        :type label: Label
        :return: Front level (0 is best front)
        :rtype: int
        :raises KeyError: If label not found in population
        """
        if label not in self.scores:
            raise KeyError(f"Label {label} not found")
        return self.sorted_labels.index(label)

    def get_crowding_distance(self, label: Label) -> float:
        """Get crowding distance for a sample.

        For scalar populations, crowding distance is always infinity since
        samples form a total order with no need for secondary sorting.

        :param label: Sample identifier
        :type label: Label
        :return: Infinity
        :rtype: float
        :raises KeyError: If label not found in population
        """
        if label not in self.scores:
            raise KeyError(f"Label {label} not found")
        return float("inf")

    def get_num_fronts(self) -> int:
        """Get number of fronts in population.

        For scalar populations, number of fronts equals number of samples
        since each sample forms its own front.

        :return: Number of fronts
        :rtype: int
        """
        return len(self.sorted_labels)

    def get_top_samples(self, num_samples: int) -> list[Label]:
        """Get best samples by score value.

        Example:
            >>> # Get 3 fastest configurations
            >>> top_3 = pop.get_top_samples(3)
            >>> for label in top_3:
            ...     time = pop.get_score(label)[0]
            ...     print(f"{label}: {time:.1f}s")

        :param num_samples: Number of samples to return
        :type num_samples: int
        :return: List of top sample labels
        :rtype: list[Label]
        """
        if not self.scores or num_samples <= 0:
            return []
        return self.get_sorted_labels()[:num_samples]


class VectorSortedPopulation(SortedPopulation[Label]):
    """Population sorted by vector scores (multi-group case).

    In the multi-group case, scores form a partial order where points are
    sorted by non-domination rank (front level) and crowding distance within
    fronts. This implementation efficiently maintains Pareto fronts and
    crowding distances.

    Example:
        >>> # Create population for optimizing accuracy and training time
        >>> pop = VectorSortedPopulation[str]()
        >>> # Add sample with 95% accuracy, 120s training time
        >>> pop.add("config1", np.array([0.95, 120.0]))
        >>> # Add sample with 90% accuracy, 60s training time
        >>> pop.add("config2", np.array([0.90, 60.0]))
        >>> # Both samples are non-dominated (Pareto front 0)
        >>> front_0 = pop.get_front(0)
        >>> print(len(front_0))  # 2
    """

    def __init__(self) -> None:
        """Initialize vector sorted population.

        The population maintains data structures for:
        - Score values
        - Pareto front assignments
        - Domination relationships
        - Crowding distances
        """
        self.scores: dict[Label, Score] = {}
        self.fronts: list[set[Label]] = []
        self.domination_counts: dict[Label, int] = {}
        self.dominated_solutions: dict[Label, set[Label]] = {}
        self.crowding_distances: dict[Label, float] = {}
        self.sorted_fronts: list[list[Label]] = []

    def add(self, label: Label, score: Score) -> None:
        """Add a new sample to the population.

        Example:
            >>> pop = VectorSortedPopulation[str]()
            >>> # Add sample with accuracy and training time scores
            >>> pop.add("config_1", np.array([0.95, 120.0]))

        :param label: Unique identifier for the sample
        :type label: Label
        :param score: Vector of score values as 1D numpy array
        :type score: npt.NDArray
        :raises ValueError: If label already exists in population or score
            dimensions don't match
        """
        if label in self.scores:
            raise ValueError(f"Sample {label} already exists in population")

        # Check score dimensions match existing samples
        if self.scores:
            first_score = next(iter(self.scores.values()))
            if score.shape != first_score.shape:
                raise ValueError(
                    f"Score dimensions must match existing samples: "
                    f"expected shape {first_score.shape}, got {score.shape}"
                )

        # Store sample data
        self.scores[label] = score
        self.domination_counts[label] = 0
        self.dominated_solutions[label] = set()

        # Process domination relationships and update fronts
        self._update_domination_relationships_new_sample(label)
        self._assign_fronts()

    def get_sorted_labels(self) -> list[Label]:
        """Get labels sorted by non-domination rank and crowding distance.

        Returns samples ordered first by Pareto front level (rank), then
        by crowding distance within each front. Lower ranks and higher
        crowding distances are considered better.

        :return: List of sorted labels
        :rtype: list[Label]
        """
        return [label for front in self.sorted_fronts for label in front]

    def num_samples(self) -> int:
        """Get number of samples in the population.

        :return: Total number of samples
        :rtype: int
        """
        return len(self.scores)

    def get_score(self, label: Label) -> Score:
        """Get score for a sample.

        Example:
            >>> score = pop.get_score("config_1")
            >>> print(f"Accuracy: {score[0]:.1%}")
            >>> print(f"Training time: {score[1]:.1f}s")

        :param label: Sample identifier
        :type label: Label
        :return: Vector of score values as 1D numpy array
        :rtype: npt.NDArray
        :raises KeyError: If label not found in population
        """
        if label not in self.scores:
            raise KeyError(f"Label {label} not found")
        return self.scores[label]

    def update_score(self, label: Label, new_score: Score) -> None:
        """Update score for existing sample and recompute fronts.

        Updates the sample's score and recalculates all domination
        relationships and Pareto fronts accordingly.

        Example:
            >>> # Update both accuracy and training time
            >>> pop.update_score("config_1", np.array([0.97, 90.0]))

        :param label: Sample identifier
        :type label: Label
        :param new_score: New vector of score values
        :type new_score: npt.NDArray
        :raises KeyError: If label not found in population
        """
        if label not in self.scores:
            raise KeyError(f"Label {label} not found")

        old_score = self.scores[label]
        self.scores[label] = new_score

        # Update domination relationships and fronts
        self._update_domination_relationships_updated_sample(label, old_score, new_score)
        self._assign_fronts()

    def get_front(self, level: int) -> list[Label]:
        """Get samples in specified Pareto front level.

        Returns all samples in the given front, sorted by crowding
        distance. Front 0 contains the non-dominated solutions, front 1
        contains solutions dominated only by front 0, and so on.

        Example:
            >>> # Get all non-dominated solutions
            >>> pareto_front = pop.get_front(0)
            >>> print(f"Found {len(pareto_front)} Pareto-optimal configs")

        :param level: Front level (0 is best front)
        :type level: int
        :return: List of labels in front, sorted by crowding distance
        :rtype: list[Label]
        :raises ValueError: If level >= number of fronts
        """
        if level >= len(self.sorted_fronts):
            raise ValueError(f"Invalid front level: {level}")
        return self.sorted_fronts[level]

    def get_front_index(self, label: Label) -> int:
        """Get Pareto front level for a sample.

        Returns the non-domination rank of the sample, where 0 indicates
        Pareto-optimal solutions.

        :param label: Sample identifier
        :type label: Label
        :return: Front level (0 is best front)
        :rtype: int
        :raises KeyError: If label not found in population
        :raises RuntimeError: If label exists but not assigned to a front
        """
        if label not in self.scores:
            raise KeyError(f"Label {label} not found")

        for idx, front in enumerate(self.fronts):
            if label in front:
                return idx
        raise RuntimeError(f"Label {label} not found in any front")

    def get_crowding_distance(self, label: Label) -> float:
        """Get crowding distance for a sample.

        Returns the crowding distance metric used to maintain diversity
        within Pareto fronts. Larger values indicate more isolated points
        that help maintain population diversity.

        :param label: Sample identifier
        :type label: Label
        :return: Crowding distance value
        :rtype: float
        :raises KeyError: If label not found in population
        """
        if label not in self.scores:
            raise KeyError(f"Label {label} not found")
        return self.crowding_distances.get(label, 0.0)

    def get_num_fronts(self) -> int:
        """Get total number of Pareto front levels.

        :return: Number of fronts
        :rtype: int
        """
        return len(self.fronts)

    def get_top_samples(self, num_samples: int) -> list[Label]:
        """Get best samples by non-domination rank and crowding distance.

        Returns samples ordered first by Pareto front level, then by
        crowding distance within fronts. This provides a total ordering
        that balances optimality and diversity.

        Example:
            >>> # Get top 5 configurations
            >>> best_configs = pop.get_top_samples(5)
            >>> for label in best_configs:
            ...     score = pop.get_score(label)
            ...     print(f"{label}: {score[0]:.1%} acc, {score[1]:.1f}s")

        :param num_samples: Number of samples to return
        :type num_samples: int
        :return: List of top sample labels
        :rtype: list[Label]
        """
        if num_samples <= 0:
            return []

        top_samples = []
        for front in self.sorted_fronts:
            if len(top_samples) + len(front) <= num_samples:
                top_samples.extend(front)
                if len(top_samples) == num_samples:
                    break
            else:
                remaining = num_samples - len(top_samples)
                top_samples.extend(front[:remaining])
                break

        return top_samples

    def _dominates(self, a: Score, b: Score) -> bool:
        """Check if score vector a dominates score vector b.

        A score vector dominates another if it is at least as good in all
        coordinates and strictly better in at least one coordinate. "Better"
        means lower value.

        Example:
            >>> # a dominates b (better in both coordinates)
            >>> a = np.array([0.90, 60.0])  # 90% accuracy, 60s
            >>> b = np.array([0.95, 120.0]) # 95% accuracy, 120s
            >>> pop._dominates(a, b)
            True

        :param a: First score vector
        :type a: npt.NDArray
        :param b: Second score vector
        :type b: npt.NDArray
        :return: True if a dominates b, False otherwise
        :rtype: bool
        """
        return np.all(a <= b) and np.any(a < b)

    def _update_domination_relationships_new_sample(self, label: Label) -> None:
        """Update domination relationships for a new sample.

        Updates the domination count and dominated solutions sets when a new
        sample is added to the population.

        :param label: Label of newly added sample
        :type label: Label
        """
        for other_label in self.scores:
            if other_label == label:
                continue

            other_score = self.scores[other_label]
            if self._dominates(self.scores[label], other_score):
                self.dominated_solutions[label].add(other_label)
                self.domination_counts[other_label] += 1
            elif self._dominates(other_score, self.scores[label]):
                self.dominated_solutions[other_label].add(label)
                self.domination_counts[label] += 1

    def _update_domination_relationships_updated_sample(
        self, label: Label, old_score: Score, new_score: Score
    ) -> None:
        """Update domination relationships for a sample with changed score.

        Efficiently updates domination relationships when a sample's score
        changes by only checking relationships that might have changed.

        :param label: Label of updated sample
        :type label: Label
        :param old_score: Previous score vector
        :type old_score: npt.NDArray
        :param new_score: New score vector
        :type new_score: npt.NDArray
        """
        for other_label in self.scores:
            if other_label == label:
                continue

            other_score = self.scores[other_label]

            # Check if old relationships need to be reversed or removed
            if self._dominates(old_score, other_score):
                if not self._dominates(new_score, other_score):
                    self.dominated_solutions[label].discard(other_label)
                    self.domination_counts[other_label] -= 1
            elif self._dominates(other_score, old_score):
                if not self._dominates(other_score, new_score):
                    self.dominated_solutions[other_label].discard(label)
                    self.domination_counts[label] -= 1

            # Check if new relationships need to be added
            if self._dominates(new_score, other_score):
                if not self._dominates(old_score, other_score):
                    self.dominated_solutions[label].add(other_label)
                    self.domination_counts[other_label] += 1
            elif self._dominates(other_score, new_score):
                if not self._dominates(other_score, old_score):
                    self.dominated_solutions[other_label].add(label)
                    self.domination_counts[label] += 1

    def _assign_fronts(self) -> None:
        """Assign samples to Pareto fronts and compute crowding distances.

        Implements the fast non-dominated sorting algorithm:
        1. Identifies non-dominated samples for the current front
        2. Removes their domination effect on remaining samples
        3. Repeats until all samples are assigned to fronts
        4. Computes crowding distances within each front

        A warning is raised if inconsistent domination relationships
        prevent all samples from being assigned to fronts.
        """
        self.fronts = []
        processed = set()

        current_domination_counts = self.domination_counts.copy()

        while len(processed) < len(self.scores):
            # Find samples with no unprocessed dominators
            current_front = {
                label
                for label, count in current_domination_counts.items()
                if count == 0 and label not in processed
            }

            if not current_front:
                warnings.warn(
                    "No samples found for next front but not all samples processed. "
                    "This suggests inconsistent domination relationships. "
                    f"Processed {len(processed)}/{len(self.scores)} samples.",
                    RuntimeWarning,
                )
                break

            # Add front and compute its crowding distances
            self.fronts.append(current_front)
            self._compute_crowding_distance_for_front(current_front)

            # Update domination counts
            for label in current_front:
                processed.add(label)
                for dominated in self.dominated_solutions[label]:
                    if dominated not in processed:
                        current_domination_counts[dominated] -= 1

        # Store sorted fronts for efficient get_top_samples
        self.sorted_fronts = [
            sorted(
                list(front),
                key=lambda label: self.crowding_distances.get(label, 0.0),
                reverse=True
            )
            for front in self.fronts
        ]

    def _compute_crowding_distance_for_front(self, front: set[Label]) -> None:
        """Compute crowding distances for samples in a Pareto front.

        For each objective, sorts samples by their score and assigns
        distances based on the score difference between neighboring
        solutions. The total crowding distance is the sum across all
        objectives. Boundary points (best/worst in any objective) get
        infinite distance to preserve extreme solutions.

        Example:
            >>> front = {"config1", "config2", "config3"}
            >>> pop._compute_crowding_distance_for_front(front)
            >>> # Check distances
            >>> for label in front:
            ...     dist = pop.get_crowding_distance(label)
            ...     print(f"{label}: {dist:.2f}")

        :param front: Set of labels in the same front
        :type front: set[Label]
        """
        front_list = list(front)
        if len(front_list) <= 2:
            for label in front_list:
                self.crowding_distances[label] = float("inf")
            return

        num_objectives = len(next(iter(self.scores.values())))
        for label in front_list:
            self.crowding_distances[label] = 0.0

        for m in range(num_objectives):
            # Sort front by objective m
            front_list.sort(key=lambda x: self.scores[x][m])

            # Set boundary points to infinity
            self.crowding_distances[front_list[0]] = float("inf")
            self.crowding_distances[front_list[-1]] = float("inf")

            # Compute distances for intermediate points
            norm = self.scores[front_list[-1]][m] - self.scores[front_list[0]][m]
            if norm < EPSILON:  # Avoid division by zero
                continue

            for i in range(1, len(front_list) - 1):
                distance = (
                    self.scores[front_list[i + 1]][m] - self.scores[front_list[i - 1]][m]
                ) / norm
                self.crowding_distances[front_list[i]] += distance

    def _remove_sample(self, label: Label) -> None:
        """Remove a sample and update population data structures.

        :param label: Label of sample to remove
        :type label: Label
        """
        # Remove basic data
        del self.scores[label]
        del self.domination_counts[label]
        del self.dominated_solutions[label]
        self.crowding_distances.pop(label, None)

        # Remove from fronts
        for front in self.fronts:
            front.discard(label)

        # Update domination counts
        for dominated_set in self.dominated_solutions.values():
            dominated_set.discard(label)