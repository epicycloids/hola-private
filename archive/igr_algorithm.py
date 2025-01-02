"""Iterative Grid Refinement (IGR) algorithm for hyperparameter optimization.

This module implements the IGR algorithm, which performs optimization by iteratively:
1. Creating a grid of points in the parameter space
2. Evaluating the objective function at these points
3. Finding the best point
4. Creating a smaller grid around the best point
5. Repeating until max iterations reached

The algorithm is particularly effective for low-dimensional problems (d < 4).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable, Final, Sequence

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field, field_validator

from hola.params import ParamConfig, ParamName, ParamsSpec, parse_params_config

# Constants
DEFAULT_SPACING: Final[int] = 3
MIN_DIMENSION_WARNING: Final[int] = 4


@dataclass(frozen=True)
class Side:
    """One dimension (side) of a hypercube.

    :param lower: Lower bound (normalized to [0, 1])
    :type lower: float
    :param upper: Upper bound (normalized to [0, 1])
    :type upper: float
    :raises ValueError: If bounds are invalid
    """

    lower: float
    upper: float

    def __post_init__(self) -> None:
        """Validate bounds."""
        if not 0 <= self.lower <= self.upper <= 1:
            raise ValueError(
                f"Side bounds must satisfy 0 ≤ lower ≤ upper ≤ 1, got: [{self.lower}, {self.upper}]"
            )

    @property
    def length(self) -> float:
        """Side length.

        :return: Length of the side
        :rtype: float
        """
        return self.upper - self.lower

    def get_lattices(self, spacing: int) -> list[float]:
        """Split the side into equally spaced points.

        :param spacing: Number of intervals to split into
        :type spacing: int
        :return: List of points including bounds
        :rtype: list[float]
        :raises ValueError: If spacing < 1
        """
        if spacing < 1:
            raise ValueError(f"Spacing must be ≥ 1, got: {spacing}")
        return [self.lower + self.length * i / spacing for i in range(spacing + 1)]

    def shrink_around(self, new_center: float, spacing: int) -> Side:
        """Create a new, smaller side centered around a point.

        Creates a new side with length reduced by factor of spacing
        and centered on the given point, while ensuring bounds stay in [0, 1].

        :param new_center: Point to center new side around
        :type new_center: float
        :param spacing: Factor to shrink by
        :type spacing: int
        :return: New, smaller Side
        :rtype: Side
        :raises ValueError: If center not in [0, 1] or spacing < 1
        """
        if not 0 <= new_center <= 1:
            raise ValueError(f"Center must be in [0,1], got: {new_center}")
        if spacing < 1:
            raise ValueError(f"Spacing must be ≥ 1, got: {spacing}")

        new_half_length = self.length / (2 * spacing)
        new_lower = np.clip(new_center - new_half_length, 0, 1)
        new_upper = np.clip(new_center + new_half_length, 0, 1)
        return Side(new_lower, new_upper)


@dataclass(frozen=True)
class Hypercube:
    """N-dimensional hypercube with sides normalized to [0, 1].

    :param sides: Dictionary mapping parameter names to their sides
    :type sides: dict[ParamName, Side]
    """

    sides: dict[ParamName, Side]

    @classmethod
    def unit(cls, params_config: dict[ParamName, ParamConfig]) -> Hypercube:
        """Create initial unit hypercube [0,1]^n.

        :param params_config: Parameter configuration
        :type params_config: dict[ParamName, ParamConfig]
        :return: Unit hypercube
        :rtype: Hypercube
        """
        return cls({param_name: Side(0, 1) for param_name in params_config})

    def get_lattices(self, spacing: int) -> list[dict[ParamName, float]]:
        """Create grid of points in the hypercube.

        :param spacing: Number of intervals in each dimension
        :type spacing: int
        :return: List of parameter dictionaries for each grid point
        :rtype: list[dict[ParamName, float]]
        :raises ValueError: If spacing < 1
        """
        if spacing < 1:
            raise ValueError(f"Spacing must be ≥ 1, got: {spacing}")

        lattices = {param: side.get_lattices(spacing) for param, side in self.sides.items()}

        # Create all combinations
        # N.B. This is surprisingly faster than the Pandas approach, as well as
        # an approach using np.meshgrid and an approach using itertools.product
        points = []
        params = list(self.sides)
        first_param = params[0]

        for val in lattices[first_param]:
            point = {first_param: val}
            points.append(point)

        for param in params[1:]:
            new_points = []
            for val in lattices[param]:
                for point in points:
                    new_point = point.copy()
                    new_point[param] = val
                    new_points.append(new_point)
            points = new_points

        return points

    def shrink_around(self, new_center: dict[ParamName, float], spacing: int) -> Hypercube:
        """Create smaller hypercube centered around a point.

        :param new_center: Point to center new hypercube around
        :type new_center: dict[ParamName, float]
        :param spacing: Factor to shrink by
        :type spacing: int
        :return: New, smaller hypercube
        :rtype: Hypercube
        :raises ValueError: If parameters don't match or values invalid
        """
        if set(new_center) != set(self.sides):
            raise ValueError("Center point parameters don't match hypercube")

        return Hypercube(
            {
                param: self.sides[param].shrink_around(value, spacing)
                for param, value in new_center.items()
            }
        )


@dataclass(frozen=True, order=True)
class Evaluation:
    """Result of evaluating objective function for a set of parameters.

    :param params: Parameter values used
    :type params: dict[ParamName, float]
    :param val: Objective function value
    :type val: float
    """

    val: float = field(compare=True)
    params: dict[ParamName, float] = field(compare=False)


class IterativeGridRefinement:
    """Iterative Grid Refinement (IGR) algorithm for hyperparameter optimization.

    IGR works by iteratively creating grids of points in the parameter space,
    evaluating the objective function at these points, and creating finer grids
    around the best points found. The algorithm is particularly effective for
    low-dimensional problems (d < 4).
    """

    def __init__(
        self,
        params_config: ParamsSpec,
        spacing: int = DEFAULT_SPACING,
    ) -> None:
        """Initialize IGR algorithm.

        :param params_config: Configuration of parameter search spaces
        :type params_config: ParamsSpec
        :param spacing: Number of intervals to split each dimension into
        :type spacing: int
        :raises ValueError: If params_config is empty or invalid
        :raises TypeError: If params contains non-float parameters
        """
        self.params_config = parse_params_config(params_config)
        if not self.params_config:
            raise ValueError("params_config cannot be empty")

        for name, config in self.params_config.items():
            if config.param_type != "float":
                raise TypeError(
                    f"IGR only supports float parameters, but {name} has type {config.param_type}"
                )

        self.num_params = len(self.params_config)
        if self.num_params >= MIN_DIMENSION_WARNING:
            warnings.warn(
                f"IGR is not recommended for dimensionality ≥ {MIN_DIMENSION_WARNING} "
                f"(got {self.num_params} dimensions)",
                RuntimeWarning,
            )

        if spacing < 1:
            raise ValueError(f"Spacing must be ≥ 1, got: {spacing}")
        if spacing > 10:
            warnings.warn(
                f"Large spacing value ({spacing}) may lead to excessive grid points", RuntimeWarning
            )
        self.spacing = spacing

    def tune(
        self,
        func: Callable[[Sequence[float]], float],
        max_iterations: int,
    ) -> Evaluation:
        """Run the IGR optimization algorithm.

        :param func: Objective function to minimize
        :type func: Callable[[Sequence[float]], float]
        :param max_iterations: Maximum number of function evaluations
        :type max_iterations: int
        :return: Best evaluation found
        :rtype: Evaluation
        :raises ValueError: If max_iterations < 1

        **Notes:**

        - The algorithm will stop early if all grid points have been evaluated
        - The objective function should accept a sequence of float values in
            the same order as params_config
        - The objective function should return a float value to minimize
        """
        if max_iterations < 1:
            raise ValueError(f"max_iterations must be ≥ 1, got: {max_iterations}")

        hypercube = Hypercube.unit(self.params_config)
        max_generations = self.get_number_of_generations(max_iterations)

        if self.get_number_of_iterations(max_generations) < max_iterations:
            warnings.warn(
                f"Maximum iterations {max_iterations} cannot be reached with "
                f"current spacing {self.spacing} and dimensionality {self.num_params}. "
                f"Will stop after {self.get_number_of_iterations(max_generations)} iterations.",
                RuntimeWarning,
            )

        iteration = 0
        params_tried: set[tuple[float, ...]] = set()
        evaluations: dict[int, list[Evaluation]] = {}  # generation -> evaluations

        for generation in range(max_generations):
            # Find lattices in the normalized hypercube
            norm_samples = hypercube.get_lattices(self.spacing)

            # Map params back to their domain and evaluate the function
            generation_evals = []
            for sample in norm_samples:
                denorm_params = {
                    param_name: config.transform_param(sample[param_name])
                    for param_name, config in self.params_config.items()
                }
                params_to_try = tuple(denorm_params.values())

                # Skip if we've already tried these parameters
                if params_to_try in params_tried:
                    continue

                # Evaluate function and record result
                try:
                    result = func(params_to_try)
                except Exception as e:
                    warnings.warn(
                        f"Function evaluation failed for parameters {params_to_try}: {str(e)}",
                        RuntimeWarning,
                    )
                    continue

                evaluation = Evaluation(result, denorm_params)
                generation_evals.append(evaluation)
                params_tried.add(params_to_try)

                iteration += 1
                if iteration >= max_iterations:
                    evaluations[generation] = generation_evals
                    return self.get_best_evaluation(evaluations)

            # Store this generation's evaluations
            evaluations[generation] = generation_evals
            if not generation_evals:
                break  # No new points to evaluate
            best_evaluation_in_generation = min(generation_evals)

            # Shrink hypercube around the normalized best params of this generation
            hypercube = hypercube.shrink_around(
                {
                    param_name: config.back_transform_param(
                        best_evaluation_in_generation.params[param_name]
                    )
                    for param_name, config in self.params_config.items()
                },
                self.spacing,
            )

        return self.get_best_evaluation(evaluations)

    def get_number_of_iterations(self, max_generations: int) -> int:
        """Calculate maximum possible iterations for given generations.

        :param max_generations: Number of generations
        :type max_generations: int
        :return: Maximum possible iterations
        :rtype: int
        """
        points_per_generation = (self.spacing + 1) ** self.num_params
        return points_per_generation * max_generations

    def get_number_of_generations(self, max_iterations: int) -> int:
        """Calculate number of generations possible within iteration limit.

        :param max_iterations: Maximum number of iterations allowed
        :type max_iterations: int
        :return: Number of possible generations
        :rtype: int
        """
        points_per_generation = (self.spacing + 1) ** self.num_params
        return (max_iterations // points_per_generation) + 1

    @staticmethod
    def get_best_evaluation(evaluations: dict[int, list[Evaluation]]) -> Evaluation:
        """Get best evaluation across all generations.

        :param evaluations: Dictionary mapping generation to list of evaluations
        :type evaluations: dict[int, list[Evaluation]]
        :return: Best evaluation found
        :rtype: Evaluation
        :raises ValueError: If no evaluations exist
        """
        if not evaluations:
            raise ValueError("No evaluations found")

        all_evals = [eval for gen_evals in evaluations.values() for eval in gen_evals]

        if not all_evals:
            raise ValueError("No valid evaluations found")

        return min(all_evals)
