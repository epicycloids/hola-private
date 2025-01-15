"""Leaderboard for tracking optimization results.

This module provides a Leaderboard class for tracking and ranking hyperparameter
optimization trials. Key features include:

- Caching of parameters and their resulting objectives
- Scoring of single and multi-group objectives
- Pareto-optimal solution tracking
- Export/import of results to CSV format
- Trial filtering and result analysis

The leaderboard maintains consistent ordering of results and efficient access
to the best performing configurations.

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
    >>>
    >>> # Create leaderboard with objective scorer
    >>> leaderboard = Leaderboard(scorer)
    >>>
    >>> # Add optimization results
    >>> leaderboard.add(
    ...     params={"learning_rate": 0.001, "batch_size": 32},
    ...     objectives={"accuracy": 0.95, "training_time": 120}
    ... )
    >>>
    >>> # Get best results
    >>> best_params, best_objectives = leaderboard.get_best_sample()
    >>> print(f"Best accuracy: {best_objectives['accuracy']:.2f}")
    Best accuracy: 0.95
    >>> # Save results for analysis
    >>> leaderboard.save("optimization_results.csv")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import pandas as pd
from pydantic import Field

from hola.core.objective import ObjectiveConfig, ObjectiveName, ObjectiveScorer, GroupId
from hola.core.params_old import ParamName
from hola.core.sorted_population import TotallyOrderedPopulation, PartiallyOrderedPopulation
from hola.core.utils import BaseConfig


# Column prefix constants
PARAM_PREFIX: Final[str] = "param_"
OBJECTIVE_PREFIX: Final[str] = "objective_"
GROUP_SCORE_PREFIX: Final[str] = "group_score_"


class Trial(BaseConfig):
    """Configuration for a single hyperparameter optimization trial.

    Stores both the parameter settings used and the resulting objective values.
    This immutable configuration ensures consistency between parameters and
    their outcomes.

    Example:
        >>> trial = Trial(
        ...     params={"learning_rate": 0.001},
        ...     objectives={"accuracy": 0.95}
        ... )
        >>> print(f"LR: {trial.params['learning_rate']}")
        LR: 0.001
        >>> print(f"Accuracy: {trial.objectives['accuracy']}")
        Accuracy: 0.95

    :param params: Parameter name to value mapping
    :type params: dict[str, float]
    :param objectives: Objective name to value mapping
    :type objectives: dict[str, float]
    """
    params: dict[ParamName, Any] = Field(
        ...,
        description="Parameter values used in trial"
    )
    objectives: dict[ObjectiveName, float] = Field(
        ...,
        description="Objective values achieved in trial"
    )
    group_values: dict[GroupId, float] = Field(
        ...,
        description="Comparison group values achieved in trial"
    )

    def __lt__(self, other: "Trial") -> bool:
        """Compares two trials based on their comparison group values.

        Returns True if all comparison group values in `self` are less than or
        equal to the corresponding comparison group values in `other`, and at
        least one value is strictly less than its counterpart in `other`.

        :param other: The other Trial to compare against.
        :type other: Trial
        :raises ValueError: If the trials have different parameters,
            objectives, or comparison group IDs.
        :return: True if self < other, False otherwise.
        :rtype: bool
        """
        if self.params.keys() != other.params.keys():
            raise ValueError("Trials must have the same parameter names.")

        if self.objectives.keys() != other.objectives.keys():
            raise ValueError("Trials must have the same objective names.")

        if self.group_values.keys() != other.group_values.keys():
            raise ValueError("Trials must have the same comparison group names.")

        self_group_values = self.group_values.values()
        other_group_values = other.group_values.values()

        all_less_equal = all(sv <= ov for sv, ov in zip(self_group_values, other_group_values))
        at_least_one_less = any(sv < ov for sv, ov in zip(self_group_values, other_group_values))

        return all_less_equal and at_least_one_less


class Leaderboard:
    """Cache and scoring interface for optimization trials.

    The leaderboard maintains a sorted population of solutions, computing their
    scores from objective values, and caches parameter settings with their
    resulting objective values. It supports both single-objective and
    multi-objective optimization through appropriate population sorting
    strategies.

    Example:
        >>> # Create leaderboard for accuracy optimization
        >>> scorer = ObjectiveScorer(...)
        >>> leaderboard = Leaderboard(scorer)
        >>>
        >>> # Add multiple trials
        >>> for params in parameter_samples:
        ...     objectives = run_training(params)
        ...     leaderboard.add(params, objectives)
        >>>
        >>> # Get top N configurations
        >>> top_params, top_objectives = leaderboard.get_top_samples(5)
        >>>
        >>> # Save results for later analysis
        >>> leaderboard.save("results.csv")

    :param objective_scorer: Scorer for computing scores from objectives
    :type objective_scorer: ObjectiveScorer
    """

    @classmethod
    def create_single_objective(
        cls,
        name: ObjectiveName,
        config: ObjectiveConfig | dict[str, Any] | str | bytes
    ) -> Leaderboard:
        """Create a leaderboard for single-objective optimization.

        Convenience method for creating a leaderboard with a single objective,
        automatically configuring the appropriate scorer.

        Example:
            >>> # Create leaderboard for maximizing accuracy
            >>> leaderboard = Leaderboard.create_single_objective(
            ...     name="accuracy",
            ...     config={
            ...         "direction": "maximize",
            ...         "target": 0.95,    # Target 95% accuracy
            ...         "limit": 0.80,     # At least 80% required
            ...     }
            ... )

        :param name: Name of the objective
        :type name: str
        :param config: Objective configuration as dict, JSON string or bytes
        :type config: dict[str, Any] | str | bytes
        :return: Configured leaderboard
        :rtype: Leaderboard
        """
        obj_config = ObjectiveConfig.parse(config)
        return cls(ObjectiveScorer(objectives={name: obj_config}))

    @classmethod
    def create_dual_objective(
        cls,
        primary_name: ObjectiveName,
        primary_config: ObjectiveConfig | dict[str, Any] | str | bytes,
        secondary_name: ObjectiveName,
        secondary_config: ObjectiveConfig | dict[str, Any] | str | bytes,
        primary_priority: float = 1.0,
        secondary_priority: float = 1.0,
    ) -> Leaderboard:
        """Create a leaderboard for dual-objective optimization.

        Common case where one objective is more important than the other.
        Both objectives are placed in the same comparison group with
        different priorities for weighted ranking.

        Example:
            >>> # Create leaderboard for accuracy vs. training time
            >>> leaderboard = Leaderboard.create_dual_objective(
            ...     primary_name="accuracy",
            ...     primary_config={
            ...         "direction": "maximize",
            ...         "target": 0.95,
            ...         "limit": 0.80
            ...     },
            ...     secondary_name="time",
            ...     secondary_config={
            ...         "direction": "minimize",
            ...         "target": 60.0,
            ...         "limit": 300.0
            ...     }
            ... )

        :param primary_name: Name of primary objective
        :type primary_name: str
        :param primary_config: Primary objective configuration
        :type primary_config: dict[str, Any] | str | bytes
        :param secondary_name: Name of secondary objective
        :type secondary_name: str
        :param secondary_config: Secondary objective configuration
        :type secondary_config: dict[str, Any] | str | bytes
        :param primary_priority: Priority for primary objective (higher is more
            important)
        :type primary_priority: float, optional
        :param secondary_priority: Priority for secondary objective
        :type secondary_priority: float, optional
        :return: Configured leaderboard
        :rtype: Leaderboard
        """
        p_config = ObjectiveConfig.parse(primary_config).with_priority(primary_priority)
        s_config = ObjectiveConfig.parse(secondary_config).with_priority(secondary_priority)

        objectives = {
            primary_name: p_config,
            secondary_name: s_config
        }
        return cls(ObjectiveScorer(objectives=objectives))

    @classmethod
    def create_pareto(
        cls,
        objective_configs: list[tuple[str, dict[str, Any] | str | bytes]],
    ) -> Leaderboard:
        """Create a leaderboard for Pareto optimization.

        Places each objective in its own comparison group for true
        multi-objective optimization using Pareto dominance relationships.

        Example:
            >>> # Create leaderboard for Pareto optimization
            >>> leaderboard = Leaderboard.create_pareto([
            ...     ("accuracy", {
            ...         "direction": "maximize",
            ...         "target": 0.95,
            ...         "limit": 0.80
            ...     }),
            ...     ("time", {
            ...         "direction": "minimize",
            ...         "target": 60.0,
            ...         "limit": 300.0
            ...     }),
            ...     ("memory", {
            ...         "direction": "minimize",
            ...         "target": 1.0,
            ...         "limit": 4.0
            ...     })
            ... ])

        :param objective_configs: List of (name, config) tuples
        :type objective_configs: list[tuple[str, dict[str, Any] | str | bytes]]
        :return: Configured leaderboard
        :rtype: Leaderboard
        :raises ValueError: If objective list is empty
        """
        # TODO: This should be changed after ObjectiveScorer is pydantic-ified
        #   also, .parse needs to be overwritten for typehinting purposes
        configs = {}
        for idx, (name, config) in enumerate(objective_configs):
            obj_config = ObjectiveConfig.parse(config).with_comparison_group(idx)
            configs[name] = obj_config

        return cls(ObjectiveScorer(objectives=configs))

    def __init__(self, objective_scorer: ObjectiveScorer):
        """Initialize leaderboard with objective scorer.

        :param objective_scorer: Scorer for computing scores from objectives
        :type objective_scorer: ObjectiveScorer
        """
        self.objective_scorer = objective_scorer
        self.sorted_population = (
            PartiallyOrderedPopulation[int, Trial]()
            if self.objective_scorer.multigroup
            else TotallyOrderedPopulation[int, Trial]()
        )

    def add(
        self,
        params: dict[ParamName, float],
        objectives: dict[ObjectiveName, float],
    ) -> None:
        """Add new sample to leaderboard.

        Records parameter settings and their achieved objective values, updating
        the sorted population accordingly.

        Example:
            >>> leaderboard.add(
            ...     params={"learning_rate": 0.001},
            ...     objectives={"accuracy": 0.95}
            ... )

        :param params: Parameter values used
        :type params: dict[str, float]
        :param objectives: Resulting objective values
        :type objectives: dict[str, float]
        :raises KeyError: If parameter or objective names don't match configuration
        """
        # TODO: The sorted population should actually just store Trials. We
        #   need to make Trials partially ordered and update the sorted
        #   populations to take generic partially ordered objects

        # Create and store new trial
        trial = Trial(params=params, objectives=objectives)
        index = len(self.trials)
        self.trials.append(trial)

        # Compute score and add to sorted population
        score = self.objective_scorer.score_objectives(objectives)
        self.sorted_population.add(index, score)

    def filter_samples(self, keep_indices: list[int]) -> None:
        """Filter leaderboard to keep only specified sample indices.

        Useful for removing unwanted or invalid samples while maintaining
        proper ordering and scoring.

        :param keep_indices: List of indices to keep
        :type keep_indices: list[int]
        :raises IndexError: If any index is invalid
        """
        # Create new trials list with only kept samples
        self.trials = [self.trials[i] for i in keep_indices]

        # Create new sorted population with remapped indices
        if self.objective_scorer.multigroup:
            new_population = PartiallyOrderedPopulation[int]()
        else:
            new_population = TotallyOrderedPopulation[int]()

        # Add samples to new population with updated indices
        for new_idx, _ in enumerate(keep_indices):
            score = self.objective_scorer.score_objectives(self.trials[new_idx].objectives)
            new_population.add(new_idx, score)

        self.sorted_population = new_population

    def get_best_sample(self) -> tuple[dict[ParamName, float], dict[ObjectiveName, float]]:
        """Get parameters and objectives for best sample.

        :return: (parameters, objectives) for best sample
        :rtype: tuple[dict[str, float], dict[str, float]]
        :raises IndexError: If no samples exist
        """
        if not self.trials:
            raise IndexError("No samples in leaderboard")

        best_idx = self.sorted_population.get_top_samples(1)[0]
        return self.get_sample(best_idx)

    def get_top_samples(
        self, num_samples: int
    ) -> list[tuple[dict[ParamName, float], dict[ObjectiveName, float]]]:
        """Get parameters and objectives for top samples.

        For multi-group objectives, returns samples ordered by Pareto
        dominance and crowding distance.

        :param num_samples: Number of samples to return
        :type num_samples: int
        :return: List of (parameters, objectives) tuples for top samples
        :rtype: list[tuple[dict[str, float], dict[str, float]]]
        """
        top_indices = self.sorted_population.get_top_samples(num_samples)
        return [self.get_sample(i) for i in top_indices]

    def num_samples(self) -> int:
        """Get number of samples in leaderboard.

        :return: Number of samples
        :rtype: int
        """
        return len(self.trials)

    def get_sample(self, index: int) -> tuple[dict[ParamName, float], dict[ObjectiveName, float]]:
        """Get parameters and objectives for specific sample.

        :param index: Sample index
        :type index: int
        :return: (parameters, objectives) for sample
        :rtype: tuple[dict[str, float], dict[str, float]]
        :raises IndexError: If index is invalid
        """
        if not 0 <= index < len(self.trials):
            raise IndexError(f"Sample index {index} out of range")
        trial = self.trials[index]
        return trial.params, trial.objectives

    def get_dataframe(self) -> pd.DataFrame:
        """Get samples as a pandas DataFrame.

        Creates a DataFrame containing all trials with columns for:
        - Parameters (prefixed with "param_")
        - Objectives (prefixed with "objective_")
        - Group scores (prefixed with "group_score_")
        - Ranking information (front index and crowding distance)

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
            >>>
            >>> # Create leaderboard with objective scorer
            >>> leaderboard = Leaderboard(scorer)
            >>>
            >>> # Add optimization results
            >>> leaderboard.add(
            ...     params={"learning_rate": 0.001, "batch_size": 32},
            ...     objectives={"accuracy": 0.95, "training_time": 120}
            ... )
            >>> leaderboard.add(
            ...     params={"learning_rate": 0.01, "batch_size": 16},
            ...     objectives={"accuracy": 0.65, "training_time": 30}
            ... )
            >>> leaderboard.add(
            ...     params={"learning_rate": 0.008, "batch_size": 64},
            ...     objectives={"accuracy": 0.97, "training_time": 250}
            ... )
            >>> leaderboard.add(
            ...     params={"learning_rate": 0.08, "batch_size": 128},
            ...     objectives={"accuracy": 0.81, "training_time": 450}
            ... )
            >>> df = leaderboard.get_dataframe()
            >>> print("Parameter columns:", df.filter(like="param_").columns)
            Parameter columns: Index(['param_learning_rate', 'param_batch_size'], dtype='object')
            >>> print("Top score:", df["group_score_0"].min())
            Top score: 0.25

        :return: DataFrame with all trial information
        :rtype: pandas.DataFrame
        """
        rows = []
        for i, trial in enumerate(self.trials):
            group_scores = self.objective_scorer.score_objectives(trial.objectives)

            row = {
                "index": i,
                **{f"{PARAM_PREFIX}{k}": v for k, v in trial.params.items()},
                **{f"{OBJECTIVE_PREFIX}{k}": v for k, v in trial.objectives.items()},
                **{f"{GROUP_SCORE_PREFIX}{i}": score for i, score in enumerate(group_scores)},
                "front": self.sorted_population.get_front_index(i),
                "crowding_distance": self.sorted_population.get_crowding_distance(i),
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def save(self, filename: Path | str) -> None:
        """Save leaderboard to CSV file.

        Example:
            >>> # Save results for analysis
            >>> leaderboard.save("optimization_results.csv")

        :param filename: Path to save file
        :type filename: Path | str
        """
        self.get_dataframe().to_csv(filename)

    @classmethod
    def load(cls, filename: Path | str, objective_scorer: ObjectiveScorer) -> Leaderboard:
        """Load leaderboard from CSV file.

        Creates a new leaderboard from saved trial data, reconstructing the
        sorted population and trial cache. The provided objective scorer must
        match the configuration used when saving the data.

        Example:
            >>> # Load previous results
            >>> leaderboard = Leaderboard.load(
            ...     "previous_results.csv",
            ...     objective_scorer
            ... )

        :param filename: Path to load file
        :type filename: Path | str
        :param objective_scorer: Scorer for computing scores from objectives
        :type objective_scorer: ObjectiveScorer
        :return: Loaded leaderboard
        :rtype: Leaderboard
        :raises ValueError: If file format is invalid or objectives don't match
            scorer configuration
        """
        df = pd.read_csv(filename, index_col=0)

        # Validate column schema
        param_cols = [c for c in df.columns if c.startswith(PARAM_PREFIX)]
        obj_cols = [c for c in df.columns if c.startswith(OBJECTIVE_PREFIX)]

        if not param_cols:
            raise ValueError(
                f"No parameter columns found (prefix: '{PARAM_PREFIX}')"
            )
        if not obj_cols:
            raise ValueError(
                f"No objective columns found (prefix: '{OBJECTIVE_PREFIX}')"
            )

        # Validate objectives match scorer configuration
        expected_objectives = set(objective_scorer.objective_configs.keys())
        found_objectives = {c[len(OBJECTIVE_PREFIX):] for c in obj_cols}

        if expected_objectives != found_objectives:
            raise ValueError(
                f"Objectives mismatch. Expected: {expected_objectives}, "
                f"Found: {found_objectives}"
            )

        # Create new leaderboard and load samples
        leaderboard = cls(objective_scorer)
        for _, row in df.iterrows():
            params = {
                c[len(PARAM_PREFIX):]: row[c]
                for c in param_cols
            }
            objectives = {
                c[len(OBJECTIVE_PREFIX):]: row[c]
                for c in obj_cols
            }
            leaderboard.add(params, objectives)

        return leaderboard