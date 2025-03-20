"""
Optimization coordination for HOLA.

This module implements the core coordination logic for the optimization process,
managing the interaction between:
- Parameter sampling strategies
- Trial evaluation and recording
- Leaderboard maintenance
- Adaptive sampling based on elite trials

The coordinator provides access to the optimization state and handles dynamic
updates to parameter and objective configurations. It uses adaptive sampling
strategies that learn from the best-performing trials to focus exploration in
promising regions of the parameter space.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from hola.core.leaderboard import Leaderboard, Trial
from hola.core.objectives import ObjectiveName, ObjectiveScorer
from hola.core.parameters import ParameterName, ParameterTransformer
from hola.core.samplers import HypercubeSampler


@dataclass
class OptimizationCoordinator:
    """
    Coordinates the hyperparameter optimization process.

    This class manages the optimization workflow, including:
    - Generating parameter suggestions using adaptive sampling
    - Recording trial evaluations
    - Maintaining the leaderboard of trials
    - Updating sampling strategies based on elite trials
    - Handling configuration changes during optimization
    """

    hypercube_sampler: HypercubeSampler
    """Sampler for generating normalized parameter values."""

    leaderboard: Leaderboard
    """Tracks and ranks all trials."""

    parameter_transformer: ParameterTransformer
    """Handles conversion between normalized and native parameter spaces."""

    top_frac: float = field(default=0.2)
    """Fraction of best trials to use for adaptive sampling."""

    minimum_fit_samples: int = field(default=20)
    """Minimum number of elite samples needed to fit exploit sampler."""

    _current_elite_indices: set[int] = field(default_factory=set, init=False)
    """Set of indices for current elite trials."""

    @classmethod
    def from_dict(
        cls,
        hypercube_sampler: HypercubeSampler,
        objectives_dict: dict[ObjectiveName, dict[str, Any]],
        parameters_dict: dict[ParameterName, dict[str, Any]],
        top_frac: float = 0.2,
        minimum_fit_samples: int = 20,
    ) -> "OptimizationCoordinator":
        """
        Create coordinator from configuration dictionaries.

        :param hypercube_sampler: Configured sampler instance
        :type hypercube_sampler: HypercubeSampler
        :param objectives_dict: Objective configuration dictionary
        :type objectives_dict: dict[ObjectiveName, dict[str, Any]]
        :param parameters_dict: Parameter configuration dictionary
        :type parameters_dict: dict[ParameterName, dict[str, Any]]
        :param top_frac: Fraction of best trials to use for adaptive sampling
        :type top_frac: float
        :param minimum_fit_samples: Minimum number of elite samples needed to fit exploit sampler
        :type minimum_fit_samples: int
        :return: Configured optimization coordinator
        :rtype: OptimizationCoordinator
        """
        objective_scorer = ObjectiveScorer.from_dict(objectives_dict)
        leaderboard = Leaderboard(objective_scorer)
        parameter_transformer = ParameterTransformer.from_dict(parameters_dict)
        return cls(
            hypercube_sampler=hypercube_sampler,
            leaderboard=leaderboard,
            parameter_transformer=parameter_transformer,
            top_frac=top_frac,
            minimum_fit_samples=minimum_fit_samples,
        )

    def __post_init__(self) -> None:
        """
        Validate coordinator configuration.

        :raises ValueError: If top_frac is not in (0, 1]
        """
        if not 0 < self.top_frac <= 1:
            raise ValueError("top_frac must be in (0, 1]")

        if self.minimum_fit_samples <= 0:
            raise ValueError("minimum_fit_samples must be positive")

    def suggest_parameters(self, n_samples: int = 1) -> Tuple[List[Dict[ParameterName, Any]], Dict[str, Any]]:
        """
        Generate parameter suggestions for new trials.

        :param n_samples: Number of parameter sets to generate
        :type n_samples: int
        :return: Tuple containing:
            - List of parameter dictionaries
            - Metadata about the generated samples
        :rtype: Tuple[List[Dict[ParameterName, Any]], Dict[str, Any]]
        """
        normalized_samples, metadata = self.hypercube_sampler.sample(n_samples)
        param_dicts = self.parameter_transformer.unnormalize(normalized_samples)
        return param_dicts, metadata

    def record_evaluation(
        self, parameters: dict[ParameterName, Any], objectives: dict[ObjectiveName, float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Trial | None:
        """
        Record a completed trial evaluation.

        This method:
        - Creates a trial from the evaluation
        - Adds it to the leaderboard
        - Updates elite samples and adaptive sampling
        - Returns the best trial

        Note:
        - Trial metadata is stored separately from the main trial data
        - Access metadata using the leaderboard's get_metadata() method
        - The metadata provides insights into the sampling strategy used

        :param parameters: Parameter values
        :type parameters: dict[ParameterName, Any]
        :param objectives: Objective values
        :type objectives: dict[ObjectiveName, float]
        :param metadata: Optional metadata about the trial
        :type metadata: Optional[Dict[str, Any]]
        :return: Best trial
        :rtype: Trial | None
        """
        # Create trial with feasibility check
        feasible = self.parameter_transformer.is_feasible(parameters)
        trial = Trial(
            trial_id=self.leaderboard.get_total_count(),
            objectives=objectives,
            parameters=parameters,
            is_feasible=feasible,
            metadata=metadata or {},
        )

        # Add trial to leaderboard
        self.leaderboard.add(trial)

        # Fit sampler if enough samples are available
        if feasible and self.top_frac * self.leaderboard.get_feasible_count() >= self.minimum_fit_samples:
            self.hypercube_sampler.fit(
                self.parameter_transformer.normalize(
                    [trial.parameters for trial in self.leaderboard.get_top_k(self.minimum_fit_samples)]
                )
            )

        # Get the new best objectives
        return self.get_best_trial()

    def get_total_evaluations(self) -> int:
        """
        Get the total number of evaluations processed.

        :return: Total number of trials evaluated
        :rtype: int
        """
        return self.leaderboard.get_total_count()

    def get_best_trial(self) -> Trial | None:
        """
        Get the best trial from the leaderboard.

        :return: Best trial
        :rtype: Trial | None
        """
        return self.leaderboard.get_best_trial()

    def get_top_k_trials(self, k: int = 1) -> list[Trial]:
        """
        Get the k best trials from the optimization process.

        Returns k trials ordered by:
        1. Non-dominated front membership
        2. Crowding distance within each front (to promote diversity)

        :param k: Number of trials to return
        :type k: int
        :return: List of up to k best trials
        :rtype: list[Trial]
        """
        return self.leaderboard.get_top_k(k)

    def get_top_k_fronts(self, k: int = 1) -> list[list[Trial]]:
        """
        Get the top k Pareto fronts of trials.

        In multi-group optimization, trials are organized into Pareto fronts based on
        dominance relationships between comparison groups. The first front contains
        non-dominated trials, the second front contains trials dominated only by those
        in the first front, and so on.

        Note: This method is primarily useful when multiple comparison groups are used.
        For single-group optimization, each front will contain trials with identical scores.

        :param k: Number of fronts to return
        :type k: int
        :return: List of up to k fronts, each containing a list of Trial objects
        :rtype: list[list[Trial]]
        """
        return self.leaderboard.get_top_k_fronts(k)

    def get_trials_dataframe(self, ranked_only: bool = True) -> pd.DataFrame:
        """
        Get a DataFrame of trials from the optimization process.

        :param ranked_only: If True, only include ranked trials (feasible with finite scores).
                           If False, include all trials with status columns.
        :type ranked_only: bool
        :return: DataFrame containing trial information
        :rtype: pd.DataFrame
        """
        return self.leaderboard.get_dataframe(ranked_only)

    def get_all_trials_dataframe(self) -> pd.DataFrame:
        """
        Get a DataFrame of all trials from the optimization process.

        This is a convenience wrapper around get_trials_dataframe(ranked_only=False).

        :return: DataFrame containing all trial information
        :rtype: pd.DataFrame
        """
        return self.leaderboard.get_all_trials_dataframe()

    def get_trials_metadata(self, trial_ids: Optional[Union[int, List[int]]] = None) -> pd.DataFrame:
        """
        Get metadata for one or more trials as a DataFrame.

        :param trial_ids: Specific trial ID(s) to retrieve metadata for, or None for all trials
        :type trial_ids: Optional[Union[int, List[int]]]
        :return: DataFrame with trial IDs as index and metadata as columns
        :rtype: pd.DataFrame
        """
        return self.leaderboard.get_metadata(trial_ids)

    def get_feasible_count(self) -> int:
        """
        Get the count of all feasible trials, including those with infinite scores.

        :return: Number of feasible trials
        :rtype: int
        """
        return self.leaderboard.get_feasible_count()

    def get_feasible_infinite_count(self) -> int:
        """
        Get the count of feasible trials with at least one infinite objective score.

        :return: Number of feasible trials with infinite scores
        :rtype: int
        """
        return self.leaderboard.get_feasible_infinite_count()

    def get_infeasible_count(self) -> int:
        """
        Get the count of infeasible trials.

        :return: Number of infeasible trials
        :rtype: int
        """
        return self.leaderboard.get_infeasible_count()

    def get_ranked_count(self) -> int:
        """
        Get the number of ranked trials (feasible with all finite scores).

        :return: Number of ranked trials
        :rtype: int
        """
        return self.leaderboard.get_ranked_count()

    def is_multi_group(self) -> bool:
        """
        Check if this is a multi-objective optimization using multiple comparison groups.

        With multiple comparison groups, Pareto dominance sorting is used
        to rank trials. With a single comparison group, trials are simply ranked by
        their scalar score.

        :return: True if using multiple objective comparison groups, False if using a single group
        :rtype: bool
        """
        return self.leaderboard._objective_scorer.is_multigroup


if __name__ == "__main__":
    from hola.core.coordinator import OptimizationCoordinator
    from hola.core.samplers import ExploreExploitSampler, SobolSampler, ClippedGaussianMixtureSampler

    # Define your parameters
    parameters = {
        "x": {"type": "continuous", "min": 0.0, "max": 10.0},
        "y": {"type": "continuous", "min": 0.0, "max": 10.0},
    }

    # Define your objectives
    objectives = {
        "f1": {
            "target": 0.0,
            "limit": 100.0,
            "direction": "minimize",
            "priority": 1.0,
            "comparison_group": 0
        },
        "f2": {
            "target": 0.0,
            "limit": 100.0,
            "direction": "minimize",
            "priority": 0.8,
            "comparison_group": 0
        },
        "f3": {
            "target": 0.0,
            "limit": 100.0,
            "direction": "minimize",
            "priority": 0.5,
            "comparison_group": 1
        },
    }

    # Create samplers for exploration and exploitation
    explore_sampler = SobolSampler(dimension=2)
    exploit_sampler = ClippedGaussianMixtureSampler(dimension=2, n_components=2)

    # Create an explore-exploit sampler (combines exploration and exploitation)
    sampler = ExploreExploitSampler(
        explore_sampler=explore_sampler,
        exploit_sampler=exploit_sampler,
    )

    # Create coordinator
    coordinator = OptimizationCoordinator.from_dict(
        hypercube_sampler=sampler,
        objectives_dict=objectives,
        parameters_dict=parameters
    )

    # Define your evaluation function
    def evaluate(x: float, y: float) -> dict[str, float]:
        f1 = x**2 + y**2
        f2 = (x-2)**2 + (y-2)**2
        f3 = (x-4)**2 + (y-4)**2
        return {"f1": f1, "f2": f2, "f3": f3}

    # Track feasible and infeasible trials
    feasible_count = 0
    infeasible_count = 0
    infinite_scores_count = 0

    for i in range(100):  # Running 100 iterations
        params_list, metadata = coordinator.suggest_parameters()
        print(f"Iteration {i+1}: Using sampler: {metadata.get('sampler_class')} in phase: {metadata.get('phase', 'n/a')}")

        for params in params_list:
            objectives = evaluate(**params)

            # Check parameter feasibility
            feasible = coordinator.parameter_transformer.is_feasible(params)

            # Check objective scores
            scores = coordinator.leaderboard._objective_scorer.score(objectives)
            has_infinite_scores = any(score == float("inf") for score in scores)

            if feasible:
                feasible_count += 1
                if has_infinite_scores:
                    infinite_scores_count += 1
                    print(f"  Trial has infinite objective scores: {objectives}")
            else:
                infeasible_count += 1
                print(f"  Trial has infeasible parameters: {params}")

            coordinator.record_evaluation(params, objectives, metadata)

        if (i+1) % 10 == 0:
            print(f"Progress: {i+1}/100 iterations")
            print(f"  Feasible trials: {feasible_count}")
            print(f"  - With finite scores: {feasible_count - infinite_scores_count}")
            print(f"  - With infinite scores: {infinite_scores_count}")
            print(f"  Infeasible trials: {infeasible_count}")
            print(f"  Total trials: {feasible_count + infeasible_count}")
            print("-"*80)

    # Print final statistics
    print("\nFinal Statistics:")
    print(f"Total trials processed: {feasible_count + infeasible_count}")
    print(f"Feasible trials: {feasible_count}")
    print(f"- With finite scores: {feasible_count - infinite_scores_count}")
    print(f"- With infinite scores: {infinite_scores_count}")
    print(f"Infeasible trials: {infeasible_count}")

    # Count trials in different data structures
    print(f"Trials in leaderboard._data: {len(coordinator.leaderboard._data)}")
    print(f"Trials in leaderboard._poset (ranked trials): {coordinator.get_ranked_count()}")
    print(f"Total trials: {coordinator.get_total_evaluations()}")
    print(f"Feasible trials: {coordinator.get_feasible_count()}")
    print(f"Feasible with infinite scores: {coordinator.get_feasible_infinite_count()}")
    print(f"Infeasible trials: {coordinator.get_infeasible_count()}")

    # Get and display ranked trial information
    trial_df = coordinator.get_trials_dataframe()
    print(f"\nRanked Trials DataFrame has {len(trial_df)} rows")
    print(trial_df.head())

    # Get and display ALL trials including those with infinite scores
    all_trials_df = coordinator.get_all_trials_dataframe()
    print(f"\nAll Trials DataFrame has {len(all_trials_df)} rows")
    print(f"- Ranked trials: {len(all_trials_df[all_trials_df['Is Ranked']])}")
    print(f"- Unranked trials: {len(all_trials_df[~all_trials_df['Is Ranked']])}")
    print(all_trials_df.head())

    # Get and display metadata separately
    metadata_df = coordinator.get_trials_metadata()
    print(f"\nMetadata DataFrame has {len(metadata_df)} rows")

    # Verify metadata is present for both ranked and unranked trials
    ranked_ids = set(trial_df['Trial'])
    ranked_metadata = metadata_df[metadata_df.index.isin(ranked_ids)]
    unranked_metadata = metadata_df[~metadata_df.index.isin(ranked_ids)]
    print(f"- Metadata entries for ranked trials: {len(ranked_metadata)}")
    print(f"- Metadata entries for unranked trials: {len(unranked_metadata)}")

    print("\nSample metadata for unranked trial with infinite scores:")
    if len(unranked_metadata) > 0:
        print(unranked_metadata.head(1))

    # Display top 3 trials
    top_trials = coordinator.get_top_k_trials(3)
    print("\nTop 3 Trials:")
    for i, trial in enumerate(top_trials):
        print(f"{i+1}. Trial {trial.trial_id}: {trial.objectives}")

    # Display top 2 Pareto fronts (for multi-group optimization)
    if coordinator.is_multi_group():
        top_fronts = coordinator.get_top_k_fronts(2)
        print("\nTop 2 Pareto Fronts:")
        for i, front in enumerate(top_fronts):
            print(f"Front {i+1} with {len(front)} trials:")
            for trial in front:
                print(f"  Trial {trial.trial_id}: {trial.objectives}")
