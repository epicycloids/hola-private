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
from typing import Any


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

    def suggest_parameters(self, n_samples: int = 1) -> list[dict[ParameterName, Any]]:
        """
        Generate parameter suggestions for new trials.

        :param n_samples: Number of parameter sets to generate
        :type n_samples: int
        :return: List of parameter dictionaries
        :rtype: list[dict[ParameterName, Any]]
        """
        normalized_samples = self.hypercube_sampler.sample(n_samples)
        return self.parameter_transformer.unnormalize(normalized_samples)

    def record_evaluation(
        self, parameters: dict[ParameterName, Any], objectives: dict[ObjectiveName, float]
    ) -> Trial | None:
        """
        Record a completed trial evaluation.

        This method:
        - Creates a trial from the evaluation
        - Adds it to the leaderboard
        - Updates elite samples and adaptive sampling
        - Returns the best trial

        :param parameters: Parameter values
        :type parameters: dict[ParameterName, Any]
        :param objectives: Objective values
        :type objectives: dict[ObjectiveName, float]
        :return: Best trial
        :rtype: Trial | None
        """
        # Create trial with feasibility check
        feasible = self.parameter_transformer.is_feasible(parameters)
        trial = Trial(
            trial_id=len(self.leaderboard),
            objectives=objectives,
            parameters=parameters,
            is_feasible=feasible,
        )

        # Add trial to leaderboard
        self.leaderboard.add(trial)

        # Fit sampler if enough samples are available
        if feasible and self.top_frac * len(self.leaderboard) >= self.minimum_fit_samples:
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
        return len(self.leaderboard)

    def get_best_trial(self) -> Trial | None:
        """
        Get the best trial from the leaderboard.

        :return: Best trial
        :rtype: Trial | None
        """
        return self.leaderboard.get_best_trial()


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

    for i in range(1000):
        params_list = coordinator.suggest_parameters()
        print(params_list)
        for params in params_list:
            objectives = evaluate(**params)
            coordinator.record_evaluation(params, objectives)
        print(coordinator.get_best_trial())
        print(coordinator.get_total_evaluations())
        print("-"*100)

    print(coordinator.leaderboard.get_dataframe())
