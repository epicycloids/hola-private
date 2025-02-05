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

import asyncio
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from msgspec import Struct

from hola.core.leaderboard import Leaderboard, Trial
from hola.core.objectives import ObjectiveName, ObjectiveScorer
from hola.core.parameters import ParameterName, ParameterTransformer
from hola.core.samplers import HypercubeSampler
from hola.messages.worker import Evaluation


class OptimizationState(Struct, frozen=True):
    """
    Immutable snapshot of the current optimization state.

    Used to provide status updates about the optimization progress.
    """

    best_result: Trial | None
    """Best trial found so far, or None if no feasible trials exist."""

    total_evaluations: int
    """Total number of trials evaluated."""


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

    _active: bool = field(default=True, init=False)
    """Whether the optimizer is currently active."""

    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
    """Lock for thread-safe access to optimization state."""

    _current_elite_indices: set[int] = field(default_factory=set, init=False)
    """Set of indices for current elite trials."""

    @classmethod
    def from_dict(
        cls,
        hypercube_sampler: HypercubeSampler,
        objectives_dict: dict[ObjectiveName, dict[str, Any]],
        parameters_dict: dict[ParameterName, dict[str, Any]],
    ) -> "OptimizationCoordinator":
        """
        Create coordinator from configuration dictionaries.

        :param hypercube_sampler: Configured sampler instance
        :type hypercube_sampler: HypercubeSampler
        :param objectives_dict: Objective configuration dictionary
        :type objectives_dict: dict[ObjectiveName, dict[str, Any]]
        :param parameters_dict: Parameter configuration dictionary
        :type parameters_dict: dict[ParameterName, dict[str, Any]]
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
        )

    def __post_init__(self) -> None:
        """
        Validate coordinator configuration.

        :raises ValueError: If top_frac is not in (0, 1]
        """
        if not 0 < self.top_frac <= 1:
            raise ValueError("top_frac must be in (0, 1]")

    async def get_state(self) -> OptimizationState:
        """
        Get current optimization state.

        :return: Snapshot of current state
        :rtype: OptimizationState
        """
        async with self._lock:
            return OptimizationState(
                best_result=self._get_best_trial(), total_evaluations=self._get_total_evaluations()
            )

    async def get_best_trial(self) -> Trial | None:
        """
        Get the best trial found so far.

        :return: Best trial, or None if no feasible trials exist
        :rtype: Trial | None
        """
        async with self._lock:
            return self._get_best_trial()

    async def get_total_evaluations(self) -> int:
        """
        Get total number of trials evaluated.

        :return: Number of trials
        :rtype: int
        """
        async with self._lock:
            return self._get_total_evaluations()

    async def pause(self) -> None:
        """Pause optimization process."""
        async with self._lock:
            self._active = False

    async def resume(self) -> None:
        """Resume optimization process."""
        async with self._lock:
            self._active = True

    async def update_objective_config(
        self, new_config: dict[ObjectiveName, dict[str, Any]]
    ) -> None:
        """
        Update objective configuration.

        This method:
        1. Creates new objective scorer from config
        2. Updates leaderboard scoring
        3. Updates elite samples and adaptive sampling

        :param new_config: New objective configuration
        :type new_config: dict[ObjectiveName, dict[str, Any]]
        """
        async with self._lock:
            new_scorer = ObjectiveScorer.from_dict(new_config)
            self.leaderboard.update_objective_scorer(new_scorer)
            self._update_elite_samples()

    async def update_parameter_config(self, new_config: dict[str, dict[str, Any]]) -> None:
        """
        Update parameter configuration.

        This method:
        1. Checks for domain expansion
        2. Updates parameter transformer
        3. Rebuilds leaderboard with new feasibility checks
        4. Resets sampler if domain expanded
        5. Updates elite samples and adaptive sampling

        :param new_config: New parameter configuration
        :type new_config: dict[str, dict[str, Any]]
        """
        async with self._lock:
            # Check for domain expansion
            new_transformer = ParameterTransformer.from_dict(new_config)
            should_restart_sampler = new_transformer.has_expanded_domain(self.parameter_transformer)

            # Update parameter transformer
            self.parameter_transformer = new_transformer

            # Rebuild leaderboard poset with feasibility checks
            self.leaderboard.rebuild_leaderboard(self.parameter_transformer)

            # Restart sampler if needed
            if should_restart_sampler:
                self.hypercube_sampler.reset()

            # Update elite set and fit sampler
            self._update_elite_samples()

    def is_active(self) -> bool:
        """
        Check if optimization is active.

        :return: True if optimization is active
        :rtype: bool
        """
        return self._active

    async def suggest_parameters(self, n_samples: int = 1) -> list[dict[ParameterName, Any]] | None:
        """
        Generate parameter suggestions for new trials.

        Returns None if optimization is paused.

        :param n_samples: Number of parameter sets to generate
        :type n_samples: int
        :return: List of parameter dictionaries, or None if paused
        :rtype: list[dict[ParameterName, Any]] | None
        """
        async with self._lock:
            if self.is_active():
                normalized_samples = self.hypercube_sampler.sample(n_samples)
                return self.parameter_transformer.unnormalize(normalized_samples)
            else:
                return None

    async def record_evaluation(
        self, evaluation: Evaluation, timestamp: str | None = None, worker_id: UUID | None = None
    ) -> None:
        """
        Record a completed trial evaluation.

        This method:
        1. Creates a trial from the evaluation
        2. Adds it to the leaderboard
        3. Updates elite samples and adaptive sampling

        :param evaluation: Evaluation results
        :type evaluation: Evaluation
        :param timestamp: Optional timestamp for the evaluation
        :type timestamp: str | None
        :param worker_id: Optional ID of worker that performed evaluation
        :type worker_id: UUID | None
        """
        async with self._lock:
            if not self.is_active():
                return

            # Create trial with feasibility check
            trial = Trial(
                trial_id=len(self.leaderboard),
                objectives=evaluation.objectives,
                parameters=evaluation.parameters,
                is_feasible=self.parameter_transformer.is_feasible(evaluation.parameters),
            )

            # Add trial to leaderboard
            self.leaderboard.add(trial)

            self._update_elite_samples()

    def _update_elite_samples(self) -> bool:
        """
        Update the set of elite samples and fit the sampler.

        Elite samples are the top fraction (top_frac) of feasible trials,
        used to guide the adaptive sampling strategy.

        :return: True if elite set changed, False otherwise
        :rtype: bool
        """
        feasible_count = sum(1 for t in self.leaderboard._data.values() if t.is_feasible)
        if not feasible_count:
            if self._current_elite_indices:
                self._current_elite_indices.clear()
                return True
            return False

        # Calculate number of elite samples
        n_elite = max(1, int(feasible_count * self.top_frac))

        # Get top n_elite trials
        elite_trials = self.leaderboard.get_top_k(n_elite)
        new_elite_indices = {
            idx for idx, t in enumerate(self.leaderboard._data) if t in elite_trials
        }

        # Check if elite set changed
        if new_elite_indices != self._current_elite_indices:
            self._current_elite_indices = new_elite_indices

            # Convert elite trials to normalized parameters and fit
            elite_params = self.parameter_transformer.normalize(
                [trial.parameters for trial in elite_trials]
            )
            self.hypercube_sampler.fit(elite_params)
            return True

        return False

    def _get_best_trial(self) -> Trial | None:
        """
        Helper method to get best trial from leaderboard.

        :return: Best trial, or None if no feasible trials exist
        :rtype: Trial | None
        """
        return self.leaderboard.get_best_trial()

    def _get_total_evaluations(self) -> int:
        """
        Helper method to get total number of trials from leaderboard.

        Note: This returns the number of feasible trials in the leaderboard's
        partial ordering, not the total number of evaluations attempted.

        :return: Number of feasible trials
        :rtype: int
        """
        return len(self.leaderboard)
