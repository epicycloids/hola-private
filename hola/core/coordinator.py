import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID

from msgspec import Struct

from hola.core.leaderboard import Leaderboard, Trial
from hola.core.objectives import ObjectiveConfig, ObjectiveName, ObjectiveScorer
from hola.core.parameters import PredefinedParameterConfig, ParameterName, ParameterTransformer
from hola.core.samplers import HypercubeSampler
from hola.messages.worker import Evaluation


class OptimizationState(Struct, frozen=True):
    best_result: Trial | None
    total_evaluations: int


@dataclass
class OptimizationCoordinator:
    hypercube_sampler: HypercubeSampler
    leaderboard: Leaderboard
    parameter_transformer: ParameterTransformer
    top_frac: float = field(default=0.2)

    _active: bool = field(default=True, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
    _current_elite_indices: set[int] = field(default_factory=set, init=False)

    @classmethod
    def from_dict(
        cls,
        hypercube_sampler: HypercubeSampler,
        objectives_dict: dict[ObjectiveName, ObjectiveConfig],
        parameters_dict: dict[ParameterName, PredefinedParameterConfig],
    ) -> "OptimizationCoordinator":
        objective_scorer = ObjectiveScorer.from_dict(objectives_dict)
        leaderboard = Leaderboard(objective_scorer)
        parameter_transformer = ParameterTransformer.from_dict(parameters_dict)
        return cls(
            hypercube_sampler=hypercube_sampler,
            leaderboard=leaderboard,
            parameter_transformer=parameter_transformer,
        )

    def __post_init__(self) -> None:
        if not 0 < self.top_frac <= 1:
            raise ValueError("top_frac must be in (0, 1]")

    async def get_state(self) -> OptimizationState:
        async with self._lock:
            # TODO Discuss what is useful to put here
            return OptimizationState(
                best_result=self._get_best_result(), total_evaluations=self._get_total_evaluations()
            )

    async def get_best_result(self) -> Trial | None:
        async with self._lock:
            return self._get_best_result()

    def _get_best_result(self) -> Trial | None:
        return self.leaderboard.get_best_result()

    async def get_total_evaluations(self) -> int:
        async with self._lock:
            return self._get_total_evaluations()

    def _get_total_evaluations(self) -> int:
        return len(self.leaderboard)

    async def pause(self) -> None:
        async with self._lock:
            self._active = False

    async def resume(self) -> None:
        async with self._lock:
            self._active = True

    async def update_objective_config(
        self, new_config: dict[ObjectiveName, ObjectiveConfig]
    ) -> None:
        async with self._lock:
            new_scorer = ObjectiveScorer(new_config)
            self.leaderboard.update_objective_scorer(new_scorer)
            self._update_elite_samples()

    async def update_parameter_config(self, new_config: dict[str, PredefinedParameterConfig]) -> None:
        async with self._lock:
            # Check for domain expansion
            new_transformer = ParameterTransformer(new_config)
            should_restart_sampler = new_transformer.has_expanded_domain(self.parameter_transformer)

            # Update parameter transformer
            self.parameter_transformer = new_transformer

            # Rebuild leaderboard poset with feasibility checks
            self.leaderboard.rebuild_poset(self.parameter_transformer)

            # Restart sampler if needed
            if should_restart_sampler:
                self.hypercube_sampler.reset()

            # Update elite set and fit sampler
            self._update_elite_samples()

    def is_active(self) -> bool:
        return self._active

    async def suggest_parameters(self, n_samples: int = 1) -> list[dict[ParameterName, Any]] | None:
        async with self._lock:
            if self.is_active():
                normalized_samples = self.hypercube_sampler.sample(n_samples)
                return self.parameter_transformer.unnormalize(normalized_samples)
            else:
                return None

    async def record_evaluation(
        self, evaluation: Evaluation, timestamp: str | None = None, worker_id: UUID | None = None
    ) -> None:
        async with self._lock:
            if not self.is_active():
                return

            # Create trial with feasibility check
            trial = Trial(
                objectives=evaluation.objectives,
                parameters=evaluation.parameters,
                is_feasible=self.parameter_transformer.is_feasible(evaluation.parameters),
            )

            # Add trial to leaderboard
            self.leaderboard.add(trial)

            self._update_elite_samples()

    async def record_failed_evaluation(
        self,
        error: str,
        parameters: dict[ParameterName, Any] | None = None,
        timestamp: str | None = None,
        worker_id: UUID | None = None,
    ) -> None:
        # TODO: Should this adjust the sampler to reassign the parameters?
        pass

    async def save_state(self, filepath: str | Path) -> None:
        async with self._lock:
            # TODO Discuss saving and loading
            raise NotImplementedError

    async def load_state(self, filepath: str | Path) -> None:
        async with self._lock:
            # TODO Discuss saving and loading
            raise NotImplementedError

    def _update_elite_samples(self) -> bool:
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
