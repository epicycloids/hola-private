from typing import Any

import msgspec
from msgspec import Struct

from hola.core.objectives import ObjectiveName, ObjectiveScorer
from hola.core.parameters import ParameterName, ParameterTransformer
from hola.core.poset import ScalarPoset, VectorPoset


class Trial(Struct, frozen=True):
    objectives: dict[ObjectiveName, float]
    parameters: dict[ParameterName, Any]
    is_feasible: bool = True


class Leaderboard:
    def __init__(self, objective_scorer: ObjectiveScorer):
        self._objective_scorer = objective_scorer
        self._poset = (
            VectorPoset[int]() if self._objective_scorer.is_multigroup else ScalarPoset[int]()
        )
        self._data: dict[int, Trial] = {}

    def __len__(self) -> int:
        return len(self._poset)

    def get_best_result(self) -> Trial | None:
        if len(self) == 0:
            return None
        best_index = self._poset.peek(1)[0][0]
        return self._data[best_index]

    def get_top_k(self, k: int = 1) -> list[Trial]:
        return [self._data[idx] for idx, _ in self._poset.peek(k)]

    def add(self, trial: Trial) -> None:
        index = len(self._data)
        self._data[index] = trial
        if trial.is_feasible:
            group_values = self._objective_scorer.score(trial.objectives)
            self._poset.add(index, group_values)

    def update_objective_scorer(self, new_scorer: ObjectiveScorer) -> None:
        old_trials = list(self._data.values())
        self._data.clear()
        self._objective_scorer = new_scorer
        self._poset = (
            VectorPoset[int]() if self._objective_scorer.is_multigroup else ScalarPoset[int]()
        )
        for trial in old_trials:
            self.add(trial)

    def rebuild_poset(self, param_transformer: ParameterTransformer) -> None:
        """Rebuild poset with only feasible trials under current parameter configs."""
        # Update feasibility for all trials
        self._data = {
            idx: msgspec.structs.replace(
                trial, is_feasible=param_transformer.is_feasible(trial.parameters)
            )
            for idx, trial in self._data.items()
        }

        # Create new poset of appropriate type
        self._poset = (
            VectorPoset[int]() if self._objective_scorer.is_multigroup else ScalarPoset[int]()
        )

        # Add only feasible trials to poset
        for idx, trial in self._data.items():
            if trial.is_feasible:
                group_values = self._objective_scorer.score(trial.objectives)
                self._poset.add(idx, group_values)
