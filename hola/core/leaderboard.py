"""
Trial tracking and ranking system for HOLA.

This module implements the leaderboard system that tracks optimization trials
and maintains their partial ordering based on objective comparison groups. Key features:

1. Stores trial information including parameters, objectives, and feasibility
2. Maintains trials in a partially ordered set using non-dominated sorting
3. Supports both single-group and multi-group comparison structures
4. Uses crowding distance within fronts to promote diversity in parameter
   choices
5. Provides methods for retrieving best trials and trial statistics
6. Handles updates to objective scoring and parameter feasibility criteria
"""

from typing import Any

import msgspec
import pandas as pd
from msgspec import Struct

from hola.core.objectives import ObjectiveName, ObjectiveScorer
from hola.core.parameters import ParameterName, ParameterTransformer
from hola.core.poset import ScalarPoset, VectorPoset


class Trial(Struct, frozen=True):
    """
    Immutable record of a single optimization trial.

    Contains all information about a trial including its unique identifier,
    the parameter values used, the objective values achieved, and whether
    the parameter values are considered feasible under current constraints.
    """

    trial_id: int
    """Unique identifier for the trial."""

    objectives: dict[ObjectiveName, float]
    """Dictionary mapping objective names to their achieved values."""

    parameters: dict[ParameterName, Any]
    """Dictionary mapping parameter names to their trial values."""

    is_feasible: bool = True
    """Whether the parameter values satisfy current constraints."""


class Leaderboard:
    """
    Tracks and ranks optimization trials.

    The leaderboard maintains trials in a partially ordered set. For
    single-group optimization, trials are totally ordered. For multi-group
    optimization, trials are organized into fronts using non-dominated sorting
    based on their comparison group scores. Within each front, trials are
    ordered by crowding distance to promote diversity in parameter choices.
    """

    def __init__(self, objective_scorer: ObjectiveScorer):
        """
        Initialize the leaderboard.

        :param objective_scorer: Scorer that defines how to evaluate trials
        :type objective_scorer: ObjectiveScorer
        """
        self._objective_scorer = objective_scorer
        self._poset = (
            VectorPoset[int]() if self._objective_scorer.is_multigroup else ScalarPoset[int]()
        )
        self._data: dict[int, Trial] = {}

    def __len__(self) -> int:
        """
        :return: Number of trials in the leaderboard's partial ordering
        :rtype: int
        """
        return len(self._poset)

    def get_trial(self, trial_id: int) -> Trial:
        """
        Retrieve a specific trial by ID.

        :param trial_id: ID of the trial to retrieve
        :type trial_id: int
        :return: The requested trial
        :rtype: Trial
        :raises KeyError: If trial_id doesn't exist
        """
        return self._data[trial_id]

    def get_best_trial(self) -> Trial | None:
        """
        Get the trial with the best objective scores.

        For multi-group optimization, returns a trial from the first
        non-dominated front, selected based on crowding distance to
        promote diversity.

        :return: Best trial, or None if leaderboard is empty
        :rtype: Trial | None
        """
        if len(self) == 0:
            return None
        best_index, _ = self._poset.peek(1)[0]
        return self._data[best_index]

    def get_top_k(self, k: int = 1) -> list[Trial]:
        """
        Get the k best trials.

        Returns k trials ordered by:
        1. Non-dominated front membership
        2. Crowding distance within each front (to promote diversity)

        :param k: Number of trials to return
        :type k: int
        :return: List of up to k best trials
        :rtype: list[Trial]
        """
        return [self._data[idx] for idx, _ in self._poset.peek(k)]

    def get_dataframe(self) -> pd.DataFrame:
        """
        Convert leaderboard to a DataFrame of feasible trials.

        Creates a DataFrame containing information about feasible trials,
        ordered by non-dominated sorting and crowding distance. The
        DataFrame includes:
        - Trial IDs
        - Parameter values
        - Objective values
        - Comparison group scores
        - Crowding distance (provides diversity measure within a Pareto front)

        Note:
        - Only feasible trials are included in the DataFrame
        - Trials are ordered first by their non-dominated front membership
        - Within each front, trials are ordered by crowding distance to
          promote diversity in parameter choices
        - The ordering ensures best trials (according to comparison group
          scores) appear in the top rows

        :return: DataFrame containing feasible trial information, sorted by ranking
        :rtype: pd.DataFrame
        """
        info = {}
        for i, (key, score) in enumerate(self._poset.items()):
            trial = self.get_trial(key)

            # Get crowding distance from the poset
            crowding_distance = self._poset.get_crowding_distance(key)

            # Handle score properly based on its type
            score_dict = {}
            if self._objective_scorer.is_multigroup:
                score_dict = {f"Group {j} Score": score[j] for j in range(len(score))}
            else:
                # If score is a single value (e.g., float)
                score_dict = {"Group Score": score}

            info[i] = {
                "Trial": key,
                "Crowding Distance": crowding_distance,
                **trial.parameters,
                **trial.objectives,
                **score_dict,
            }
        return pd.DataFrame(info)

    def add(self, trial: Trial) -> None:
        """
        Add a new trial to the leaderboard.

        If the trial is feasible, it is also added to the partial ordering
        based on its objective scores.

        :param trial: Trial to add
        :type trial: Trial
        """
        index = trial.trial_id
        self._data[index] = trial
        if trial.is_feasible:
            group_values = self._objective_scorer.score(trial.objectives)
            if all(group_values < float("inf")):
                self._poset.add(index, group_values)
