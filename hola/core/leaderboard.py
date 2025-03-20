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

Trials can be:
- Feasible with all finite scores (ranked in the poset)
- Feasible with at least one infinite score (stored but not ranked)
- Infeasible (stored but not included in ranking)
"""

from typing import Any, Dict, List, Optional, Union

import pandas as pd
from msgspec import Struct

from hola.core.objectives import ObjectiveName, ObjectiveScorer
from hola.core.parameters import ParameterName
from hola.core.poset import ScalarPoset, VectorPoset


class Trial(Struct, frozen=True):
    """
    Immutable record of a single optimization trial.

    Contains all information about a trial including its unique identifier,
    the parameter values used, the objective values achieved, whether
    the parameter values are considered feasible under current constraints,
    and any associated metadata.
    """

    trial_id: int
    """Unique identifier for the trial."""

    objectives: dict[ObjectiveName, float]
    """Dictionary mapping objective names to their achieved values."""

    parameters: dict[ParameterName, Any]
    """Dictionary mapping parameter names to their trial values."""

    is_feasible: bool = True
    """Whether the parameter values satisfy current constraints."""

    metadata: Dict[str, Any] = {}
    """Additional metadata about the trial, such as sampler information."""


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

    def get_feasible_count(self) -> int:
        """
        Get the count of all feasible trials, including those with infinite scores.

        This includes both:
        - Feasible trials with all finite scores (ranked in poset)
        - Feasible trials with at least one infinite score (not ranked)

        :return: Number of feasible trials in the leaderboard
        :rtype: int
        """
        return sum(1 for trial in self._data.values() if trial.is_feasible)

    def get_feasible_infinite_count(self) -> int:
        """
        Get the count of feasible trials with at least one infinite score.

        These trials are stored in the leaderboard but not ranked in the poset
        due to having at least one infinite objective score.

        :return: Number of feasible trials with infinite scores
        :rtype: int
        """
        return sum(
            1 for tid, trial in self._data.items()
            if trial.is_feasible and tid not in self._poset.get_indices()
        )

    def get_infeasible_count(self) -> int:
        """
        Get the count of infeasible trials.

        These are trials where parameter values violated feasibility constraints.

        :return: Number of infeasible trials
        :rtype: int
        """
        return sum(1 for trial in self._data.values() if not trial.is_feasible)

    def get_total_count(self) -> int:
        """
        Get the total count of all trials in the leaderboard.

        This includes:
        - Feasible trials with all finite scores (ranked in poset)
        - Feasible trials with at least one infinite score (not ranked)
        - Infeasible trials

        :return: Total number of trials stored in the leaderboard
        :rtype: int
        """
        return len(self._data)

    def get_ranked_count(self) -> int:
        """
        Get the number of trials in the leaderboard's partial ordering.

        This only counts feasible trials with all finite scores that are
        included in the partial ordering (poset).

        :return: Number of ranked trials
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
        if len(self._poset) == 0:
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

    def get_top_k_fronts(self, k: int = 1) -> list[list[Trial]]:
        """
        Get the top k Pareto fronts of trials.

        In multi-objective optimization with multiple comparison groups, trials are
        organized into Pareto fronts based on dominance relationships between group scores.
        The first front contains non-dominated trials, the second front contains
        trials dominated only by those in the first front, and so on.

        This method returns complete fronts (not just individual trials), preserving
        the Pareto dominance relationships. Within each front, trials are ordered
        by crowding distance to promote diversity.

        :param k: Number of fronts to return
        :type k: int
        :return: List of up to k fronts, each containing a list of Trial objects
        :rtype: list[list[Trial]]
        :raises ValueError: If k < 1
        """
        if k < 1:
            raise ValueError("k must be positive.")

        result = []
        # Take the first k fronts from the poset
        for i, front in enumerate(self._poset.fronts()):
            if i >= k:
                break

            # Convert each front from (id, score) tuples to Trial objects
            trial_front = [self._data[trial_id] for trial_id, _ in front]
            result.append(trial_front)

        return result

    def get_dataframe(self, ranked_only: bool = True) -> pd.DataFrame:
        """
        Convert leaderboard trials to a DataFrame.

        Creates a DataFrame containing information about trials, including:
        - Trial IDs
        - Parameter values
        - Objective values
        - Comparison group scores (for ranked trials)
        - Crowding distance (for ranked trials)

        :param ranked_only: If True, only include ranked trials (feasible with finite scores).
                           If False, include all trials with status columns.
        :type ranked_only: bool
        :return: DataFrame containing trial information
        :rtype: pd.DataFrame
        """
        data_rows = []

        # Determine which trials to process
        if ranked_only:
            trial_ids = [key for key, _ in self._poset.items()]
        else:
            trial_ids = list(self._data.keys())

        # Process each trial
        for tid in trial_ids:
            trial = self._data[tid]
            is_ranked = tid in self._poset.get_indices()

            # Skip unranked trials if ranked_only is True
            if ranked_only and not is_ranked:
                continue

            # Basic information for all trials
            row_data = {
                "Trial": tid,
                **{str(k): v for k, v in trial.parameters.items()},
                **{str(k): v for k, v in trial.objectives.items()},
            }

            # Add status columns for all trials mode
            if not ranked_only:
                row_data["Is Ranked"] = is_ranked
                row_data["Is Feasible"] = trial.is_feasible

            # Add score information for ranked trials
            if is_ranked:
                score = self._poset[tid]
                crowding_distance = self._poset.get_crowding_distance(tid)
                row_data["Crowding Distance"] = crowding_distance

                if self._objective_scorer.is_multigroup:
                    for j in range(len(score)):
                        row_data[f"Group {j} Score"] = score[j]
                else:
                    row_data["Group Score"] = score

            data_rows.append(row_data)

        # Create and return DataFrame
        if not data_rows:
            # Return empty DataFrame with appropriate columns
            columns = ["Trial"]
            if not ranked_only:
                columns.extend(["Is Ranked", "Is Feasible"])
            if ranked_only or any(row.get("Crowding Distance") is not None for row in data_rows):
                columns.append("Crowding Distance")
            return pd.DataFrame(columns=columns)

        return pd.DataFrame(data_rows)

    def get_all_trials_dataframe(self) -> pd.DataFrame:
        """
        Convert all trials in the leaderboard to a DataFrame, including those with infinite scores.

        This is a convenience wrapper around get_dataframe(ranked_only=False).

        :return: DataFrame containing all trial information
        :rtype: pd.DataFrame
        """
        return self.get_dataframe(ranked_only=False)

    def get_metadata(self, trial_ids: Optional[Union[int, List[int]]] = None) -> pd.DataFrame:
        """
        Get metadata for one or more trials as a DataFrame.

        This method extracts the metadata from trials and formats it into a DataFrame
        for analysis. It flattens nested metadata dictionaries for easier access.

        By default, this method includes metadata for all trials, including infeasible ones.

        :param trial_ids: Specific trial ID(s) to retrieve metadata for, or None for all trials
        :type trial_ids: Optional[Union[int, List[int]]]
        :return: DataFrame with trial IDs as index and metadata as columns
        :rtype: pd.DataFrame
        """
        # If no trial_ids provided, use all trials that have metadata
        if trial_ids is None:
            trial_ids = [tid for tid, trial in self._data.items() if trial.metadata]
        elif isinstance(trial_ids, int):
            trial_ids = [trial_ids]

        # Create rows for the DataFrame
        metadata_rows = []
        for tid in trial_ids:
            if tid not in self._data:
                continue

            trial = self._data[tid]
            if not trial.metadata:
                continue

            # Start with trial ID and feasibility for the row
            row_data = {
                "Trial": tid,
                "Is Feasible": trial.is_feasible,
                "Is Ranked": tid in self._poset.get_indices(),
            }

            # Flatten nested metadata
            for meta_key, meta_value in trial.metadata.items():
                if isinstance(meta_value, dict):
                    # Flatten nested dictionary
                    for sub_key, sub_value in meta_value.items():
                        row_data[f"{meta_key}_{sub_key}"] = sub_value
                else:
                    row_data[meta_key] = meta_value

            metadata_rows.append(row_data)

        # Create DataFrame from metadata rows
        if not metadata_rows:
            return pd.DataFrame(columns=["Trial", "Is Feasible", "Is Ranked"])

        return pd.DataFrame(metadata_rows).set_index("Trial")

    def add(self, trial: Trial) -> None:
        """
        Add a new trial to the leaderboard.

        All trials are stored in the data dictionary, but only feasible trials
        with all finite scores are added to the partial ordering for ranking.

        Trials with infinite scores or that violate feasibility constraints
        are stored but not ranked.

        :param trial: Trial to add
        :type trial: Trial
        """
        index = trial.trial_id
        self._data[index] = trial
        if trial.is_feasible:
            group_values = self._objective_scorer.score(trial.objectives)
            if all(value < float("inf") for value in group_values):
                self._poset.add(index, group_values)
