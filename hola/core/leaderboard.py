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

Trials are categorized into three types:
1. Feasible trials with all finite-value scores (ranked in poset)
2. Feasible trials with at least one infinite-value score (stored but not ranked)
3. Infeasible trials (stored but not counted in total length)
"""

from typing import Any, Dict, List, Optional, Union

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
        - Category 1: Feasible trials with all finite scores (ranked in poset)
        - Category 2: Feasible trials with at least one infinite score (not ranked)

        :return: Number of feasible trials in the leaderboard
        :rtype: int
        """
        return sum(1 for trial in self._data.values() if trial.is_feasible)

    def get_feasible_infinite_count(self) -> int:
        """
        Get the count of feasible trials with at least one infinite score.

        This counts only Category 2 trials: feasible but with at least one
        infinite score, which are not ranked in the poset.

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

        This counts only Category 3 trials: those marked as infeasible.

        :return: Number of infeasible trials
        :rtype: int
        """
        return sum(1 for trial in self._data.values() if not trial.is_feasible)

    def get_total_count(self) -> int:
        """
        Get the total count of all trials in the leaderboard.

        This includes all three categories:
        1. Feasible trials with all finite scores (ranked in poset)
        2. Feasible trials with at least one infinite score (not ranked)
        3. Infeasible trials

        :return: Total number of trials stored in the leaderboard
        :rtype: int
        """
        return len(self._data)

    def get_ranked_count(self) -> int:
        """
        Get the number of trials in the leaderboard's partial ordering.

        Note: This only counts feasible trials with all finite scores (category 1).

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

    def get_dataframe(self) -> pd.DataFrame:
        """
        Convert leaderboard to a DataFrame of ranked trials.

        Creates a DataFrame containing information about trials with finite scores,
        ordered by non-dominated sorting and crowding distance. The
        DataFrame includes:
        - Trial IDs
        - Parameter values
        - Objective values
        - Comparison group scores
        - Crowding distance (provides diversity measure within a Pareto front)

        Note:
        - Only trials with finite scores are included in the DataFrame
        - Trials are ordered first by their non-dominated front membership
        - Within each front, trials are ordered by crowding distance to
          promote diversity in parameter choices
        - The ordering ensures best trials (according to comparison group
          scores) appear in the top rows
        - For trials with infinite scores, use the get_metadata() method

        :return: DataFrame containing ranked trial information, sorted by ranking
        :rtype: pd.DataFrame
        """
        data_rows = []
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

            # Create a row with all information
            row_data = {
                "Trial": key,
                "Crowding Distance": crowding_distance,
                **{str(k): v for k, v in trial.parameters.items()},
                **{str(k): v for k, v in trial.objectives.items()},
                **score_dict,
            }
            data_rows.append(row_data)

        # Create DataFrame from the list of dictionaries
        if not data_rows:
            # Return empty DataFrame with expected columns if no data
            return pd.DataFrame(columns=["Trial", "Crowding Distance"])

        return pd.DataFrame(data_rows)

    def get_all_trials_dataframe(self) -> pd.DataFrame:
        """
        Convert all trials in the leaderboard to a DataFrame, including those with infinite scores.

        Unlike get_dataframe(), this method includes ALL trials, even those with infinite scores
        that aren't ranked in the partial ordering.

        :return: DataFrame containing all trial information
        :rtype: pd.DataFrame
        """
        data_rows = []
        for tid, trial in self._data.items():
            # Check if this trial is in the poset (has finite scores)
            is_ranked = tid in self._poset.get_indices()

            # Create a row with basic trial information
            row_data = {
                "Trial": tid,
                "Is Ranked": is_ranked,
                "Is Feasible": trial.is_feasible,
                **{str(k): v for k, v in trial.parameters.items()},
                **{str(k): v for k, v in trial.objectives.items()},
            }

            # Add scores for ranked trials
            if is_ranked:
                score = self._poset[tid]  # Use __getitem__ instead of get_score
                crowding_distance = self._poset.get_crowding_distance(tid)
                row_data["Crowding Distance"] = crowding_distance

                if self._objective_scorer.is_multigroup:
                    for j in range(len(score)):
                        row_data[f"Group {j} Score"] = score[j]
                else:
                    row_data["Group Score"] = score

            data_rows.append(row_data)

        # Create DataFrame from the list of dictionaries
        if not data_rows:
            return pd.DataFrame(columns=["Trial", "Is Ranked", "Is Feasible"])

        return pd.DataFrame(data_rows)

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

        Trials are categorized as follows:
        1. Feasible trials with all finite scores are added to the partial ordering
        2. Feasible trials with infinite scores are stored but not ranked
        3. Infeasible trials are stored but not counted in total length

        :param trial: Trial to add
        :type trial: Trial
        """
        index = trial.trial_id
        self._data[index] = trial
        if trial.is_feasible:
            group_values = self._objective_scorer.score(trial.objectives)
            if all(value < float("inf") for value in group_values):
                self._poset.add(index, group_values)
