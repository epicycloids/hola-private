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
import json
import os

import pandas as pd
from msgspec import Struct
import numpy as np
import msgspec

from hola.core.objectives import ObjectiveName, ObjectiveScorer
from hola.core.parameters import ParameterName
from hola.core.poset import ScalarPoset, VectorPoset
from hola.core.repository import Trial, TrialRepository, MemoryTrialRepository


class Leaderboard:
    """
    Tracks and ranks optimization trials.

    The leaderboard maintains trials in a partially ordered set. For
    single-group optimization, trials are totally ordered. For multi-group
    optimization, trials are organized into fronts using non-dominated sorting
    based on their comparison group scores. Within each front, trials are
    ordered by crowding distance to promote diversity in parameter choices.
    """

    def __init__(self, objective_scorer: ObjectiveScorer, repository: Optional[TrialRepository] = None):
        """
        Initialize the leaderboard.

        :param objective_scorer: Scorer that defines how to evaluate trials
        :type objective_scorer: ObjectiveScorer
        :param repository: Repository for trial storage, or None to use in-memory storage
        :type repository: Optional[TrialRepository]
        """
        self._objective_scorer = objective_scorer
        self._poset = (
            VectorPoset[int]() if self._objective_scorer.is_multigroup else ScalarPoset[int]()
        )
        self._repository = repository if repository is not None else MemoryTrialRepository()

    def get_feasible_count(self) -> int:
        """
        Get the count of all feasible trials, including those with infinite scores.

        This includes both:
        - Feasible trials with all finite scores (ranked in poset)
        - Feasible trials with at least one infinite score (not ranked)

        :return: Number of feasible trials in the leaderboard
        :rtype: int
        """
        return len(self._repository.get_feasible_trials())

    def get_feasible_infinite_count(self) -> int:
        """
        Get the count of feasible trials with at least one infinite score.

        These trials are stored in the leaderboard but not ranked in the poset
        due to having at least one infinite objective score.

        :return: Number of feasible trials with infinite scores
        :rtype: int
        """
        poset_indices = self._poset.get_indices() if len(self._poset) > 0 else set()
        return sum(
            1 for trial in self._repository.get_feasible_trials()
            if trial.trial_id not in poset_indices
        )

    def get_infeasible_count(self) -> int:
        """
        Get the count of infeasible trials.

        These are trials where parameter values violated feasibility constraints.

        :return: Number of infeasible trials
        :rtype: int
        """
        return sum(1 for trial in self._repository.get_all_trials() if not trial.is_feasible)

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
        return len(self._repository.get_trial_ids())

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
        trial = self._repository.get_trial(trial_id)
        if trial is None:
            raise KeyError(f"Trial with ID {trial_id} not found")
        return trial

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

        # Get the best trial from the poset (first front, best crowding distance)
        best_indices = self._poset.peek(1)
        if not best_indices:
            return None

        best_index, _ = best_indices[0]
        return self._repository.get_trial(best_index)

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
        if len(self._poset) == 0:
            return []

        top_indices = self._poset.peek(k)
        result = []
        for idx, _ in top_indices:
            trial = self._repository.get_trial(idx)
            if trial is not None:
                result.append(trial)
        return result

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

        # Early return if poset is empty
        if len(self._poset) == 0:
            return []

        result = []
        # Take the first k fronts from the poset
        for i, front in enumerate(self._poset.fronts()):
            if i >= k:
                break

            # Convert each front from (id, score) tuples to Trial objects
            trial_front = []
            for trial_id, _ in front:
                trial = self._repository.get_trial(trial_id)
                if trial is not None:
                    trial_front.append(trial)
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
        # First get basic dataframe from repository
        df = self._repository.get_dataframe(ranked_only)

        if df.empty:
            return df

        # Add poset-specific information for ranked trials
        if ranked_only:
            # Add score and crowding distance columns
            df["Crowding Distance"] = df["Trial"].apply(
                lambda tid: self._poset.get_crowding_distance(tid)
                if tid in self._poset.get_indices() else None
            )

            # Add score columns
            for tid in df["Trial"]:
                if tid in self._poset:
                    score = self._poset[tid]
                    if self._objective_scorer.is_multigroup:
                        for j in range(len(score)):
                            df.loc[df["Trial"] == tid, f"Group {j} Score"] = score[j]
                    else:
                        df.loc[df["Trial"] == tid, "Group Score"] = score

        return df

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
            all_trials = self._repository.get_all_trials()
            trial_ids = [trial.trial_id for trial in all_trials if trial.metadata]
        elif isinstance(trial_ids, int):
            trial_ids = [trial_ids]

        # Get indices from poset (safely handle empty poset)
        poset_indices = self._poset.get_indices() if len(self._poset) > 0 else set()

        # Create rows for the DataFrame
        metadata_rows = []
        for tid in trial_ids:
            trial = self._repository.get_trial(tid)
            if trial is None or not trial.metadata:
                continue

            # Start with trial ID and feasibility for the row
            row_data = {
                "Trial": tid,
                "Is Feasible": trial.is_feasible,
                "Is Ranked": tid in poset_indices,
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

        All trials are stored in the repository, but only feasible trials
        with all finite scores are added to the partial ordering for ranking.

        Trials with infinite scores or that violate feasibility constraints
        are stored but not ranked.

        :param trial: Trial to add
        :type trial: Trial
        """
        is_ranked = False
        if trial.is_feasible:
            group_values = self._objective_scorer.score(trial.objectives)
            if np.all(np.array(group_values) < float("inf")):
                self._poset.add(trial.trial_id, group_values)
                is_ranked = True

        # Add to repository with ranked status
        self._repository.add_trial(trial, is_ranked=is_ranked)

    def save_to_file(self, filepath: str) -> None:
        """
        Save the leaderboard state to a JSON file.

        This method serializes all trials and their data to a JSON file
        that can later be loaded back into a Leaderboard.

        :param filepath: Path where the JSON file will be saved
        :type filepath: str
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        # Create output structure with trials list
        output = {
            "trials": self._repository.get_all_trials(),
            "is_multigroup": self._objective_scorer.is_multigroup
        }

        # Encode and write to file
        encoded = msgspec.json.encode(output)
        with open(filepath, 'wb') as f:
            f.write(encoded)

    @classmethod
    def load_from_file(cls, filepath: str, objective_scorer: ObjectiveScorer,
                      repository: Optional[TrialRepository] = None) -> "Leaderboard":
        """
        Load a leaderboard from a JSON file.

        This class method creates a new Leaderboard instance from a file
        previously created with save_to_file.

        :param filepath: Path to the JSON file to load
        :type filepath: str
        :param objective_scorer: The objective scorer to use for the loaded leaderboard
        :type objective_scorer: ObjectiveScorer
        :param repository: Optional repository to use for trial storage
        :type repository: Optional[TrialRepository]
        :return: A new Leaderboard instance with the loaded trials
        :rtype: Leaderboard
        :raises FileNotFoundError: If the file doesn't exist
        :raises ValueError: If there's a mismatch between the file's multigroup setting
                           and the provided objective_scorer
        """
        # Check if file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No file found at {filepath}")

        # Load and decode data
        with open(filepath, 'rb') as f:
            data = msgspec.json.decode(f.read(), type=dict[str, Any])

        # Verify compatibility with the provided objective_scorer
        if data.get("is_multigroup") != objective_scorer.is_multigroup:
            raise ValueError(
                "Mismatch between loaded file and provided objective_scorer. "
                f"File has is_multigroup={data.get('is_multigroup')}, but "
                f"objective_scorer has is_multigroup={objective_scorer.is_multigroup}"
            )

        # Create new leaderboard with specified or in-memory repository
        leaderboard = cls(objective_scorer, repository)

        # Manually deserialize and add trials
        for trial_data in data.get("trials", []):
            # Make sure all objective values are floats (no None values)
            if "objectives" in trial_data:
                objectives_dict = {}
                for obj_name, obj_value in trial_data["objectives"].items():
                    # Convert None to infinity to ensure it's a float
                    if obj_value is None:
                        obj_value = float('inf')
                    objectives_dict[ObjectiveName(obj_name)] = float(obj_value)

                # Create the trial
                trial = Trial(
                    trial_id=trial_data["trial_id"],
                    objectives=objectives_dict,
                    parameters={ParameterName(k): v for k, v in trial_data.get("parameters", {}).items()},
                    is_feasible=trial_data.get("is_feasible", True),
                    metadata=trial_data.get("metadata", {})
                )

                # Add the trial to the leaderboard
                leaderboard.add(trial)

        return leaderboard
