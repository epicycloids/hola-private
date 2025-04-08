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

from typing import Any, Dict, List, Optional, Union, Tuple
import json
import os

import pandas as pd
from msgspec import Struct
import numpy as np
import msgspec
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from hola.core.objectives import ObjectiveName, ObjectiveScorer
from hola.core.parameters import ParameterName, CategoricalParameterConfig
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
        poset_indices = self._poset.get_indices() if len(self._poset) > 0 else set()
        return sum(
            1 for tid, trial in self._data.items()
            if trial.is_feasible and tid not in poset_indices
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

        # Get the best trial from the poset (first front, best crowding distance)
        best_indices = self._poset.peek(1)
        if not best_indices:
            return None

        best_index, _ = best_indices[0]
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
        if len(self._poset) == 0:
            return []

        top_indices = self._poset.peek(k)
        return [self._data[idx] for idx, _ in top_indices]

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

        # Get indices from poset (safely handle empty poset)
        poset_indices = self._poset.get_indices() if len(self._poset) > 0 else set()

        # Determine which trials to process
        if ranked_only:
            trial_ids = [key for key, _ in self._poset.items()] if len(self._poset) > 0 else []
        else:
            trial_ids = list(self._data.keys())

        # Process each trial
        for tid in trial_ids:
            trial = self._data[tid]
            is_ranked = tid in poset_indices

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

        # Get indices from poset (safely handle empty poset)
        poset_indices = self._poset.get_indices() if len(self._poset) > 0 else set()

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
            if np.all(group_values < float("inf")):
                self._poset.add(index, group_values)

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
            "trials": list(self._data.values()),
            "is_multigroup": self._objective_scorer.is_multigroup
        }

        # Encode and write to file
        encoded = msgspec.json.encode(output)
        with open(filepath, 'wb') as f:
            f.write(encoded)

    @classmethod
    def load_from_file(cls, filepath: str, objective_scorer: ObjectiveScorer) -> "Leaderboard":
        """
        Load a leaderboard from a JSON file.

        This class method creates a new Leaderboard instance from a file
        previously created with save_to_file.

        :param filepath: Path to the JSON file to load
        :type filepath: str
        :param objective_scorer: The objective scorer to use for the loaded leaderboard
        :type objective_scorer: ObjectiveScorer
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

        # Create new leaderboard
        leaderboard = cls(objective_scorer)

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

    def plot_parameters(
        self,
        param1: ParameterName,
        param2: Optional[ParameterName] = None,
        figsize: Tuple[int, int] = (800, 600)
    ) -> go.Figure:
        """
        Create a scatter plot visualizing the sampled points in parameter space.

        Points are colored by their front level, with front 0 being the Pareto front.
        For categorical parameters, the values are mapped to numeric indices.

        :param param1: Name of the first parameter to plot on x-axis
        :type param1: ParameterName
        :param param2: Optional name of the second parameter to plot on y-axis.
                      If None, y-axis will be set to 0.
        :type param2: Optional[ParameterName]
        :param figsize: Width and height of the figure in pixels
        :type figsize: Tuple[int, int]
        :return: Plotly figure object
        :rtype: go.Figure
        :raises ValueError: If parameter names are not found in the trials
        """
        # Get data from leaderboard
        df = self.get_dataframe(ranked_only=False)

        # Check if parameter exists in data
        if str(param1) not in df.columns:
            raise ValueError(f"Parameter '{param1}' not found in trials")

        # For ranked trials, get their front index
        df_ranked = self.get_dataframe(ranked_only=True)
        front_indices = {}

        if not self._objective_scorer.is_multigroup and len(self._poset) > 0:
            # For single-group, use the order in the poset as the front index
            for i, (tid, _) in enumerate(self._poset.items()):
                front_indices[tid] = i
        else:
            # For multi-group, get the actual fronts
            for i, front in enumerate(self._poset.fronts()):
                for tid, _ in front:
                    front_indices[tid] = i

        # Add front index to dataframe
        df['Front'] = df['Trial'].map(lambda x: front_indices.get(x, np.nan))

        # Prepare data for plotting
        x_data = df[str(param1)]

        # Handle the y-axis data
        if param2 is not None:
            if str(param2) not in df.columns:
                raise ValueError(f"Parameter '{param2}' not found in trials")
            y_data = df[str(param2)]
            y_label = str(param2)
        else:
            # If param2 is not provided, set y values to 0
            y_data = np.zeros_like(df['Trial'].values)
            y_label = ""

        # Check if parameters are categorical and map them to indices if needed
        x_is_categorical = False
        y_is_categorical = False
        x_categories = []
        y_categories = []

        # This check assumes we can determine if a parameter is categorical
        # from its values in the dataframe. A more robust approach would be
        # to access the parameter configurations directly if available.
        if not np.issubdtype(x_data.dtype, np.number):
            x_is_categorical = True
            x_categories = sorted(x_data.unique())
            x_data = x_data.map({cat: i for i, cat in enumerate(x_categories)})

        if param2 is not None and not np.issubdtype(y_data.dtype, np.number):
            y_is_categorical = True
            y_categories = sorted(y_data.unique())
            y_data = y_data.map({cat: i for i, cat in enumerate(y_categories)})

        # Create figure
        fig = px.scatter(
            x=x_data,
            y=y_data,
            color=df['Front'],
            color_continuous_scale='viridis',
            labels={
                'x': str(param1),
                'y': y_label,
                'color': 'Front'
            },
            title=f"Parameter Space: {param1} vs {param2 if param2 else 'Default'}",
            height=figsize[1],
            width=figsize[0],
        )

        # Update layout for categorical parameters if needed
        if x_is_categorical:
            fig.update_xaxes(
                tickmode='array',
                tickvals=list(range(len(x_categories))),
                ticktext=x_categories
            )

        if y_is_categorical:
            fig.update_yaxes(
                tickmode='array',
                tickvals=list(range(len(y_categories))),
                ticktext=y_categories
            )

        # Update layout
        fig.update_layout(
            coloraxis_colorbar=dict(title="Front"),
            hovermode="closest"
        )

        return fig

    def plot_objectives(
        self,
        obj1: ObjectiveName,
        obj2: ObjectiveName,
        figsize: Tuple[int, int] = (800, 600)
    ) -> go.Figure:
        """
        Create a scatter plot visualizing the tradeoff curve between two objectives.

        Points are colored by their front level, with front 0 being the Pareto front.
        Target and limit values for both objectives are shown as dashed lines.

        :param obj1: Name of the first objective to plot on x-axis
        :type obj1: ObjectiveName
        :param obj2: Name of the second objective to plot on y-axis
        :type obj2: ObjectiveName
        :param figsize: Width and height of the figure in pixels
        :type figsize: Tuple[int, int]
        :return: Plotly figure object
        :rtype: go.Figure
        :raises ValueError: If objective names are not found in the trials
        """
        # Get data from leaderboard
        df = self.get_dataframe(ranked_only=False)

        # Check if objectives exist in data
        if str(obj1) not in df.columns:
            raise ValueError(f"Objective '{obj1}' not found in trials")
        if str(obj2) not in df.columns:
            raise ValueError(f"Objective '{obj2}' not found in trials")

        # For ranked trials, get their front index
        df_ranked = self.get_dataframe(ranked_only=True)
        front_indices = {}

        if not self._objective_scorer.is_multigroup and len(self._poset) > 0:
            # For single-group, use the order in the poset as the front index
            for i, (tid, _) in enumerate(self._poset.items()):
                front_indices[tid] = i
        else:
            # For multi-group, get the actual fronts
            for i, front in enumerate(self._poset.fronts()):
                for tid, _ in front:
                    front_indices[tid] = i

        # Add front index to dataframe
        df['Front'] = df['Trial'].map(lambda x: front_indices.get(x, np.nan))

        # Create figure
        fig = px.scatter(
            x=df[str(obj1)],
            y=df[str(obj2)],
            color=df['Front'],
            color_continuous_scale='viridis',
            labels={
                'x': str(obj1),
                'y': str(obj2),
                'color': 'Front'
            },
            title=f"Objective Space: {obj1} vs {obj2}",
            height=figsize[1],
            width=figsize[0],
        )

        # Get target and limit values for objectives
        if obj1 in self._objective_scorer.objectives and obj2 in self._objective_scorer.objectives:
            obj1_config = self._objective_scorer.objectives[obj1]
            obj2_config = self._objective_scorer.objectives[obj2]

            # Add target and limit lines for objective 1
            fig.add_shape(
                type="line",
                x0=obj1_config.target,
                y0=df[str(obj2)].min(),
                x1=obj1_config.target,
                y1=df[str(obj2)].max(),
                line=dict(color="green", width=2, dash="dash"),
                name=f"{obj1} Target"
            )
            fig.add_shape(
                type="line",
                x0=obj1_config.limit,
                y0=df[str(obj2)].min(),
                x1=obj1_config.limit,
                y1=df[str(obj2)].max(),
                line=dict(color="red", width=2, dash="dash"),
                name=f"{obj1} Limit"
            )

            # Add target and limit lines for objective 2
            fig.add_shape(
                type="line",
                x0=df[str(obj1)].min(),
                y0=obj2_config.target,
                x1=df[str(obj1)].max(),
                y1=obj2_config.target,
                line=dict(color="green", width=2, dash="dash"),
                name=f"{obj2} Target"
            )
            fig.add_shape(
                type="line",
                x0=df[str(obj1)].min(),
                y0=obj2_config.limit,
                x1=df[str(obj1)].max(),
                y1=obj2_config.limit,
                line=dict(color="red", width=2, dash="dash"),
                name=f"{obj2} Limit"
            )

            # Add annotations
            fig.add_annotation(
                x=obj1_config.target,
                y=df[str(obj2)].min(),
                text="Target",
                showarrow=False,
                yshift=-20,
                font=dict(color="green")
            )
            fig.add_annotation(
                x=obj1_config.limit,
                y=df[str(obj2)].min(),
                text="Limit",
                showarrow=False,
                yshift=-20,
                font=dict(color="red")
            )
            fig.add_annotation(
                x=df[str(obj1)].min(),
                y=obj2_config.target,
                text="Target",
                showarrow=False,
                xshift=-40,
                font=dict(color="green")
            )
            fig.add_annotation(
                x=df[str(obj1)].min(),
                y=obj2_config.limit,
                text="Limit",
                showarrow=False,
                xshift=-40,
                font=dict(color="red")
            )

        # Update layout
        fig.update_layout(
            coloraxis_colorbar=dict(title="Front"),
            hovermode="closest"
        )

        return fig

    def plot_comparison_groups(
        self,
        group1: int,
        group2: int,
        figsize: Tuple[int, int] = (800, 600)
    ) -> go.Figure:
        """
        Create a scatter plot visualizing the tradeoff curve between two comparison groups.

        Points are colored by their front level, with front 0 being the Pareto front.

        :param group1: ID of the first comparison group to plot on x-axis
        :type group1: int
        :param group2: ID of the second comparison group to plot on y-axis
        :type group2: int
        :param figsize: Width and height of the figure in pixels
        :type figsize: Tuple[int, int]
        :return: Plotly figure object
        :rtype: go.Figure
        :raises ValueError: If comparison group IDs are not found or for single-group problems
        """
        # Verify multi-group configuration
        if not self._objective_scorer.is_multigroup:
            raise ValueError("This method is only applicable for multi-group optimization problems")

        # Get data from leaderboard
        df = self.get_dataframe(ranked_only=True)

        # Check if group scores exist in data
        group1_col = f"Group {group1} Score"
        group2_col = f"Group {group2} Score"

        if group1_col not in df.columns:
            raise ValueError(f"Comparison group '{group1}' not found in trials")
        if group2_col not in df.columns:
            raise ValueError(f"Comparison group '{group2}' not found in trials")

        # Get front indices
        front_indices = {}
        for i, front in enumerate(self._poset.fronts()):
            for tid, _ in front:
                front_indices[tid] = i

        # Add front index to dataframe
        df['Front'] = df['Trial'].map(lambda x: front_indices.get(x, np.nan))

        # Create figure
        fig = px.scatter(
            x=df[group1_col],
            y=df[group2_col],
            color=df['Front'],
            color_continuous_scale='viridis',
            labels={
                'x': f"Group {group1} Score",
                'y': f"Group {group2} Score",
                'color': 'Front'
            },
            title=f"Comparison Group Space: Group {group1} vs Group {group2}",
            height=figsize[1],
            width=figsize[0],
        )

        # Update layout
        fig.update_layout(
            coloraxis_colorbar=dict(title="Front"),
            hovermode="closest"
        )

        return fig

    def plot_comparison_groups_3d(
        self,
        group1: int,
        group2: int,
        group3: int,
        figsize: Tuple[int, int] = (800, 600)
    ) -> go.Figure:
        """
        Create a 3D scatter plot visualizing the tradeoff surface between three comparison groups.

        Points are colored by their front level, with front 0 being the Pareto front.

        :param group1: ID of the first comparison group to plot on x-axis
        :type group1: int
        :param group2: ID of the second comparison group to plot on y-axis
        :type group2: int
        :param group3: ID of the third comparison group to plot on z-axis
        :type group3: int
        :param figsize: Width and height of the figure in pixels
        :type figsize: Tuple[int, int]
        :return: Plotly figure object
        :rtype: go.Figure
        :raises ValueError: If comparison group IDs are not found or for single-group problems
        """
        # Verify multi-group configuration
        if not self._objective_scorer.is_multigroup:
            raise ValueError("This method is only applicable for multi-group optimization problems")

        # Get data from leaderboard
        df = self.get_dataframe(ranked_only=True)

        # Check if group scores exist in data
        group1_col = f"Group {group1} Score"
        group2_col = f"Group {group2} Score"
        group3_col = f"Group {group3} Score"

        if group1_col not in df.columns:
            raise ValueError(f"Comparison group '{group1}' not found in trials")
        if group2_col not in df.columns:
            raise ValueError(f"Comparison group '{group2}' not found in trials")
        if group3_col not in df.columns:
            raise ValueError(f"Comparison group '{group3}' not found in trials")

        # Get front indices
        front_indices = {}
        for i, front in enumerate(self._poset.fronts()):
            for tid, _ in front:
                front_indices[tid] = i

        # Add front index to dataframe
        df['Front'] = df['Trial'].map(lambda x: front_indices.get(x, np.nan))

        # Create figure
        fig = px.scatter_3d(
            x=df[group1_col],
            y=df[group2_col],
            z=df[group3_col],
            color=df['Front'],
            color_continuous_scale='viridis',
            labels={
                'x': f"Group {group1} Score",
                'y': f"Group {group2} Score",
                'z': f"Group {group3} Score",
                'color': 'Front'
            },
            title=f"3D Comparison Group Space: Group {group1} vs Group {group2} vs Group {group3}",
            height=figsize[1],
            width=figsize[0],
        )

        # Update layout
        fig.update_layout(
            coloraxis_colorbar=dict(title="Front"),
            scene=dict(
                xaxis_title=f"Group {group1} Score",
                yaxis_title=f"Group {group2} Score",
                zaxis_title=f"Group {group3} Score"
            )
        )

        return fig

    def plot_objective_vs_trial(
        self,
        objective: ObjectiveName,
        figsize: Tuple[int, int] = (800, 600)
    ) -> go.Figure:
        """
        Create a line plot visualizing how an objective value changes across trials.

        Includes horizontal lines for the target and limit values of the objective.

        :param objective: Name of the objective to plot
        :type objective: ObjectiveName
        :param figsize: Width and height of the figure in pixels
        :type figsize: Tuple[int, int]
        :return: Plotly figure object
        :rtype: go.Figure
        :raises ValueError: If objective name is not found in the trials
        """
        # Get data from leaderboard
        df = self.get_dataframe(ranked_only=False)

        # Check if objective exists in data
        if str(objective) not in df.columns:
            raise ValueError(f"Objective '{objective}' not found in trials")

        # Sort by trial ID to show chronological progression
        df = df.sort_values('Trial')

        # Create figure
        fig = go.Figure()

        # Add objective values
        fig.add_trace(go.Scatter(
            x=df['Trial'],
            y=df[str(objective)],
            mode='lines+markers',
            name=str(objective),
            line=dict(color='blue'),
            marker=dict(size=8)
        ))

        # Add target and limit lines if available
        if objective in self._objective_scorer.objectives:
            obj_config = self._objective_scorer.objectives[objective]

            # Add target line
            fig.add_trace(go.Scatter(
                x=[df['Trial'].min(), df['Trial'].max()],
                y=[obj_config.target, obj_config.target],
                mode='lines',
                name='Target',
                line=dict(color='green', width=2, dash='dash')
            ))

            # Add limit line
            fig.add_trace(go.Scatter(
                x=[df['Trial'].min(), df['Trial'].max()],
                y=[obj_config.limit, obj_config.limit],
                mode='lines',
                name='Limit',
                line=dict(color='red', width=2, dash='dash')
            ))

        # Update layout
        fig.update_layout(
            title=f"Objective {objective} vs Trial ID",
            xaxis_title="Trial ID",
            yaxis_title=f"{objective} Value",
            height=figsize[1],
            width=figsize[0],
            hovermode="closest",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        return fig

    def get_objective_names(self) -> list[ObjectiveName]:
        """
        Get all objective names defined in the leaderboard.

        :return: List of objective names
        :rtype: list[ObjectiveName]
        """
        return list(self._objective_scorer.objectives.keys())

    def get_parameter_names(self) -> list[ParameterName]:
        """
        Get all parameter names used by trials in the leaderboard.

        :return: List of parameter names
        :rtype: list[ParameterName]
        """
        if not self._data:
            return []

        # Extract parameter names from the first trial
        # All trials should have the same parameter names
        first_trial = next(iter(self._data.values()))
        return list(first_trial.parameters.keys())

    def get_comparison_group_ids(self) -> list[int]:
        """
        Get all comparison group IDs for multi-group optimization.

        For single-group optimization, returns [0].

        :return: List of comparison group IDs
        :rtype: list[int]
        """
        if not self._objective_scorer.is_multigroup:
            return [0]  # Single group optimization has only group 0

        # For multi-group, determine the number of groups from the first ranked trial
        if len(self._poset) > 0:
            first_trial_id = next(iter(self._poset.keys()))
            score = self._poset[first_trial_id]
            return list(range(len(score)))

        # If no ranked trials, try to get group count from the objective scorer
        if hasattr(self._objective_scorer, 'group_count'):
            return list(range(self._objective_scorer.group_count))

        # Last resort: we don't know how many groups there are
        return []
