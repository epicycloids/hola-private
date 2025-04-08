import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import pandas as pd

from hola.core.leaderboard import Leaderboard, Trial
from hola.core.objectives import ObjectiveName, ObjectiveScorer
from hola.core.parameters import ParameterName


class TestLeaderboard:
    """
    Tests for Leaderboard class functionality.
    """

    @pytest.fixture
    def simple_objective_scorer(self):
        """
        Create a simple objective scorer for testing.
        """
        class MockObjectiveConfig:
            def __init__(self):
                self.comparison_group = 0

        class SimpleScorer(ObjectiveScorer):
            def __init__(self):
                super().__init__(objectives={ObjectiveName("dummy"): MockObjectiveConfig()})

            @property
            def is_multigroup(self) -> bool:
                return False

            def score(self, objectives):
                # Return a scalar sum value for single-group scoring
                return float(sum(objectives.values()))

        return SimpleScorer()

    @pytest.fixture
    def multigroup_objective_scorer(self):
        """
        Create a multigroup objective scorer for testing.
        """
        class MockObjectiveConfig:
            def __init__(self, group=0):
                self.comparison_group = group

        class MultiGroupScorer(ObjectiveScorer):
            def __init__(self):
                objectives = {
                    ObjectiveName("obj1"): MockObjectiveConfig(0),
                    ObjectiveName("obj2"): MockObjectiveConfig(1)
                }
                super().__init__(objectives=objectives)

            @property
            def is_multigroup(self) -> bool:
                return True

            def score(self, objectives):
                # Return a two-group vector score: [obj1, obj2]
                return np.array([
                    objectives.get(ObjectiveName("obj1"), 0.0),
                    objectives.get(ObjectiveName("obj2"), 0.0)
                ])

        return MultiGroupScorer()

    @pytest.fixture
    def sample_trials(self):
        """
        Create a list of sample trials for testing.
        """
        return [
            Trial(
                trial_id=1,
                objectives={
                    ObjectiveName("obj1"): 1.0,
                    ObjectiveName("obj2"): 2.0,
                },
                parameters={
                    ParameterName("param1"): 10,
                    ParameterName("param2"): "test",
                },
                is_feasible=True,
                metadata={"source": "test", "nested": {"value": 123}}
            ),
            Trial(
                trial_id=2,
                objectives={
                    ObjectiveName("obj1"): 3.0,
                    ObjectiveName("obj2"): 4.0,
                },
                parameters={
                    ParameterName("param1"): 20,
                    ParameterName("param2"): "test2",
                },
                is_feasible=True,
                metadata={"source": "test"}
            ),
            Trial(
                trial_id=3,
                objectives={
                    ObjectiveName("obj1"): 5.0,
                    ObjectiveName("obj2"): float("inf"),
                },
                parameters={
                    ParameterName("param1"): 30,
                    ParameterName("param2"): "test3",
                },
                is_feasible=True,
                metadata={"source": "test"}
            ),
            Trial(
                trial_id=4,
                objectives={
                    ObjectiveName("obj1"): 7.0,
                    ObjectiveName("obj2"): 8.0,
                },
                parameters={
                    ParameterName("param1"): 40,
                    ParameterName("param2"): "test4",
                },
                is_feasible=False,
                metadata={"source": "test"}
            )
        ]

    @pytest.fixture
    def populated_leaderboard(self, simple_objective_scorer, sample_trials):
        """
        Create a populated leaderboard for testing.
        """
        leaderboard = Leaderboard(simple_objective_scorer)
        for trial in sample_trials:
            leaderboard.add(trial)
        return leaderboard

    @pytest.fixture
    def populated_multigroup_leaderboard(self, multigroup_objective_scorer, sample_trials):
        """
        Create a populated multigroup leaderboard for testing.
        """
        leaderboard = Leaderboard(multigroup_objective_scorer)
        for trial in sample_trials:
            leaderboard.add(trial)
        return leaderboard

    def test_initialization(self, simple_objective_scorer):
        """
        Test leaderboard initialization.
        """
        leaderboard = Leaderboard(simple_objective_scorer)
        assert leaderboard.get_total_count() == 0
        assert leaderboard.get_ranked_count() == 0
        assert leaderboard.get_best_trial() is None

    def test_add_feasible_trial(self, simple_objective_scorer):
        """
        Test adding a feasible trial with finite objectives.
        """
        leaderboard = Leaderboard(simple_objective_scorer)
        trial = Trial(
            trial_id=1,
            objectives={ObjectiveName("obj"): 1.0},
            parameters={ParameterName("param"): 10},
        )

        leaderboard.add(trial)

        assert leaderboard.get_total_count() == 1
        assert leaderboard.get_feasible_count() == 1
        assert leaderboard.get_infeasible_count() == 0
        assert leaderboard.get_ranked_count() == 1
        assert leaderboard.get_best_trial() == trial

    def test_add_infeasible_trial(self, simple_objective_scorer):
        """
        Test adding an infeasible trial.
        """
        leaderboard = Leaderboard(simple_objective_scorer)
        trial = Trial(
            trial_id=1,
            objectives={ObjectiveName("obj"): 1.0},
            parameters={ParameterName("param"): 10},
            is_feasible=False
        )

        leaderboard.add(trial)

        assert leaderboard.get_total_count() == 1
        assert leaderboard.get_feasible_count() == 0
        assert leaderboard.get_infeasible_count() == 1
        assert leaderboard.get_ranked_count() == 0
        assert leaderboard.get_best_trial() is None

    def test_add_infinite_score_trial(self, simple_objective_scorer):
        """
        Test adding a feasible trial with infinite objective score.
        """
        leaderboard = Leaderboard(simple_objective_scorer)
        trial = Trial(
            trial_id=1,
            objectives={ObjectiveName("obj"): float("inf")},
            parameters={ParameterName("param"): 10},
        )

        leaderboard.add(trial)

        assert leaderboard.get_total_count() == 1
        assert leaderboard.get_feasible_count() == 1
        assert leaderboard.get_feasible_infinite_count() == 1
        assert leaderboard.get_ranked_count() == 0
        assert leaderboard.get_best_trial() is None

    def test_get_trial(self, populated_leaderboard, sample_trials):
        """
        Test retrieval of a specific trial.
        """
        trial = populated_leaderboard.get_trial(2)
        assert trial.trial_id == 2
        assert trial.objectives == sample_trials[1].objectives
        assert trial.parameters == sample_trials[1].parameters

        # Test with nonexistent ID
        with pytest.raises(KeyError):
            populated_leaderboard.get_trial(999)

    def test_get_best_trial(self, simple_objective_scorer):
        """
        Test retrieval of the best trial (lowest score).
        """
        leaderboard = Leaderboard(simple_objective_scorer)

        trial1 = Trial(
            trial_id=1,
            objectives={ObjectiveName("obj"): 10.0},
            parameters={ParameterName("param"): 10},
        )

        trial2 = Trial(
            trial_id=2,
            objectives={ObjectiveName("obj"): 5.0},
            parameters={ParameterName("param"): 20},
        )

        leaderboard.add(trial1)
        leaderboard.add(trial2)

        best_trial = leaderboard.get_best_trial()
        assert best_trial is not None
        assert best_trial.trial_id == 2  # Trial 2 has lower score (5.0 vs 10.0)

    def test_get_top_k(self, simple_objective_scorer):
        """
        Test retrieval of top k trials.
        """
        leaderboard = Leaderboard(simple_objective_scorer)

        # Add trials with scores 5, 3, 10, 7
        for i, score in enumerate([5, 3, 10, 7], 1):
            trial = Trial(
                trial_id=i,
                objectives={ObjectiveName("obj"): float(score)},
                parameters={ParameterName("param"): i * 10},
            )
            leaderboard.add(trial)

        # Get top 2 trials (should be trials with scores 3, 5)
        top_trials = leaderboard.get_top_k(2)
        assert len(top_trials) == 2
        assert top_trials[0].trial_id == 2  # Score 3
        assert top_trials[1].trial_id == 1  # Score 5

        # Get top 5 trials (should return all 4 trials)
        top_trials = leaderboard.get_top_k(5)
        assert len(top_trials) == 4

        # Get top 0 trials (should return empty list)
        with pytest.raises(ValueError):
            leaderboard.get_top_k(0)

    def test_get_top_k_fronts_simple(self, simple_objective_scorer):
        """
        Test retrieval of top k fronts with a simple scorer.
        """
        leaderboard = Leaderboard(simple_objective_scorer)

        # Add trials with scores 5, 3, 10, 7
        for i, score in enumerate([5, 3, 10, 7], 1):
            trial = Trial(
                trial_id=i,
                objectives={ObjectiveName("obj"): float(score)},
                parameters={ParameterName("param"): i * 10},
            )
            leaderboard.add(trial)

        # For a simple scorer, each front should contain one trial
        fronts = leaderboard.get_top_k_fronts(2)
        assert len(fronts) == 2
        assert len(fronts[0]) == 1  # First front has 1 trial
        assert len(fronts[1]) == 1  # Second front has 1 trial
        assert fronts[0][0].trial_id == 2  # Score 3
        assert fronts[1][0].trial_id == 1  # Score 5

    def test_get_top_k_fronts_multigroup(self, multigroup_objective_scorer):
        """
        Test retrieval of top k fronts with a multigroup scorer.
        """
        leaderboard = Leaderboard(multigroup_objective_scorer)

        # Add trials with different objective combinations for 2D Pareto front
        trials = [
            Trial(
                trial_id=1,
                objectives={
                    ObjectiveName("obj1"): 1.0,
                    ObjectiveName("obj2"): 5.0,
                },
                parameters={ParameterName("param"): 10},
            ),
            Trial(
                trial_id=2,
                objectives={
                    ObjectiveName("obj1"): 2.0,
                    ObjectiveName("obj2"): 3.0,
                },
                parameters={ParameterName("param"): 20},
            ),
            Trial(
                trial_id=3,
                objectives={
                    ObjectiveName("obj1"): 4.0,
                    ObjectiveName("obj2"): 1.0,
                },
                parameters={ParameterName("param"): 30},
            ),
            Trial(
                trial_id=4,
                objectives={
                    ObjectiveName("obj1"): 3.0,
                    ObjectiveName("obj2"): 6.0,
                },
                parameters={ParameterName("param"): 40},
            ),
        ]

        for trial in trials:
            leaderboard.add(trial)

        # Get Pareto fronts
        fronts = leaderboard.get_top_k_fronts(2)
        assert len(fronts) <= 2  # Should have at most 2 fronts

        # First front should contain non-dominated trials
        # With these objective values, trials 1, 2, and 3 should form the first front
        first_front_ids = {trial.trial_id for trial in fronts[0]}
        assert len(first_front_ids) == 3
        assert 1 in first_front_ids
        assert 2 in first_front_ids
        assert 3 in first_front_ids

        # Second front should contain dominated trial (4)
        if len(fronts) > 1:
            assert len(fronts[1]) == 1
            assert fronts[1][0].trial_id == 4

    def test_get_dataframe(self, populated_leaderboard):
        """
        Test conversion of leaderboard to DataFrame.
        """
        # Test with ranked trials only
        df = populated_leaderboard.get_dataframe(ranked_only=True)
        assert isinstance(df, pd.DataFrame)

        # Should only include feasible trials with finite scores (2 in our sample data)
        assert len(df) == 2

        # All trials should have Trial column
        assert "Trial" in df.columns

        # Parameter and objective columns should be included
        assert "param1" in df.columns
        assert "obj1" in df.columns

        # Check all trials dataframe
        df_all = populated_leaderboard.get_all_trials_dataframe()
        assert isinstance(df_all, pd.DataFrame)
        assert len(df_all) == 4  # All 4 trials

        # Should include status columns
        assert "Is Ranked" in df_all.columns
        assert "Is Feasible" in df_all.columns

    def test_get_metadata(self, populated_leaderboard):
        """
        Test retrieval of trial metadata.
        """
        # All trials have metadata
        df = populated_leaderboard.get_metadata()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4  # All 4 trials have metadata

        # Single trial metadata
        df_single = populated_leaderboard.get_metadata(1)
        assert len(df_single) == 1
        assert df_single.index[0] == 1
        assert df_single.loc[1, "source"] == "test"
        assert df_single.loc[1, "nested_value"] == 123  # Nested metadata

        # Multiple trials metadata
        df_multi = populated_leaderboard.get_metadata([1, 2])
        assert len(df_multi) == 2
        assert list(df_multi.index) == [1, 2]

        # Non-existent trial ID
        df_nonexistent = populated_leaderboard.get_metadata(999)
        assert len(df_nonexistent) == 0

    def test_save_to_file(self, populated_leaderboard):
        """
        Test that save_to_file correctly serializes the leaderboard to a JSON file.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, "test_leaderboard.json")
            populated_leaderboard.save_to_file(filepath)

            # Verify the file exists
            assert os.path.exists(filepath)

            # Verify the file contains valid JSON
            with open(filepath, 'r') as f:
                data = json.load(f)

            # Check basic structure
            assert "trials" in data
            assert "is_multigroup" in data
            assert not data["is_multigroup"]  # Simple scorer is not multigroup

            # Check number of trials
            assert len(data["trials"]) == 4  # All trials should be saved, including infeasible ones

    def test_load_from_file(self, populated_leaderboard, simple_objective_scorer):
        """
        Test that load_from_file correctly loads a leaderboard from a JSON file.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, "test_leaderboard.json")

            # Save the leaderboard
            populated_leaderboard.save_to_file(filepath)

            # Load the leaderboard
            loaded_leaderboard = Leaderboard.load_from_file(filepath, simple_objective_scorer)

            # Verify the loaded leaderboard has the correct number of trials
            assert loaded_leaderboard.get_total_count() == populated_leaderboard.get_total_count()

            # Verify the counts match
            assert loaded_leaderboard.get_feasible_count() == populated_leaderboard.get_feasible_count()
            assert loaded_leaderboard.get_infeasible_count() == populated_leaderboard.get_infeasible_count()
            assert loaded_leaderboard.get_ranked_count() == populated_leaderboard.get_ranked_count()

            # Check that we can retrieve a specific trial and its data is correct
            original_trial = populated_leaderboard.get_trial(1)
            loaded_trial = loaded_leaderboard.get_trial(1)

            assert loaded_trial.trial_id == original_trial.trial_id
            assert loaded_trial.is_feasible == original_trial.is_feasible
            assert loaded_trial.metadata == original_trial.metadata

            # Check objective values
            for obj_name, obj_value in original_trial.objectives.items():
                assert loaded_trial.objectives[obj_name] == obj_value

            # Check parameter values
            for param_name, param_value in original_trial.parameters.items():
                assert loaded_trial.parameters[param_name] == param_value

    def test_round_trip_multigroup(self, populated_multigroup_leaderboard, multigroup_objective_scorer):
        """
        Test a complete round-trip save and load for a multigroup leaderboard.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, "test_multigroup_leaderboard.json")

            # Save the leaderboard
            populated_multigroup_leaderboard.save_to_file(filepath)

            # Load the leaderboard
            loaded_leaderboard = Leaderboard.load_from_file(filepath, multigroup_objective_scorer)

            # Verify the counts match
            assert loaded_leaderboard.get_total_count() == populated_multigroup_leaderboard.get_total_count()
            assert loaded_leaderboard.get_ranked_count() == populated_multigroup_leaderboard.get_ranked_count()

            # Get and compare best trials from both leaderboards
            if populated_multigroup_leaderboard.get_best_trial() is not None:
                original_best = populated_multigroup_leaderboard.get_best_trial()
                loaded_best = loaded_leaderboard.get_best_trial()

                assert loaded_best.trial_id == original_best.trial_id

                # Check objective values
                for obj_name, obj_value in original_best.objectives.items():
                    assert loaded_best.objectives[obj_name] == obj_value

    def test_file_not_found(self, simple_objective_scorer):
        """
        Test that load_from_file raises FileNotFoundError when the file doesn't exist.
        """
        with pytest.raises(FileNotFoundError):
            Leaderboard.load_from_file("nonexistent_file.json", simple_objective_scorer)

    def test_scorer_mismatch(self, populated_leaderboard, multigroup_objective_scorer):
        """
        Test that load_from_file raises ValueError when the objective_scorer doesn't match.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, "test_leaderboard.json")

            # Save the leaderboard with a simple scorer
            populated_leaderboard.save_to_file(filepath)

            # Try to load with a multigroup scorer, which should fail
            with pytest.raises(ValueError, match="Mismatch between loaded file and provided objective_scorer"):
                Leaderboard.load_from_file(filepath, multigroup_objective_scorer)

    def test_save_to_file_creates_directory(self, populated_leaderboard):
        """
        Test that save_to_file creates the directory structure if it doesn't exist.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a nested path that doesn't exist
            nested_dir = os.path.join(temp_dir, "nested", "path")
            filepath = os.path.join(nested_dir, "test_leaderboard.json")

            # Directory shouldn't exist yet
            assert not os.path.exists(nested_dir)

            # Save should create the directory
            populated_leaderboard.save_to_file(filepath)

            # Verify the directory and file now exist
            assert os.path.exists(nested_dir)
            assert os.path.exists(filepath)

    def test_save_and_load_empty_leaderboard(self, simple_objective_scorer):
        """
        Test saving and loading an empty leaderboard.
        """
        empty_leaderboard = Leaderboard(simple_objective_scorer)

        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, "empty_leaderboard.json")

            # Save the empty leaderboard
            empty_leaderboard.save_to_file(filepath)

            # Load the leaderboard
            loaded_leaderboard = Leaderboard.load_from_file(filepath, simple_objective_scorer)

            # Verify it's still empty
            assert loaded_leaderboard.get_total_count() == 0
            assert loaded_leaderboard.get_best_trial() is None