import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from hola.core.leaderboard import Leaderboard, Trial
from hola.core.objectives import ObjectiveName, ObjectiveScorer
from hola.core.parameters import ParameterName


class TestLeaderboard:
    """
    Tests for Leaderboard class, particularly focused on serialization and deserialization.
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
            def __init__(self):
                self.comparison_group = 0

        class MultiGroupScorer(ObjectiveScorer):
            def __init__(self):
                super().__init__(objectives={ObjectiveName("dummy"): MockObjectiveConfig()})

            @property
            def is_multigroup(self) -> bool:
                return True

            def score(self, objectives):
                # Group 1: Sum of first half of objectives
                # Group 2: Sum of second half of objectives
                obj_values = np.array(list(objectives.values()))
                mid = len(obj_values) // 2
                return np.array([sum(obj_values[:mid]), sum(obj_values[mid:])])

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
                metadata={"source": "test"}
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