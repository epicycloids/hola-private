"""Tests for leaderboard functionality."""

from pathlib import Path

import pytest

from hola.core.leaderboard import (
    GROUP_SCORE_PREFIX,
    OBJECTIVE_PREFIX,
    PARAM_PREFIX,
    Leaderboard,
    ParameterName,
    Trial,
)
from hola.core.objective import Direction, ObjectiveConfig, ObjectiveName, ObjectiveScorer


class TestTrial:
    def test_valid_trial(self) -> None:
        trial = Trial(parameters={ParameterName("lr"): 0.001}, objectives={ObjectiveName("acc"): 0.95})
        assert trial.parameters[ParameterName("lr")] == 0.001
        assert trial.objectives[ObjectiveName("acc")] == 0.95


class TestLeaderboard:
    @pytest.fixture
    def simple_scorer(self) -> ObjectiveScorer:
        """Single objective scorer - maximize accuracy."""
        objectives = {
            ObjectiveName("accuracy"): ObjectiveConfig(
                target=0.95, limit=0.80, direction=Direction.MAXIMIZE
            )
        }
        return ObjectiveScorer(objectives)

    @pytest.fixture
    def dual_scorer(self) -> ObjectiveScorer:
        """Dual objective scorer - accuracy and time in same group."""
        objectives = {
            ObjectiveName("accuracy"): ObjectiveConfig(
                target=0.95,
                limit=0.80,
                direction=Direction.MAXIMIZE,
                priority=2.0,
            ),
            ObjectiveName("time"): ObjectiveConfig(
                target=60.0,
                limit=300.0,
                direction=Direction.MINIMIZE,
                priority=1.0,
            ),
        }
        return ObjectiveScorer(objectives)

    @pytest.fixture
    def pareto_scorer(self) -> ObjectiveScorer:
        """Multi-group scorer for Pareto optimization."""
        objectives = {
            ObjectiveName("accuracy"): ObjectiveConfig(
                target=0.95,
                limit=0.80,
                direction=Direction.MAXIMIZE,
                comparison_group=0,
            ),
            ObjectiveName("time"): ObjectiveConfig(
                target=60.0,
                limit=300.0,
                direction=Direction.MINIMIZE,
                comparison_group=1,
            ),
        }
        return ObjectiveScorer(objectives)

    @pytest.fixture
    def populated_leaderboard(self, simple_scorer: ObjectiveScorer) -> Leaderboard:
        """Leaderboard with some samples for single objective case."""
        board = Leaderboard(simple_scorer)
        board.add(params={ParameterName("lr"): 0.001}, objectives={ObjectiveName("accuracy"): 0.90})
        board.add(params={ParameterName("lr"): 0.01}, objectives={ObjectiveName("accuracy"): 0.85})
        return board

    @pytest.fixture
    def populated_dual_leaderboard(self, dual_scorer: ObjectiveScorer) -> Leaderboard:
        """Leaderboard with samples for dual objective case."""
        board = Leaderboard(dual_scorer)
        board.add(
            params={ParameterName("lr"): 0.001},
            objectives={ObjectiveName("accuracy"): 0.90, ObjectiveName("time"): 120.0},
        )
        board.add(
            params={ParameterName("lr"): 0.01},
            objectives={ObjectiveName("accuracy"): 0.85, ObjectiveName("time"): 90.0},
        )
        return board

    @pytest.fixture
    def populated_pareto_leaderboard(self, pareto_scorer: ObjectiveScorer) -> Leaderboard:
        """Leaderboard with samples for Pareto optimization."""
        board = Leaderboard(pareto_scorer)
        # Add non-dominated solutions
        board.add(
            params={ParameterName("lr"): 0.001},
            objectives={ObjectiveName("accuracy"): 0.90, ObjectiveName("time"): 120.0},
        )
        board.add(
            params={ParameterName("lr"): 0.01},
            objectives={ObjectiveName("accuracy"): 0.85, ObjectiveName("time"): 90.0},
        )
        # Add dominated solution
        board.add(
            params={ParameterName("lr"): 0.1},
            objectives={ObjectiveName("accuracy"): 0.82, ObjectiveName("time"): 150.0},
        )
        return board

    def test_initialization(self, simple_scorer: ObjectiveScorer) -> None:
        board = Leaderboard(simple_scorer)
        assert board.num_samples() == 0

    def test_add_sample(self, simple_scorer: ObjectiveScorer) -> None:
        board = Leaderboard(simple_scorer)
        board.add(params={ParameterName("lr"): 0.001}, objectives={ObjectiveName("accuracy"): 0.90})
        assert board.num_samples() == 1

    def test_get_sample(self, populated_leaderboard: Leaderboard) -> None:
        params, objectives = populated_leaderboard.get_sample(0)
        assert params[ParameterName("lr")] == 0.001
        assert objectives[ObjectiveName("accuracy")] == 0.90

    def test_get_invalid_sample(self, populated_leaderboard: Leaderboard) -> None:
        with pytest.raises(IndexError):
            populated_leaderboard.get_sample(10)

    def test_get_best_sample_single_objective(self, populated_leaderboard: Leaderboard) -> None:
        params, objectives = populated_leaderboard.get_best_sample()
        assert params[ParameterName("lr")] == 0.001  # Higher accuracy is better
        assert objectives[ObjectiveName("accuracy")] == 0.90

    def test_get_best_sample_dual_objective(self, populated_dual_leaderboard: Leaderboard) -> None:
        params, objectives = populated_dual_leaderboard.get_best_sample()
        # Should prioritize accuracy over time due to priority=2.0
        assert params[ParameterName("lr")] == 0.001
        assert objectives[ObjectiveName("accuracy")] == 0.90

    def test_get_top_samples_pareto(self, populated_pareto_leaderboard: Leaderboard) -> None:
        top_samples = populated_pareto_leaderboard.get_top_samples(2)
        # Should get non-dominated solutions first
        top_objectives = [obj for _, obj in top_samples]
        assert len(top_objectives) == 2
        assert any(
            obj[ObjectiveName("accuracy")] == 0.90 and obj[ObjectiveName("time")] == 120.0
            for obj in top_objectives
        )
        assert any(
            obj[ObjectiveName("accuracy")] == 0.85 and obj[ObjectiveName("time")] == 90.0
            for obj in top_objectives
        )

    def test_filter_samples(self, populated_leaderboard: Leaderboard) -> None:
        initial_count = populated_leaderboard.num_samples()
        populated_leaderboard.filter_samples([0])
        assert populated_leaderboard.num_samples() == 1
        params, objectives = populated_leaderboard.get_sample(0)
        assert params[ParameterName("lr")] == 0.001
        assert objectives[ObjectiveName("accuracy")] == 0.90

    def test_get_dataframe(self, populated_dual_leaderboard: Leaderboard) -> None:
        df = populated_dual_leaderboard.get_dataframe()

        # Check column prefixes
        assert any(col.startswith(PARAM_PREFIX) for col in df.columns)
        assert any(col.startswith(OBJECTIVE_PREFIX) for col in df.columns)
        assert any(col.startswith(GROUP_SCORE_PREFIX) for col in df.columns)

        # Check data preservation
        assert df[f"{PARAM_PREFIX}lr"].iloc[0] == 0.001
        assert df[f"{OBJECTIVE_PREFIX}accuracy"].iloc[0] == 0.90
        assert df[f"{OBJECTIVE_PREFIX}time"].iloc[0] == 120.0

    def test_save_load_roundtrip(
        self, populated_leaderboard: Leaderboard, simple_scorer: ObjectiveScorer, tmp_path: Path
    ) -> None:
        # Save
        save_path = tmp_path / "test_leaderboard.csv"
        populated_leaderboard.save(save_path)

        # Load
        loaded = Leaderboard.load(save_path, simple_scorer)
        assert loaded.num_samples() == populated_leaderboard.num_samples()

        # Compare samples
        orig_params, orig_obj = populated_leaderboard.get_sample(0)
        loaded_params, loaded_obj = loaded.get_sample(0)
        assert orig_params == loaded_params
        assert orig_obj == loaded_obj

    def test_create_single_objective(self) -> None:
        board = Leaderboard.create_single_objective(
            "accuracy",
            {
                "direction": "maximize",
                "target": 0.95,
                "limit": 0.80,
            },
        )
        assert isinstance(board, Leaderboard)
        board.add(params={ParameterName("lr"): 0.001}, objectives={ObjectiveName("accuracy"): 0.90})
        params, obj = board.get_best_sample()
        assert obj[ObjectiveName("accuracy")] == 0.90

    def test_create_dual_objective(self) -> None:
        board = Leaderboard.create_dual_objective(
            primary_name="accuracy",
            primary_config={
                "direction": "maximize",
                "target": 0.95,
                "limit": 0.80,
            },
            secondary_name="time",
            secondary_config={
                "direction": "minimize",
                "target": 60.0,
                "limit": 300.0,
            },
            primary_priority=2.0,
            secondary_priority=1.0,
        )
        assert isinstance(board, Leaderboard)
        board.add(
            params={ParameterName("lr"): 0.001},
            objectives={ObjectiveName("accuracy"): 0.90, ObjectiveName("time"): 120.0},
        )

    def test_create_pareto(self) -> None:
        board = Leaderboard.create_pareto(
            [
                (
                    "accuracy",
                    {
                        "direction": "maximize",
                        "target": 0.95,
                        "limit": 0.80,
                    },
                ),
                (
                    "time",
                    {
                        "direction": "minimize",
                        "target": 60.0,
                        "limit": 300.0,
                    },
                ),
            ]
        )
        assert isinstance(board, Leaderboard)
        board.add(
            params={ParameterName("lr"): 0.001},
            objectives={ObjectiveName("accuracy"): 0.90, ObjectiveName("time"): 120.0},
        )

    def test_add_missing_objective(self, simple_scorer: ObjectiveScorer) -> None:
        board = Leaderboard(simple_scorer)
        with pytest.raises(KeyError):
            board.add(params={ParameterName("lr"): 0.001}, objectives={ObjectiveName("invalid"): 0.90})

    def test_load_invalid_file(self, simple_scorer: ObjectiveScorer, tmp_path: Path) -> None:
        invalid_path = tmp_path / "invalid.csv"
        invalid_path.write_text("not,a,valid,csv")
        with pytest.raises(ValueError):
            Leaderboard.load(invalid_path, simple_scorer)


if __name__ == "__main__":
    pytest.main([__file__])
