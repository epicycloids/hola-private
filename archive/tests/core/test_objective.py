"""Tests for objective configuration and scalarization utilities."""

import numpy as np
import pytest
from pydantic import ValidationError

from hola.core.objective import (
    Direction,
    ObjectiveConfig,
    ObjectiveName,
    ObjectiveScorer,
    create_objective,
)


class TestObjectiveConfig:
    @pytest.mark.parametrize(
        "target, limit, direction, priority, group",
        [
            (0.95, 0.80, Direction.MAXIMIZE, 1.0, 0),  # Standard maximize case
            (60.0, 300.0, Direction.MINIMIZE, 2.0, 1),  # Standard minimize case
            (0.0, 1.0, Direction.MINIMIZE, 0.1, 0),  # Edge case with zero target
        ],
    )
    def test_instantiation_valid(
        self,
        target: float,
        limit: float,
        direction: Direction,
        priority: float,
        group: int,
    ) -> None:
        config = ObjectiveConfig(
            target=target,
            limit=limit,
            direction=direction,
            priority=priority,
            comparison_group=group,
        )
        assert config.target == target
        assert config.limit == limit
        assert config.direction == direction
        assert config.priority == priority
        assert config.comparison_group == group

    @pytest.mark.parametrize(
        "target, limit, direction",
        [
            (0.8, 0.9, Direction.MAXIMIZE),  # Invalid target/limit for maximize
            (0.9, 0.8, Direction.MINIMIZE),  # Invalid target/limit for minimize
            (float("nan"), 1.0, Direction.MAXIMIZE),  # NaN target
            (0.5, float("nan"), Direction.MINIMIZE),  # NaN limit
        ],
    )
    def test_instantiation_invalid(self, target: float, limit: float, direction: Direction) -> None:
        with pytest.raises(ValidationError):
            ObjectiveConfig(target=target, limit=limit, direction=direction)

    @pytest.mark.parametrize(
        "value, expected_score",
        [
            (1.0, 0.0),  # Exactly at target (maximize)
            (1.1, 0.0),  # Better than target
            (0.9, 0.5),  # Between target and limit
            (0.75, float("inf")),  # Worse than limit
        ],
    )
    def test_score_maximize(self, value: float, expected_score: float) -> None:
        config = ObjectiveConfig(target=1.0, limit=0.80, direction=Direction.MAXIMIZE)
        score = config.score(value)
        if isinstance(expected_score, float) and expected_score == float("inf"):
            assert score == float("inf")
        else:
            assert score == pytest.approx(expected_score)

    @pytest.mark.parametrize(
        "value, expected_score",
        [
            (60.0, 0.0),  # Exactly at target (minimize)
            (30.0, 0.0),  # Better than target
            (180.0, 0.5),  # Between target and limit
            (350.0, float("inf")),  # Worse than limit
        ],
    )
    def test_score_minimize(self, value: float, expected_score: float) -> None:
        config = ObjectiveConfig(target=60.0, limit=300.0, direction=Direction.MINIMIZE)
        score = config.score(value)
        if isinstance(expected_score, float) and expected_score == float("inf"):
            assert score == float("inf")
        else:
            assert score == pytest.approx(expected_score)

    def test_with_priority(self) -> None:
        original = ObjectiveConfig(target=0.95, limit=0.80, direction=Direction.MAXIMIZE)
        modified = original.with_priority(2.0)
        assert modified.priority == 2.0
        assert original.priority == 1.0  # Original unchanged
        assert modified.target == original.target  # Other attributes preserved

    def test_with_comparison_group(self) -> None:
        original = ObjectiveConfig(target=0.95, limit=0.80, direction=Direction.MAXIMIZE)
        modified = original.with_comparison_group(2)
        assert modified.comparison_group == 2
        assert original.comparison_group == 0  # Original unchanged
        assert modified.target == original.target  # Other attributes preserved


def test_create_objective() -> None:
    obj = create_objective("maximize", target=0.95, limit=0.80, priority=2.0)
    assert isinstance(obj, ObjectiveConfig)
    assert obj.direction == Direction.MAXIMIZE
    assert obj.target == 0.95
    assert obj.limit == 0.80
    assert obj.priority == 2.0

    with pytest.raises(ValueError):
        create_objective("invalid_direction", target=0.95, limit=0.80)


class TestObjectiveScorer:
    @pytest.fixture
    def example_objectives(self) -> dict[ObjectiveName, ObjectiveConfig]:
        return {
            ObjectiveName("accuracy"): create_objective(
                "maximize", target=0.95, limit=0.80, priority=2.0
            ),
            ObjectiveName("time"): create_objective(
                "minimize", target=60.0, limit=300.0, priority=1.0
            ),
        }

    def test_init(self, example_objectives: dict[ObjectiveName, ObjectiveConfig]) -> None:
        scorer = ObjectiveScorer(example_objectives)
        assert scorer.multigroup is False

    def test_score_objectives_single(
        self, example_objectives: dict[ObjectiveName, ObjectiveConfig]
    ) -> None:
        scorer = ObjectiveScorer(example_objectives)
        scores = scorer.score_objectives(
            {
                ObjectiveName("accuracy"): 0.90,
                ObjectiveName("time"): 120.0,
            }
        )
        assert isinstance(scores, np.ndarray)
        assert len(scores) == 1
        assert scores[0] > 0.0  # Non-zero score for non-optimal values
        assert isinstance(scores[0], float)

    def test_score_objectives_multigroup(self) -> None:
        objectives = {
            ObjectiveName("acc_1"): create_objective(
                "maximize", target=0.95, limit=0.80, priority=1.0, comparison_group=0
            ),
            ObjectiveName("acc_2"): create_objective(
                "maximize", target=0.90, limit=0.75, priority=1.0, comparison_group=1
            ),
        }
        scorer = ObjectiveScorer(objectives)
        assert scorer.multigroup is True

        scores = scorer.score_objectives(
            {
                ObjectiveName("acc_1"): 0.85,
                ObjectiveName("acc_2"): 0.80,
            }
        )
        assert isinstance(scores, np.ndarray)
        assert len(scores) == 2  # One score per comparison group
        assert all(isinstance(score, float) for score in scores)

    def test_score_objectives_invalid_keys(
        self, example_objectives: dict[ObjectiveName, ObjectiveConfig]
    ) -> None:
        scorer = ObjectiveScorer(example_objectives)
        with pytest.raises(KeyError):
            scorer.score_objectives(
                {
                    ObjectiveName("invalid"): 0.90,
                    ObjectiveName("time"): 120.0,
                }
            )


if __name__ == "__main__":
    pytest.main([__file__])
