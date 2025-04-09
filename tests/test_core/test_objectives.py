import numpy as np
import pytest
import msgspec
import os
import tempfile
from typing import Any, Dict

from hola.core.objectives import (
    ObjectiveConfig,
    ObjectiveScorer,
    Direction,
)


# --- Test ObjectiveConfig ---

class TestObjectiveConfig:

    # Test valid configurations
    @pytest.mark.parametrize(
        "params",
        [
            {"target": 0.0, "limit": 1.0, "direction": Direction.MINIMIZE},
            {"target": 1.0, "limit": 0.0, "direction": Direction.MAXIMIZE},
            {"target": -1.0, "limit": 1.0, "direction": Direction.MINIMIZE},
            {"target": 1.0, "limit": -1.0, "direction": Direction.MAXIMIZE},
            {"target": 0, "limit": 100, "priority": 0.5, "comparison_group": 1},
        ]
    )
    def test_valid_config_creation(self, params):
        """Test that valid configurations can be created without error."""
        try:
            ObjectiveConfig(**params)
        except ValueError as e:
            pytest.fail(f"Valid config raised ValueError: {e}")

    # Test invalid configurations (__post_init__ validation)
    @pytest.mark.parametrize(
        "params, match_str",
        [
            ({"target": 1.0, "limit": 1.0}, "Limit .* cannot equal target"),
            ({"target": 1.0, "limit": 0.0, "direction": Direction.MINIMIZE}, "limit .* must be > target"),
            ({"target": 0.0, "limit": 1.0, "direction": Direction.MAXIMIZE}, "limit .* must be < target"),
            ({"target": 0, "limit": 1, "priority": 0.0}, "Priority should be positive"),
            ({"target": 0, "limit": 1, "priority": -0.5}, "Priority should be positive"),
            ({"target": 0, "limit": 1, "comparison_group": -1}, "Comparison group ID must be"),
        ]
    )
    def test_invalid_config_creation(self, params, match_str):
        """Test that invalid configurations raise ValueError on creation."""
        with pytest.raises(ValueError, match=match_str):
            ObjectiveConfig(**params)

    # Test score calculation - MINIMIZE
    @pytest.mark.parametrize(
        "target, limit, priority, value, expected_score",
        [
            # Target = 0, Limit = 10, Prio = 1
            (0.0, 10.0, 1.0, -1.0, 0.0),    # Below target
            (0.0, 10.0, 1.0, 0.0, 0.0),     # At target
            (0.0, 10.0, 1.0, 5.0, 0.5),     # Halfway
            (0.0, 10.0, 1.0, 10.0, float("inf")), # At limit
            (0.0, 10.0, 1.0, 11.0, float("inf")), # Above limit
            # Target = 10, Limit = 100, Prio = 2
            (10.0, 100.0, 2.0, 5.0, 0.0),    # Below target
            (10.0, 100.0, 2.0, 10.0, 0.0),   # At target
            (10.0, 100.0, 2.0, 55.0, 1.0),   # Halfway value -> Score = Prio * 0.5 = 1.0
            (10.0, 100.0, 2.0, 100.0, float("inf")),# At limit
            (10.0, 100.0, 2.0, 101.0, float("inf")),# Above limit
        ]
    )
    def test_score_minimize(self, target, limit, priority, value, expected_score):
        config = ObjectiveConfig(
            target=target, limit=limit, priority=priority, direction=Direction.MINIMIZE
        )
        assert config.score(value) == expected_score

    # Test score calculation - MAXIMIZE
    @pytest.mark.parametrize(
        "target, limit, priority, value, expected_score",
        [
            # Target = 10, Limit = 0, Prio = 1
            (10.0, 0.0, 1.0, 11.0, 0.0),    # Above target
            (10.0, 0.0, 1.0, 10.0, 0.0),    # At target
            (10.0, 0.0, 1.0, 5.0, 0.5),     # Halfway
            (10.0, 0.0, 1.0, 0.0, float("inf")),  # At limit
            (10.0, 0.0, 1.0, -1.0, float("inf")), # Below limit
            # Target = 0, Limit = -100, Prio = 0.5
            (0.0, -100.0, 0.5, 5.0, 0.0),    # Above target
            (0.0, -100.0, 0.5, 0.0, 0.0),    # At target
            (0.0, -100.0, 0.5, -50.0, 0.25), # Halfway value -> Score = Prio * 0.5 = 0.25
            (0.0, -100.0, 0.5, -100.0, float("inf")),# At limit
            (0.0, -100.0, 0.5, -101.0, float("inf")),# Below limit
        ]
    )
    def test_score_maximize(self, target, limit, priority, value, expected_score):
        config = ObjectiveConfig(
            target=target, limit=limit, priority=priority, direction=Direction.MAXIMIZE
        )
        assert config.score(value) == expected_score


# --- Test ObjectiveScorer ---

OBJECTIVES_DICT = {
    "loss": {"target": 0.1, "limit": 1.0, "direction": "minimize"},
    "accuracy": {"target": 0.95, "limit": 0.8, "direction": "maximize"},
    "latency": {"target": 50, "limit": 200, "direction": "minimize", "priority": 0.5},
}

OBJECTIVES_DICT_MULTI_GROUP = {
    "loss": {"target": 0.1, "limit": 1.0, "direction": "minimize", "comparison_group": 0},
    "accuracy": {"target": 0.95, "limit": 0.8, "direction": "maximize", "comparison_group": 0},
    "latency": {"target": 50, "limit": 200, "direction": "minimize", "priority": 0.5, "comparison_group": 1},
    "power": {"target": 10, "limit": 30, "direction": "minimize", "comparison_group": 1},
}

OBJECTIVES_JSON = msgspec.json.encode({"objectives": OBJECTIVES_DICT})
OBJECTIVES_MULTI_GROUP_JSON = msgspec.json.encode({"objectives": OBJECTIVES_DICT_MULTI_GROUP})

@pytest.fixture
def scorer_single_group() -> ObjectiveScorer:
    """Fixture for a scorer with objectives in a single group."""
    return ObjectiveScorer.from_dict(OBJECTIVES_DICT)

@pytest.fixture
def scorer_multi_group() -> ObjectiveScorer:
    """Fixture for a scorer with objectives in multiple groups."""
    return ObjectiveScorer.from_dict(OBJECTIVES_DICT_MULTI_GROUP)


class TestObjectiveScorer:

    def test_from_dict(self, scorer_single_group, scorer_multi_group):
        assert len(scorer_single_group.objectives) == 3
        assert "loss" in scorer_single_group.objectives
        assert scorer_single_group.objectives["loss"].target == 0.1
        assert scorer_single_group.objectives["accuracy"].direction == Direction.MAXIMIZE

        assert len(scorer_multi_group.objectives) == 4
        assert scorer_multi_group.objectives["latency"].comparison_group == 1

    def test_from_json(self):
        scorer_sg = ObjectiveScorer.from_json(OBJECTIVES_JSON)
        assert len(scorer_sg.objectives) == 3
        assert scorer_sg.objectives["loss"].target == 0.1

        scorer_mg = ObjectiveScorer.from_json(OBJECTIVES_MULTI_GROUP_JSON)
        assert len(scorer_mg.objectives) == 4
        assert scorer_mg.objectives["power"].comparison_group == 1

    def test_save_load_file(self, scorer_single_group, scorer_multi_group):
        for scorer in [scorer_single_group, scorer_multi_group]:
            with tempfile.TemporaryDirectory() as tmpdir:
                filepath = os.path.join(tmpdir, "objectives.json")
                scorer.save_to_file(filepath)
                assert os.path.exists(filepath)

                loaded_scorer = ObjectiveScorer.load_from_file(filepath)

                assert loaded_scorer.objectives == scorer.objectives
                assert loaded_scorer.group_ids == scorer.group_ids
                assert loaded_scorer.num_groups == scorer.num_groups

    def test_load_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            ObjectiveScorer.load_from_file("non_existent_obj.json")

    def test_post_init_validation(self):
        with pytest.raises(ValueError, match="At least one objective must be provided"):
            ObjectiveScorer(objectives={})

    def test_properties(self, scorer_single_group, scorer_multi_group):
        assert scorer_single_group.group_ids == {0}
        assert scorer_single_group.num_groups == 1
        assert not scorer_single_group.is_multigroup

        assert scorer_multi_group.group_ids == {0, 1}
        assert scorer_multi_group.num_groups == 2
        assert scorer_multi_group.is_multigroup

    # Test scoring - Single Group
    def test_score_single_group(self, scorer_single_group):
        # Case 1: All objectives meet target
        values1 = {"loss": 0.05, "accuracy": 0.98, "latency": 40}
        score1 = scorer_single_group.score(values1)
        assert isinstance(score1, float)
        assert score1 == 0.0

        # Case 2: Objectives partially met
        # loss: (0.5 - 0.1) / (1.0 - 0.1) * 1.0 = 0.4 / 0.9 ~= 0.4444
        # accuracy: (1 - (0.9 - 0.8) / (0.95 - 0.8)) * 1.0 = 1 - (0.1 / 0.15) = 1 - 2/3 ~= 0.3333
        # latency: (100 - 50) / (200 - 50) * 0.5 = 50 / 150 * 0.5 = 1/3 * 0.5 ~= 0.1667
        # Total = 0.4444 + 0.3333 + 0.1667 = 0.9444
        values2 = {"loss": 0.55, "accuracy": 0.90, "latency": 125}
        loss_score = (0.55 - 0.1) / (1.0 - 0.1)
        acc_score = 1.0 - (0.90 - 0.8) / (0.95 - 0.8)
        lat_score = (125 - 50) / (200 - 50) * 0.5
        expected_score2 = loss_score + acc_score + lat_score
        score2 = scorer_single_group.score(values2)
        assert isinstance(score2, float)
        assert np.isclose(score2, expected_score2)

        # Case 3: One objective exceeds limit
        values3 = {"loss": 1.1, "accuracy": 0.9, "latency": 100}
        score3 = scorer_single_group.score(values3)
        assert score3 == float("inf")

    # Test scoring - Multi Group
    def test_score_multi_group(self, scorer_multi_group):
        # Case 1: All meet target
        values1 = {"loss": 0.05, "accuracy": 0.96, "latency": 40, "power": 8}
        score1 = scorer_multi_group.score(values1)
        assert isinstance(score1, np.ndarray)
        np.testing.assert_array_equal(score1, [0.0, 0.0])

        # Case 2: Partial scores
        # Group 0: loss(0.55) + accuracy(0.9)
        # loss = (0.55 - 0.1)/(1.0 - 0.1) = 0.45 / 0.9 = 0.5
        # acc = (1 - (0.9 - 0.8)/(0.95 - 0.8)) = 1 - 0.1/0.15 = 1 - 2/3 = 1/3
        # Group 0 Score = 0.5 + 1/3 = 5/6
        # Group 1: latency(125) + power(20)
        # latency = (125 - 50)/(200 - 50) * 0.5 = 75/150 * 0.5 = 0.5 * 0.5 = 0.25
        # power = (20 - 10)/(30 - 10) * 1.0 = 10/20 = 0.5
        # Group 1 Score = 0.25 + 0.5 = 0.75
        values2 = {"loss": 0.55, "accuracy": 0.90, "latency": 125, "power": 20}
        expected_score2 = np.array([5/6, 0.75])
        score2 = scorer_multi_group.score(values2)
        assert isinstance(score2, np.ndarray)
        np.testing.assert_allclose(score2, expected_score2)

        # Case 3: One exceeds limit in Group 0
        values3 = {"loss": 1.1, "accuracy": 0.90, "latency": 125, "power": 20}
        score3 = scorer_multi_group.score(values3)
        assert isinstance(score3, np.ndarray)
        assert score3[0] == float("inf")
        assert np.isclose(score3[1], 0.75)

        # Case 4: One exceeds limit in Group 1
        values4 = {"loss": 0.55, "accuracy": 0.90, "latency": 125, "power": 35}
        score4 = scorer_multi_group.score(values4)
        assert isinstance(score4, np.ndarray)
        assert np.isclose(score4[0], 5/6)
        assert score4[1] == float("inf")

    # Test score error handling
    def test_score_mismatched_keys(self, scorer_single_group):
        # Missing key
        values_missing = {"loss": 0.5, "accuracy": 0.9}
        with pytest.raises(KeyError, match="Missing objectives: .*'latency'"):
            scorer_single_group.score(values_missing)

        # Extra key
        values_extra = {"loss": 0.5, "accuracy": 0.9, "latency": 100, "extra": 1}
        with pytest.raises(KeyError, match="Unexpected objectives: .*'extra'"):
            scorer_single_group.score(values_extra)