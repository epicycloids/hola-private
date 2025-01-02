"""Tests for sorted population data structures."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from hola.core.sorted_population import ScalarSortedPopulation, Score, VectorSortedPopulation


class TestScalarSortedPopulation:
    @pytest.fixture
    def empty_population(self) -> ScalarSortedPopulation[str]:
        return ScalarSortedPopulation[str]()

    @pytest.fixture
    def populated_population(self) -> ScalarSortedPopulation[str]:
        pop = ScalarSortedPopulation[str]()
        pop.add("a", Score(np.array([0.5])))
        pop.add("b", Score(np.array([0.3])))
        pop.add("c", Score(np.array([0.8])))
        return pop

    def test_initialization(self, empty_population: ScalarSortedPopulation[str]) -> None:
        assert empty_population.num_samples() == 0
        assert empty_population.get_num_fronts() == 0

    def test_add_valid(self, empty_population: ScalarSortedPopulation[str]) -> None:
        empty_population.add("test", Score(np.array([0.5])))
        assert empty_population.num_samples() == 1
        assert_array_equal(empty_population.get_score("test"), np.array([0.5]))

    def test_add_duplicate_label(self, populated_population: ScalarSortedPopulation[str]) -> None:
        with pytest.raises(ValueError):
            populated_population.add("a", Score(np.array([0.1])))

    def test_add_invalid_score_dimension(
        self, empty_population: ScalarSortedPopulation[str]
    ) -> None:
        with pytest.raises(ValueError):
            empty_population.add("test", Score(np.array([0.5, 0.3])))

    def test_get_score_missing_label(self, empty_population: ScalarSortedPopulation[str]) -> None:
        with pytest.raises(KeyError):
            empty_population.get_score("missing")

    def test_update_score(self, populated_population: ScalarSortedPopulation[str]) -> None:
        populated_population.update_score("a", Score(np.array([0.1])))
        assert_array_equal(populated_population.get_score("a"), np.array([0.1]))
        assert populated_population.get_sorted_labels() == ["a", "b", "c"]

    def test_sorted_labels(self, populated_population: ScalarSortedPopulation[str]) -> None:
        assert populated_population.get_sorted_labels() == ["b", "a", "c"]

    def test_front_access(self, populated_population: ScalarSortedPopulation[str]) -> None:
        assert populated_population.get_front(0) == ["b"]
        assert populated_population.get_front(1) == ["a"]
        assert populated_population.get_front(2) == ["c"]
        with pytest.raises(ValueError):
            populated_population.get_front(3)

    def test_front_index(self, populated_population: ScalarSortedPopulation[str]) -> None:
        assert populated_population.get_front_index("b") == 0
        assert populated_population.get_front_index("a") == 1
        assert populated_population.get_front_index("c") == 2

    def test_crowding_distance(self, populated_population: ScalarSortedPopulation[str]) -> None:
        assert np.isinf(populated_population.get_crowding_distance("a"))
        assert np.isinf(populated_population.get_crowding_distance("b"))
        assert np.isinf(populated_population.get_crowding_distance("c"))

    def test_get_top_samples(self, populated_population: ScalarSortedPopulation[str]) -> None:
        assert populated_population.get_top_samples(2) == ["b", "a"]
        assert populated_population.get_top_samples(1) == ["b"]
        assert populated_population.get_top_samples(3) == ["b", "a", "c"]


class TestVectorSortedPopulation:
    @pytest.fixture
    def empty_population(self) -> VectorSortedPopulation[str]:
        return VectorSortedPopulation[str]()

    @pytest.fixture
    def simple_population(self) -> VectorSortedPopulation[str]:
        """Population with clear dominance relationships."""
        pop = VectorSortedPopulation[str]()
        # a dominates b
        pop.add("a", Score(np.array([0.1, 0.1])))
        pop.add("b", Score(np.array([0.2, 0.2])))
        return pop

    @pytest.fixture
    def pareto_population(self) -> VectorSortedPopulation[str]:
        """Population with multiple Pareto fronts."""
        pop = VectorSortedPopulation[str]()
        # Front 0: a and b are non-dominated
        pop.add("a", Score(np.array([0.1, 0.3])))
        pop.add("b", Score(np.array([0.3, 0.1])))
        # Front 1: c is dominated by both a and b
        pop.add("c", Score(np.array([0.4, 0.4])))
        return pop

    @pytest.fixture
    def complex_pareto_population(self) -> VectorSortedPopulation[str]:
        """Population with more complex Pareto front arrangement.

        Front 0: Three points forming clear extremes and middle:
        - a: Best in objective 1 (0.1, 0.5)
        - b: Middle point (0.2, 0.2)
        - c: Best in objective 2 (0.5, 0.1)

        Front 1: One point dominated by all front 0 points:
        - d: (0.6, 0.6)

        The crowding distances should be:
        - a: infinity (extreme point)
        - b: 2.0 (middle point)
        - c: infinity (extreme point)
        - d: infinity (only point in its front)
        """
        pop = VectorSortedPopulation[str]()
        pop.add("a", Score(np.array([0.1, 0.5])))  # Extreme point (best obj 1)
        pop.add("b", Score(np.array([0.2, 0.2])))  # Middle point
        pop.add("c", Score(np.array([0.5, 0.1])))  # Extreme point (best obj 2)
        pop.add("d", Score(np.array([0.6, 0.6])))  # Dominated point
        return pop

    def test_initialization(self, empty_population: VectorSortedPopulation[str]) -> None:
        assert empty_population.num_samples() == 0
        assert empty_population.get_num_fronts() == 0

    def test_add_valid(self, empty_population: VectorSortedPopulation[str]) -> None:
        empty_population.add("test", Score(np.array([0.5, 0.3])))
        assert empty_population.num_samples() == 1
        assert_array_equal(empty_population.get_score("test"), np.array([0.5, 0.3]))

    def test_add_duplicate_label(self, simple_population: VectorSortedPopulation[str]) -> None:
        with pytest.raises(ValueError):
            simple_population.add("a", Score(np.array([0.1, 0.1])))

    def test_add_inconsistent_dimensions(
        self, simple_population: VectorSortedPopulation[str]
    ) -> None:
        with pytest.raises(ValueError):
            simple_population.add("c", Score(np.array([0.1, 0.1, 0.1])))

    def test_dominates(self, simple_population: VectorSortedPopulation[str]) -> None:
        a_score = simple_population.get_score("a")
        b_score = simple_population.get_score("b")
        assert simple_population._dominates(a_score, b_score)
        assert not simple_population._dominates(b_score, a_score)

    def test_front_assignment_simple(self, simple_population: VectorSortedPopulation[str]) -> None:
        assert simple_population.get_front(0) == ["a"]
        assert simple_population.get_front(1) == ["b"]
        assert simple_population.get_num_fronts() == 2

    def test_front_assignment_pareto(self, pareto_population: VectorSortedPopulation[str]) -> None:
        front_0 = set(pareto_population.get_front(0))
        assert front_0 == {"a", "b"}
        assert pareto_population.get_front(1) == ["c"]
        assert pareto_population.get_num_fronts() == 2

    def test_front_index(self, pareto_population: VectorSortedPopulation[str]) -> None:
        assert pareto_population.get_front_index("a") == 0
        assert pareto_population.get_front_index("b") == 0
        assert pareto_population.get_front_index("c") == 1

    def test_crowding_distance(self, pareto_population: VectorSortedPopulation[str]) -> None:
        # Both extremes in front 0 should have infinite crowding distance
        assert np.isinf(pareto_population.get_crowding_distance("a"))
        assert np.isinf(pareto_population.get_crowding_distance("b"))
        # Single point in front 1 should have infinite crowding distance
        assert np.isinf(pareto_population.get_crowding_distance("c"))

    def test_complex_front_structure(
        self, complex_pareto_population: VectorSortedPopulation[str]
    ) -> None:
        # Verify front assignments
        front_0 = set(complex_pareto_population.get_front(0))
        assert front_0 == {"a", "b", "c"}
        assert complex_pareto_population.get_front(1) == ["d"]

        # Verify front indices
        assert complex_pareto_population.get_front_index("a") == 0
        assert complex_pareto_population.get_front_index("b") == 0
        assert complex_pareto_population.get_front_index("c") == 0
        assert complex_pareto_population.get_front_index("d") == 1

    def test_complex_crowding_distances(
        self, complex_pareto_population: VectorSortedPopulation[str]
    ) -> None:
        # Extreme points in first front should have infinite distance
        assert np.isinf(complex_pareto_population.get_crowding_distance("a"))
        assert np.isinf(complex_pareto_population.get_crowding_distance("c"))

        # Middle point should have finite distance of 2.0
        # (normalized distance of 1.0 in each objective)
        assert complex_pareto_population.get_crowding_distance("b") == pytest.approx(2.0)

        # Only point in second front should have infinite distance
        assert np.isinf(complex_pareto_population.get_crowding_distance("d"))

    def test_score_update_front_changes(
        self, pareto_population: VectorSortedPopulation[str]
    ) -> None:
        # Update c to dominate everyone
        pareto_population.update_score("c", Score(np.array([0.05, 0.05])))
        assert pareto_population.get_front_index("c") == 0
        assert pareto_population.get_front_index("a") == 1
        assert pareto_population.get_front_index("b") == 1

    def test_get_top_samples_respect_fronts(
        self, pareto_population: VectorSortedPopulation[str]
    ) -> None:
        # Should get both front 0 samples before front 1
        top_samples = pareto_population.get_top_samples(3)
        assert set(top_samples[:2]) == {"a", "b"}
        assert top_samples[2] == "c"

    def test_get_sorted_labels(self, pareto_population: VectorSortedPopulation[str]) -> None:
        sorted_labels = pareto_population.get_sorted_labels()
        # Front 0 samples should come before front 1
        assert set(sorted_labels[:2]) == {"a", "b"}
        assert sorted_labels[2] == "c"


if __name__ == "__main__":
    pytest.main([__file__])
