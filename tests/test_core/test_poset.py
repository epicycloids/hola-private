import numpy as np
import pytest

from hola.core.poset import ScalarPoset, VectorPoset


class TestScalarPoset:
    """
    Tests for ScalarPoset (totally ordered values).
    """

    def test_empty_poset(self):
        """
        Ensure a newly created poset is empty, and poptopitem() returns None.
        """
        poset = ScalarPoset()
        assert len(poset) == 0
        assert poset.poptopitem() is None

    def test_add_and_retrieve(self):
        """
        Check adding new entries, retrieving them, and that keys are in ascending
        order of their scalar values.
        """
        poset = ScalarPoset()
        poset.add("a", 5.0)
        poset.add("b", 3.0)
        poset.add("c", 7.0)

        assert len(poset) == 3
        assert poset["a"] == 5.0
        assert list(poset.keys()) == ["b", "a", "c"]  # sorted by value

    def test_pop_operations(self):
        """
        Test poptopitem() for removing the best element, and pop(key) for
        removing a specific element.
        """
        poset = ScalarPoset()
        poset.add("x", 2.0)
        poset.add("y", 1.0)

        # poptopitem() should remove the smallest (best) score
        assert poset.poptopitem() == ("y", 1.0)

        # pop(key) removes a specific entry
        assert poset.pop("x") == 2.0
        assert len(poset) == 0

    def test_fronts(self):
        """
        Test grouping by identical scores via fronts().
        """
        poset = ScalarPoset()
        poset.add("a", 1.0)
        poset.add("b", 2.0)
        poset.add("c", 2.0)  # Same as 'b'

        fronts = list(poset.fronts())
        assert len(fronts) == 2
        # First front has 1.0
        assert fronts[0] == [("a", 1.0)]
        # Second front has items with 2.0
        assert set(fronts[1]) == {("b", 2.0), ("c", 2.0)}

    def test_error_handling(self):
        """
        Test adding a duplicate key or a non-scalar value raises ValueError.
        """
        poset = ScalarPoset()
        poset.add("valid", 5.0)

        with pytest.raises(ValueError):
            poset.add("valid", 6.0)  # Duplicate key

        with pytest.raises(ValueError):
            poset.add("array", np.array([1, 2]))  # Non-scalar

    def test_peek(self):
        """
        Test peek(k) returns the top-k in ascending order of scores,
        without removing them.
        """
        poset = ScalarPoset()
        poset.add("a", 2.0)
        poset.add("b", 1.0)

        # Peek 1 => smallest item
        assert poset.peek(1) == [("b", 1.0)]
        assert len(poset) == 2  # not removed

        # Peek 2 => both items, sorted
        assert poset.peek(2) == [("b", 1.0), ("a", 2.0)]

        with pytest.raises(ValueError):
            poset.peek(0)  # invalid k


class TestVectorPoset:
    """
    Tests for VectorPoset (partially ordered values).
    """

    @pytest.fixture
    def simple_fronts(self):
        """
        Provide a set of 2D scores forming two Pareto fronts.
        """
        return {
            "a": [1, 5],  # Front 0
            "b": [2, 4],  # Front 0
            "c": [5, 5],  # Front 1 (dominated by a/b)
            "d": [3, 3],  # Front 0
        }

    def test_getitem(self):
        """
        Test that items can be retrieved by direct indexing.
        """
        poset = VectorPoset()
        poset.add("test", [1.0, 2.0])
        assert np.allclose(poset["test"], [1.0, 2.0])

    def test_basic_dominance(self, simple_fronts: dict[str, list[int]]):
        """
        'a', 'b', 'd' should be in the first front, 'c' in the second.
        """
        poset = VectorPoset()
        for k, v in simple_fronts.items():
            poset.add(k, v)

        fronts = list(poset.fronts())
        assert len(fronts) == 2

        front0_keys = {key for key, _ in fronts[0]}
        front1_keys = {key for key, _ in fronts[1]}
        assert front0_keys == {"a", "b", "d"}
        assert front1_keys == {"c"}

    def test_crowding_distance(self):
        """
        Boundary items get infinite crowding distance, interior items do not.
        """
        poset = VectorPoset()
        poset.add("left", [1, 3])
        poset.add("right", [5, 1])
        poset.add("middle", [3, 2])

        # left, right => boundaries => infinite
        assert poset.get_crowding_distance("left") == float("inf")
        assert poset.get_crowding_distance("right") == float("inf")

        # middle => finite
        assert poset.get_crowding_distance("middle") < float("inf")

    def test_pop_operations(self, simple_fronts: dict[str, list[int]]):
        """
        pop() removes a specific key, poptopitem() removes highest-priority item
        (lowest front, then highest crowding distance).
        """
        poset = VectorPoset()
        for k, v in simple_fronts.items():
            poset.add(k, v)

        # Remove a single known key
        removed = poset.pop("c")
        np.testing.assert_allclose(removed, [5, 5])
        assert len(poset) == 3

        # Now pop the 'best' item from front 0
        top_item = poset.poptopitem()
        assert top_item is not None
        # Should be one of the remaining {a, b, d}, typically a boundary item
        assert top_item[0] in ["a", "b", "d"]
        assert len(poset) == 2

    def test_identical_vectors(self):
        """
        Identical vectors share a front and typically get infinite distance.
        """
        poset = VectorPoset()
        poset.add("a", [2, 2])
        poset.add("b", [2, 2])

        fronts = list(poset.fronts())
        assert len(fronts) == 1
        front0_keys = {key for key, _ in fronts[0]}
        assert front0_keys == {"a", "b"}
        assert poset.get_crowding_distance("a") == float("inf")
        assert poset.get_crowding_distance("b") == float("inf")

    def test_single_element(self):
        """
        A single element is trivially in its own front, and poptopitem() removes it.
        """
        poset = VectorPoset()
        poset.add("lonely", [10, 20])
        top = poset.poptopitem()
        assert top is not None
        assert top[0] == "lonely"
        np.testing.assert_allclose(top[1], [10.0, 20.0])
        assert len(poset) == 0

    def test_peek(self, simple_fronts: dict[str, list[int]]):
        """
        peek(k) returns up to k items from the highest-priority fronts
        in crowding order, without removal.
        """
        poset = VectorPoset()
        for k, v in simple_fronts.items():
            poset.add(k, v)

        # "a", "b", "d" are in front 0, "c" in front 1
        top2 = poset.peek(2)
        assert len(top2) == 2
        # Not removed from poset
        assert len(poset) == 4

        with pytest.raises(ValueError):
            poset.peek(0)

    def test_items_and_values(self):
        """
        items() and values() provide iterators over the items and values
        (respectively) in the poset. They should iterate in order of decreasing
        priority.
        """
        poset = VectorPoset()
        poset.add("p", [1.0, 2.0])
        poset.add("q", [2.0, 1.0])
        poset.add("r", [1.5, 1.5])

        # Because crowding distance might reorder them, we just check membership:
        k_list = list(poset.keys())
        assert set(k_list) == {"p", "q", "r"}

        # Similarly with values()
        v_list = list(poset.values())
        assert any(np.allclose(v, [1.0, 2.0]) for v in v_list)
        assert any(np.allclose(v, [2.0, 1.0]) for v in v_list)
        assert np.allclose(v_list[-1], [1.5, 1.5])

        # items() similarly
        i_list = list(poset.items())
        assert len(i_list) == 3

    def test_items_with_distances(self):
        """
        items_with_distances() should iterate over the items and tag each with
        its crowding distance
        """
        poset = VectorPoset()
        poset.add("r", [1, 3])
        poset.add("s", [5, 1])
        poset.add("t", [2, 2])
        results = list(poset.items_with_distances())

        # Each entry is (key, vector, distance). Check that we have the keys
        keys = {r[0] for r in results}
        assert keys == {"r", "s", "t"}
        assert results[0][-1] == float("inf")
        assert results[1][-1] == float("inf")
        assert results[2][-1] < float("inf")
