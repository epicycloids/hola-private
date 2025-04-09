import numpy as np
import pytest
import msgspec
import os
import tempfile

from hola.core.parameters import (
    ContinuousParameterConfig,
    CategoricalParameterConfig,
    IntegerParameterConfig,
    LatticeParameterConfig,
    ParameterTransformer,
    Scale,
    ParamInfeasibleError,
    PredefinedParameterConfig,
    BaseParameterConfig,
    PARAMETER_CLASSES,
    PARAMETER_TAGS
)
from hola.core.utils import MIN_FLOAT_TOLERANCE


# --- Test BaseParameterConfig & Shared Logic ---

@pytest.mark.parametrize(
    "config",
    [
        ContinuousParameterConfig(min=1.0, max=10.0, scale=Scale.LOG),
        CategoricalParameterConfig(categories=("cat1", "cat2")),
        IntegerParameterConfig(min=-5, max=5),
        LatticeParameterConfig(min=0.1, max=0.9, num_values=9),
    ]
)
def test_parameter_config_get_state_create_from_state(config):
    """Test state serialization and deserialization for all config types."""
    state = config.get_state()
    assert "type" in state
    assert state["type"] == PARAMETER_CLASSES[config.__class__.__name__]
    assert PARAMETER_TAGS[state["type"]] == config.__class__.__name__

    # Ensure BaseParameterConfig.create_from_state raises NotImplementedError
    with pytest.raises(NotImplementedError):
        BaseParameterConfig.create_from_state(state)

    # Test the actual subclass implementation
    recreated_config = config.__class__.create_from_state(state)
    assert recreated_config == config


# --- Test ContinuousParameterConfig ---

class TestContinuousParameterConfig:

    @pytest.mark.parametrize(
        "min_val, max_val, scale, u, expected",
        [
            (0.0, 10.0, Scale.LINEAR, 0.5, 5.0),
            (0.0, 10.0, Scale.LINEAR, 0.0, 0.0),
            (0.0, 10.0, Scale.LINEAR, 1.0, 10.0),
            (1.0, 100.0, Scale.LOG, 0.0, 1.0),
            (1.0, 100.0, Scale.LOG, 1.0, 100.0),
            (1.0, 100.0, Scale.LOG, 0.5, 10.0),
            (-10.0, 10.0, Scale.LINEAR, 0.25, -5.0),
        ],
    )
    def test_unnormalize_scalar(self, min_val, max_val, scale, u, expected):
        config = ContinuousParameterConfig(min=min_val, max=max_val, scale=scale)
        result = config.unnormalize(u)
        assert isinstance(result, float)
        assert np.isclose(result, expected, atol=MIN_FLOAT_TOLERANCE)

    @pytest.mark.parametrize(
        "min_val, max_val, scale, u, expected",
        [
            (0.0, 1.0, Scale.LINEAR, np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.5, 1.0])),
            (1.0, 10.0, Scale.LOG, np.array([0.0, 1.0]), np.array([1.0, 10.0])),
            (1.0, 100.0, Scale.LOG, np.array([0.0, 0.5, 1.0]), np.array([1.0, 10.0, 100.0])),
        ],
    )
    def test_unnormalize_array(self, min_val, max_val, scale, u, expected):
        config = ContinuousParameterConfig(min=min_val, max=max_val, scale=scale)
        result = config.unnormalize(u)
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, expected, atol=MIN_FLOAT_TOLERANCE)

    @pytest.mark.parametrize(
        "min_val, max_val, scale, x, expected",
        [
            (0.0, 10.0, Scale.LINEAR, 5.0, 0.5),
            (0.0, 10.0, Scale.LINEAR, 0.0, 0.0),
            (0.0, 10.0, Scale.LINEAR, 10.0, 1.0),
            (1.0, 100.0, Scale.LOG, 1.0, 0.0),
            (1.0, 100.0, Scale.LOG, 100.0, 1.0),
            (1.0, 100.0, Scale.LOG, 10.0, 0.5),
            (-10.0, 10.0, Scale.LINEAR, -5.0, 0.25),
        ],
    )
    def test_normalize_scalar(self, min_val, max_val, scale, x, expected):
        config = ContinuousParameterConfig(min=min_val, max=max_val, scale=scale)
        result = config.normalize(x)
        assert isinstance(result, float)
        assert np.isclose(result, expected, atol=MIN_FLOAT_TOLERANCE)

    @pytest.mark.parametrize(
        "min_val, max_val, scale, x, expected",
        [
            (0.0, 1.0, Scale.LINEAR, np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.5, 1.0])),
            (1.0, 10.0, Scale.LOG, np.array([1.0, 10.0]), np.array([0.0, 1.0])),
            (1.0, 100.0, Scale.LOG, np.array([1.0, 10.0, 100.0]), np.array([0.0, 0.5, 1.0])),
        ],
    )
    def test_normalize_array(self, min_val, max_val, scale, x, expected):
        config = ContinuousParameterConfig(min=min_val, max=max_val, scale=scale)
        result = config.normalize(x)
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, expected, atol=MIN_FLOAT_TOLERANCE)

    @pytest.mark.parametrize(
        "config_params, value, expected",
        [
            ({"min": 0.0, "max": 1.0}, 0.5, True),
            ({"min": 0.0, "max": 1.0}, 0.0, True),
            ({"min": 0.0, "max": 1.0}, 1.0, True),
            ({"min": 0.0, "max": 1.0}, -0.1, False),
            ({"min": 0.0, "max": 1.0}, 1.1, False),
            ({"min": 0.0, "max": 1.0}, np.array([0.1, 0.9]), True),
            ({"min": 0.0, "max": 1.0}, np.array([-0.1, 0.9]), False),
        ]
    )
    def test_is_feasible(self, config_params, value, expected):
        config = ContinuousParameterConfig(**config_params)
        assert config.is_feasible(value) == expected

    @pytest.mark.parametrize("u_invalid", [-0.1, 1.1, np.array([0.5, 1.2])])
    def test_unnormalize_invalid_input(self, u_invalid):
        config = ContinuousParameterConfig(min=0.0, max=1.0)
        expected_match = "All array values" if isinstance(u_invalid, np.ndarray) else "Value must be in"
        with pytest.raises(ValueError, match=expected_match):
            config.unnormalize(u_invalid)

    @pytest.mark.parametrize("x_infeasible", [-1.0, 11.0, np.array([5.0, 12.0])])
    def test_normalize_infeasible_input(self, x_infeasible):
        config = ContinuousParameterConfig(min=0.0, max=10.0)
        with pytest.raises(ParamInfeasibleError, match="Values outside bounds"):
            config.normalize(x_infeasible)

    def test_invalid_config(self):
        with pytest.raises(ValueError, match="max must be strictly greater than min"):
            ContinuousParameterConfig(min=1.0, max=0.0)
        with pytest.raises(ValueError, match="max must be strictly greater than min"):
            ContinuousParameterConfig(min=1.0, max=1.0)
        with pytest.raises(ValueError, match="log scale requires strictly positive bounds"):
            ContinuousParameterConfig(min=0.0, max=1.0, scale=Scale.LOG)
        with pytest.raises(ValueError, match="log scale requires strictly positive bounds"):
            ContinuousParameterConfig(min=-1.0, max=1.0, scale=Scale.LOG)

    @pytest.mark.parametrize(
        "config1_params, config2_params, expected",
        [
            ({"min": 0, "max": 10}, {"min": -1, "max": 10}, True),
            ({"min": 0, "max": 10}, {"min": 0, "max": 11}, True),
            ({"min": 0, "max": 10}, {"min": -1, "max": 11}, True),
            ({"min": 0, "max": 10}, {"min": 0, "max": 10}, False),
            ({"min": 0, "max": 10}, {"min": 1, "max": 9}, False),
            ({"min": 1, "max": 10, "scale": Scale.LINEAR}, {"min": 1, "max": 10, "scale": Scale.LOG}, True),
        ]
    )
    def test_has_expanded_domain(self, config1_params, config2_params, expected):
        config1 = ContinuousParameterConfig(**config1_params)
        config2 = ContinuousParameterConfig(**config2_params)
        assert config2.has_expanded_domain(config1) == expected

    def test_has_expanded_domain_type_mismatch(self):
        config1 = ContinuousParameterConfig(min=0, max=10)
        config2 = IntegerParameterConfig(min=0, max=10)
        with pytest.raises(TypeError):
            config1.has_expanded_domain(config2)


# --- Test CategoricalParameterConfig ---

CATEGORIES = ("apple", "banana", "cherry")
N_CATEGORIES = len(CATEGORIES)

class TestCategoricalParameterConfig:

    @pytest.mark.parametrize(
        "u, expected_category",
        [
            (0.0, "apple"),
            (0.1, "apple"),
            (1/3 - 1e-9, "apple"),
            (1/3, "banana"),
            (0.5, "banana"),
            (2/3 - 1e-9, "banana"),
            (2/3, "cherry"),
            (0.9, "cherry"),
            (1.0, "cherry"),
        ],
    )
    def test_unnormalize_scalar(self, u, expected_category):
        config = CategoricalParameterConfig(categories=CATEGORIES)
        result = config.unnormalize(u)
        assert result == expected_category

    def test_unnormalize_array(self):
        config = CategoricalParameterConfig(categories=CATEGORIES)
        u_array = np.array([0.1, 0.5, 0.9, 1.0])
        expected = np.array(["apple", "banana", "cherry", "cherry"])
        result = config.unnormalize(u_array)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize(
        "category, expected_u",
        [
            ("apple", (0 + 0.5) / N_CATEGORIES),
            ("banana", (1 + 0.5) / N_CATEGORIES),
            ("cherry", (2 + 0.5) / N_CATEGORIES),
        ],
    )
    def test_normalize_scalar(self, category, expected_u):
        config = CategoricalParameterConfig(categories=CATEGORIES)
        result = config.normalize(category)
        assert isinstance(result, float)
        assert np.isclose(result, expected_u, atol=MIN_FLOAT_TOLERANCE)

    def test_normalize_array(self):
        config = CategoricalParameterConfig(categories=CATEGORIES)
        categories_array = np.array(["apple", "cherry", "banana"])
        expected_u = np.array(
            [(0 + 0.5) / N_CATEGORIES, (2 + 0.5) / N_CATEGORIES, (1 + 0.5) / N_CATEGORIES]
        )
        result = config.normalize(categories_array)
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, expected_u, atol=MIN_FLOAT_TOLERANCE)

    @pytest.mark.parametrize(
        "value, expected",
        [
            ("apple", True),
            ("banana", True),
            ("grape", False),
            (123, False),
            (np.array(["apple", "cherry"]), True),
            (np.array(["apple", "grape"]), False),
        ]
    )
    def test_is_feasible(self, value, expected):
        config = CategoricalParameterConfig(categories=CATEGORIES)
        assert config.is_feasible(value) == expected

    @pytest.mark.parametrize("u_invalid", [-0.1, 1.1, np.array([0.5, 1.2])])
    def test_unnormalize_invalid_input(self, u_invalid):
        config = CategoricalParameterConfig(categories=CATEGORIES)
        expected_match = "All array values" if isinstance(u_invalid, np.ndarray) else "Value must be in"
        with pytest.raises(ValueError, match=expected_match):
            config.unnormalize(u_invalid)

    @pytest.mark.parametrize("x_infeasible", ["grape", np.array(["apple", "durian"])])
    def test_normalize_infeasible_input(self, x_infeasible):
        config = CategoricalParameterConfig(categories=CATEGORIES)
        with pytest.raises(ParamInfeasibleError, match="not in categories"):
            config.normalize(x_infeasible)

    def test_invalid_config(self):
        with pytest.raises(ValueError, match="`categories` cannot be empty"):
            CategoricalParameterConfig(categories=())

    @pytest.mark.parametrize(
        "config1_cats, config2_cats, expected_impl",
        [
            (("a", "b"), ("a", "b", "c"), True),
            (("a", "b", "c"), ("a", "b"), False),
            (("a", "b"), ("a", "b"), False),
            (("a", "b"), ("a", "c"), True),
        ]
    )
    def test_has_expanded_domain(self, config1_cats, config2_cats, expected_impl):
        config1 = CategoricalParameterConfig(categories=config1_cats)
        config2 = CategoricalParameterConfig(categories=config2_cats)
        assert config2.has_expanded_domain(config1) == expected_impl

    def test_has_expanded_domain_type_mismatch(self):
        config1 = CategoricalParameterConfig(categories=("a", "b"))
        config2 = IntegerParameterConfig(min=0, max=1)
        with pytest.raises(TypeError):
            config1.has_expanded_domain(config2)


# --- Test IntegerParameterConfig ---

INT_MIN, INT_MAX = 5, 10
NUM_INT_VALUES = INT_MAX - INT_MIN + 1

class TestIntegerParameterConfig:

    @pytest.mark.parametrize(
        "u, expected_int",
        [
            (0.0, 5),
            (0.1, 5),
            (1/6 - 1e-9, 5),
            (1/6, 6),
            (0.5, 8),
            (5/6 - 1e-9, 9),
            (5/6, 10),
            (0.9, 10),
            (1.0, 10),
        ],
    )
    def test_unnormalize_scalar(self, u, expected_int):
        config = IntegerParameterConfig(min=INT_MIN, max=INT_MAX)
        result = config.unnormalize(u)
        assert isinstance(result, int)
        assert result == expected_int

    def test_unnormalize_array(self):
        config = IntegerParameterConfig(min=INT_MIN, max=INT_MAX)
        u_array = np.array([0.1, 0.5, 0.9, 1.0])
        expected = np.array([5, 8, 10, 10])
        result = config.unnormalize(u_array)
        assert isinstance(result, np.ndarray)
        assert result.dtype == int
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize(
        "x_int, expected_u",
        [
            (5, (0 + 0.5) / NUM_INT_VALUES),
            (6, (1 + 0.5) / NUM_INT_VALUES),
            (8, (3 + 0.5) / NUM_INT_VALUES),
            (10, (5 + 0.5) / NUM_INT_VALUES),
        ],
    )
    def test_normalize_scalar(self, x_int, expected_u):
        config = IntegerParameterConfig(min=INT_MIN, max=INT_MAX)
        result = config.normalize(x_int)
        assert isinstance(result, float)
        assert np.isclose(result, expected_u, atol=MIN_FLOAT_TOLERANCE)

    def test_normalize_array(self):
        config = IntegerParameterConfig(min=INT_MIN, max=INT_MAX)
        x_array = np.array([5, 10, 8])
        expected_u = np.array(
            [(0 + 0.5) / NUM_INT_VALUES, (5 + 0.5) / NUM_INT_VALUES, (3 + 0.5) / NUM_INT_VALUES]
        )
        result = config.normalize(x_array)
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, expected_u, atol=MIN_FLOAT_TOLERANCE)

    @pytest.mark.parametrize(
        "value, expected",
        [
            (5, True),
            (8, True),
            (10, True),
            (4, False),
            (11, False),
            (7.1, True),
            (10.4, True),
            (4.8, True),
            (10.6, False),
            (np.array([5, 7, 10]), True),
            (np.array([5, 11]), False),
            (np.array([5.1, 9.9]), True),
            (np.array([4.4, 8]), False),
        ]
    )
    def test_is_feasible(self, value, expected):
        config = IntegerParameterConfig(min=INT_MIN, max=INT_MAX)
        assert config.is_feasible(value) == expected

    @pytest.mark.parametrize("u_invalid", [-0.1, 1.1, np.array([0.5, 1.2])])
    def test_unnormalize_invalid_input(self, u_invalid):
        config = IntegerParameterConfig(min=INT_MIN, max=INT_MAX)
        expected_match = "All array values" if isinstance(u_invalid, np.ndarray) else "Value must be in"
        with pytest.raises(ValueError, match=expected_match):
            config.unnormalize(u_invalid)

    @pytest.mark.parametrize("x_infeasible", [4, 11, np.array([5, 12])])
    def test_normalize_infeasible_input(self, x_infeasible):
        config = IntegerParameterConfig(min=INT_MIN, max=INT_MAX)
        with pytest.raises(ParamInfeasibleError, match="Values outside bounds"):
            config.normalize(x_infeasible)

    def test_invalid_config(self):
        with pytest.raises(ValueError, match="min .* must be <= max"):
            IntegerParameterConfig(min=10, max=5)

    @pytest.mark.parametrize(
        "config1_params, config2_params, expected",
        [
            ({"min": 5, "max": 10}, {"min": 4, "max": 10}, True),
            ({"min": 5, "max": 10}, {"min": 5, "max": 11}, True),
            ({"min": 5, "max": 10}, {"min": 4, "max": 11}, True),
            ({"min": 5, "max": 10}, {"min": 5, "max": 10}, False),
            ({"min": 5, "max": 10}, {"min": 6, "max": 9}, False),
        ]
    )
    def test_has_expanded_domain(self, config1_params, config2_params, expected):
        config1 = IntegerParameterConfig(**config1_params)
        config2 = IntegerParameterConfig(**config2_params)
        assert config2.has_expanded_domain(config1) == expected

    def test_has_expanded_domain_type_mismatch(self):
        config1 = IntegerParameterConfig(min=0, max=10)
        config2 = ContinuousParameterConfig(min=0, max=10)
        with pytest.raises(TypeError):
            config1.has_expanded_domain(config2)


# --- Test LatticeParameterConfig ---

LAT_MIN, LAT_MAX = 0.0, 1.0
NUM_LAT_VALUES = 5
LAT_STEP = (LAT_MAX - LAT_MIN) / (NUM_LAT_VALUES - 1)

class TestLatticeParameterConfig:

    @pytest.mark.parametrize(
        "u, expected_val",
        [
            (0.0, 0.0),
            (0.1, 0.0),
            (0.2 - 1e-9, 0.0),
            (0.2, 0.25),
            (0.3, 0.25),
            (0.5, 0.5),
            (0.7, 0.75),
            (0.9, 1.0),
            (1.0, 1.0),
        ],
    )
    def test_unnormalize_scalar(self, u, expected_val):
        config = LatticeParameterConfig(min=LAT_MIN, max=LAT_MAX, num_values=NUM_LAT_VALUES)
        result = config.unnormalize(u)
        assert isinstance(result, float)
        assert np.isclose(result, expected_val, atol=MIN_FLOAT_TOLERANCE)

    def test_unnormalize_array(self):
        config = LatticeParameterConfig(min=LAT_MIN, max=LAT_MAX, num_values=NUM_LAT_VALUES)
        u_array = np.array([0.1, 0.5, 0.9, 1.0])
        expected = np.array([0.0, 0.5, 1.0, 1.0])
        result = config.unnormalize(u_array)
        assert isinstance(result, np.ndarray)
        assert result.dtype == float
        np.testing.assert_allclose(result, expected, atol=MIN_FLOAT_TOLERANCE)

    @pytest.mark.parametrize(
        "x_val, expected_u",
        [
            (0.0, (0 + 0.5) / NUM_LAT_VALUES),
            (0.25, (1 + 0.5) / NUM_LAT_VALUES),
            (0.5, (2 + 0.5) / NUM_LAT_VALUES),
            (0.75, (3 + 0.5) / NUM_LAT_VALUES),
            (1.0, (4 + 0.5) / NUM_LAT_VALUES),
            (0.51, (2 + 0.5) / NUM_LAT_VALUES),
            (0.73, (3 + 0.5) / NUM_LAT_VALUES),
        ],
    )
    def test_normalize_scalar(self, x_val, expected_u):
        config = LatticeParameterConfig(min=LAT_MIN, max=LAT_MAX, num_values=NUM_LAT_VALUES)
        result = config.normalize(x_val)
        assert isinstance(result, float)
        assert np.isclose(result, expected_u, atol=MIN_FLOAT_TOLERANCE)

    def test_normalize_array(self):
        config = LatticeParameterConfig(min=LAT_MIN, max=LAT_MAX, num_values=NUM_LAT_VALUES)
        x_array = np.array([0.0, 1.0, 0.5, 0.26])
        expected_u = np.array([
            (0 + 0.5) / NUM_LAT_VALUES,
            (4 + 0.5) / NUM_LAT_VALUES,
            (2 + 0.5) / NUM_LAT_VALUES,
            (1 + 0.5) / NUM_LAT_VALUES,
        ])
        result = config.normalize(x_array)
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, expected_u, atol=MIN_FLOAT_TOLERANCE)

    @pytest.mark.parametrize(
        "value, expected",
        [
            (0.0, True),
            (0.5, True),
            (1.0, True),
            (0.25, True),
            (0.75, True),
            (0.1, False),
            (1.1, False),
            (-0.1, False),
            (0.50000001, True),
            (np.array([0.0, 0.5, 1.0]), True),
            (np.array([0.0, 0.1, 1.0]), False),
            (np.array([0.0, 1.1]), False),
        ]
    )
    def test_is_feasible(self, value, expected):
        config = LatticeParameterConfig(min=LAT_MIN, max=LAT_MAX, num_values=NUM_LAT_VALUES)
        assert config.is_feasible(value) == expected

    @pytest.mark.parametrize("u_invalid", [-0.1, 1.1, np.array([0.5, 1.2])])
    def test_unnormalize_invalid_input(self, u_invalid):
        config = LatticeParameterConfig(min=LAT_MIN, max=LAT_MAX, num_values=NUM_LAT_VALUES)
        expected_match = "All array values" if isinstance(u_invalid, np.ndarray) else "Value must be in"
        with pytest.raises(ValueError, match=expected_match):
            config.unnormalize(u_invalid)

    @pytest.mark.parametrize("x_infeasible", [-0.1, 1.1])
    def test_normalize_infeasible_input(self, x_infeasible):
        config = LatticeParameterConfig(min=LAT_MIN, max=LAT_MAX, num_values=NUM_LAT_VALUES)
        with pytest.raises(ParamInfeasibleError, match="Values outside bounds"):
            config.normalize(x_infeasible)

    def test_invalid_config(self):
        with pytest.raises(ValueError, match="max must be strictly greater than min"):
            LatticeParameterConfig(min=1.0, max=0.0, num_values=5)
        with pytest.raises(ValueError, match="max must be strictly greater than min"):
            LatticeParameterConfig(min=1.0, max=1.0, num_values=5)
        with pytest.raises(ValueError, match="num_values must be greater than 1"):
            LatticeParameterConfig(min=0.0, max=1.0, num_values=1)
        with pytest.raises(ValueError, match="num_values must be greater than 1"):
            LatticeParameterConfig(min=0.0, max=1.0, num_values=0)

    @pytest.mark.parametrize(
        "config1_params, config2_params, expected",
        [
            ({"min": 0, "max": 1, "num_values": 5}, {"min": -1, "max": 1, "num_values": 5}, True),
            ({"min": 0, "max": 1, "num_values": 5}, {"min": 0, "max": 2, "num_values": 5}, True),
            ({"min": 0, "max": 1, "num_values": 5}, {"min": 0, "max": 1, "num_values": 6}, True),
            ({"min": 0, "max": 1, "num_values": 5}, {"min": -1, "max": 2, "num_values": 6}, True),
            ({"min": 0, "max": 1, "num_values": 5}, {"min": 0, "max": 1, "num_values": 5}, False),
            ({"min": 0, "max": 1, "num_values": 5}, {"min": 0.1, "max": 0.9, "num_values": 4}, False),
        ]
    )
    def test_has_expanded_domain(self, config1_params, config2_params, expected):
        config1 = LatticeParameterConfig(**config1_params)
        config2 = LatticeParameterConfig(**config2_params)
        assert config2.has_expanded_domain(config1) == expected

    def test_has_expanded_domain_type_mismatch(self):
        config1 = LatticeParameterConfig(min=0, max=1, num_values=5)
        config2 = ContinuousParameterConfig(min=0, max=1)
        with pytest.raises(TypeError):
            config1.has_expanded_domain(config2)


# --- Test ParameterTransformer ---

PARAMS_DICT = {
    "lr": {"type": "continuous", "min": 1e-4, "max": 1e-1, "scale": "log"},
    "optimizer": {"type": "categorical", "categories": ["adam", "sgd"]},
    "layers": {"type": "integer", "min": 1, "max": 5},
    "dropout": {"type": "lattice", "min": 0.0, "max": 0.5, "num_values": 6}
}

PARAMS_DICT_JSON = msgspec.json.encode(PARAMS_DICT)

@pytest.fixture
def transformer_fixture() -> ParameterTransformer:
    """Fixture providing a ParameterTransformer instance for tests."""
    return ParameterTransformer.from_dict(PARAMS_DICT)


class TestParameterTransformer:

    def test_from_dict(self, transformer_fixture):
        transformer = transformer_fixture
        assert transformer.num_params == 4
        assert isinstance(transformer.parameters["lr"], ContinuousParameterConfig)
        assert isinstance(transformer.parameters["optimizer"], CategoricalParameterConfig)
        assert isinstance(transformer.parameters["layers"], IntegerParameterConfig)
        assert isinstance(transformer.parameters["dropout"], LatticeParameterConfig)
        assert transformer.param_names == ["lr", "optimizer", "layers", "dropout"]

    def test_from_json(self):
        transformer = ParameterTransformer.from_json(PARAMS_DICT_JSON)
        assert transformer.num_params == 4
        assert isinstance(transformer.parameters["lr"], ContinuousParameterConfig)
        assert transformer.parameters["optimizer"].categories == ("adam", "sgd")
        assert transformer.parameters["layers"].min == 1

    def test_save_load_file(self, transformer_fixture):
        transformer = transformer_fixture
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "params.json")
            transformer.save_to_file(filepath)
            assert os.path.exists(filepath)

            loaded_transformer = ParameterTransformer.load_from_file(filepath)

            assert loaded_transformer.num_params == transformer.num_params
            assert loaded_transformer.param_names == transformer.param_names
            assert type(loaded_transformer.parameters["lr"]) == type(transformer.parameters["lr"])
            assert loaded_transformer.parameters["lr"].min == transformer.parameters["lr"].min
            assert loaded_transformer.parameters["optimizer"].categories == transformer.parameters["optimizer"].categories

    def test_load_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            ParameterTransformer.load_from_file("non_existent_file.json")

    def test_unnormalize(self, transformer_fixture):
        transformer = transformer_fixture
        normalized_batch = np.array([
            [0.0, 0.2, 0.0, 0.0],
            [1.0, 0.8, 1.0, 1.0],
            [0.5, 0.6, 0.5, 0.5],
        ])

        expected_unnormalized = [
            {"lr": 1e-4, "optimizer": "adam", "layers": 1, "dropout": 0.0},
            {"lr": 1e-1, "optimizer": "sgd", "layers": 5, "dropout": 0.5},
            {"lr": pytest.approx(np.sqrt(1e-4 * 1e-1)), "optimizer": "sgd", "layers": 3, "dropout": pytest.approx(0.3)},
        ]

        result = transformer.unnormalize(normalized_batch)

        assert len(result) == 3
        for res_dict, exp_dict in zip(result, expected_unnormalized):
            assert res_dict.keys() == exp_dict.keys()
            assert res_dict["lr"] == pytest.approx(exp_dict["lr"])
            assert res_dict["optimizer"] == exp_dict["optimizer"]
            assert res_dict["layers"] == exp_dict["layers"]
            assert res_dict["dropout"] == pytest.approx(exp_dict["dropout"])

    def test_normalize(self, transformer_fixture):
        transformer = transformer_fixture
        unnormalized_batch = [
            {"lr": 1e-4, "optimizer": "adam", "layers": 1, "dropout": 0.0},
            {"lr": 1e-1, "optimizer": "sgd", "layers": 5, "dropout": 0.5},
            {"lr": np.sqrt(1e-4 * 1e-1), "optimizer": "sgd", "layers": 3, "dropout": 0.2},
        ]

        expected_normalized = np.array([
            [0.0, 0.25, 0.1, 1/12],
            [1.0, 0.75, 0.9, 11/12],
            [0.5, 0.75, 0.5, 5/12],
        ])

        result = transformer.normalize(unnormalized_batch)

        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 4)
        np.testing.assert_allclose(result, expected_normalized, atol=MIN_FLOAT_TOLERANCE)

    def test_unnormalize_invalid_shape(self, transformer_fixture):
        transformer = transformer_fixture
        with pytest.raises(ValueError, match="Expected .* array, got"):
            transformer.unnormalize(np.array([0.1, 0.2, 0.3, 0.4]))
        with pytest.raises(ValueError, match="Expected .* array, got"):
            transformer.unnormalize(np.array([[0.1, 0.2, 0.3]]))

    def test_normalize_empty_list(self, transformer_fixture):
        transformer = transformer_fixture
        with pytest.raises(ValueError, match="Empty parameter list"):
            transformer.normalize([])

    def test_normalize_mismatched_keys(self, transformer_fixture):
        transformer = transformer_fixture
        bad_batch_missing = [
            {"lr": 1e-3, "optimizer": "adam", "layers": 2, "dropout": 0.1},
            {"lr": 1e-2, "optimizer": "sgd", "layers": 4},
        ]
        with pytest.raises(ValueError, match="Param names in sample 1 do not match"):
            transformer.normalize(bad_batch_missing)

        bad_batch_extra = [
            {"lr": 1e-3, "optimizer": "adam", "layers": 2, "dropout": 0.1, "extra": True},
        ]
        with pytest.raises(ValueError, match="Parameter names do not match configuration"):
            transformer.normalize(bad_batch_extra)

        bad_batch_misspelled = [
            {"lr": 1e-3, "optimizer": "adam", "layers": 2, "dropout": 0.1},
            {"lr": 1e-2, "optimizer": "sgd", "layers": 4, "droput": 0.3},
        ]
        with pytest.raises(ValueError, match="Param names in sample 1 do not match"):
            transformer.normalize(bad_batch_misspelled)

    @pytest.mark.parametrize(
        "param_dict, expected",
        [
            ({"lr": 1e-3, "optimizer": "adam", "layers": 2, "dropout": 0.1}, True),
            ({"lr": 1e-5, "optimizer": "adam", "layers": 2, "dropout": 0.1}, False),
            ({"lr": 1e-3, "optimizer": "rmsprop", "layers": 2, "dropout": 0.1}, False),
            ({"lr": 1e-3, "optimizer": "adam", "layers": 6, "dropout": 0.1}, False),
            ({"lr": 1e-3, "optimizer": "adam", "layers": 2, "dropout": 0.15}, False),
            ({"lr": 1e-3, "optimizer": "adam", "layers": 2}, False),
            ({"lr": 1e-3, "optimizer": "adam", "layers": 2, "dropout": 0.1, "extra": 1}, False),
        ]
    )
    def test_is_feasible(self, transformer_fixture, param_dict, expected):
        transformer = transformer_fixture
        assert transformer.is_feasible(param_dict) == expected

    def test_has_expanded_domain(self, transformer_fixture):
        transformer = transformer_fixture
        transformer_same = ParameterTransformer.from_dict(PARAMS_DICT)
        assert not transformer.has_expanded_domain(transformer_same)

        params_expanded_lr = PARAMS_DICT.copy()
        params_expanded_lr["lr"] = {"type": "continuous", "min": 1e-4, "max": 1e-0, "scale": "log"}
        transformer_expanded_lr = ParameterTransformer.from_dict(params_expanded_lr)
        assert transformer_expanded_lr.has_expanded_domain(transformer)
        assert not transformer.has_expanded_domain(transformer_expanded_lr)

        params_expanded_opt = PARAMS_DICT.copy()
        params_expanded_opt["optimizer"] = {"type": "categorical", "categories": ["adam", "sgd", "rmsprop"]}
        transformer_expanded_opt = ParameterTransformer.from_dict(params_expanded_opt)
        assert transformer_expanded_opt.has_expanded_domain(transformer)
        assert not transformer.has_expanded_domain(transformer_expanded_opt)

        params_shrunk_layers = PARAMS_DICT.copy()
        params_shrunk_layers["layers"] = {"type": "integer", "min": 2, "max": 4}
        transformer_shrunk_layers = ParameterTransformer.from_dict(params_shrunk_layers)
        assert not transformer_shrunk_layers.has_expanded_domain(transformer)
        assert transformer.has_expanded_domain(transformer_shrunk_layers)

        params_changed_dropout = PARAMS_DICT.copy()
        params_changed_dropout["dropout"] = {"type": "lattice", "min": 0.0, "max": 0.5, "num_values": 11}
        transformer_changed_dropout = ParameterTransformer.from_dict(params_changed_dropout)
        assert transformer_changed_dropout.has_expanded_domain(transformer)
        assert not transformer.has_expanded_domain(transformer_changed_dropout)

    def test_has_expanded_domain_mismatched_params(self, transformer_fixture):
        transformer = transformer_fixture
        params_missing = PARAMS_DICT.copy()
        del params_missing["dropout"]
        transformer_missing = ParameterTransformer.from_dict(params_missing)
        with pytest.raises(ValueError, match="Parameter sets do not match"):
            transformer.has_expanded_domain(transformer_missing)

        params_extra = PARAMS_DICT.copy()
        params_extra["new_param"] = {"type": "integer", "min": 1, "max": 2}
        transformer_extra = ParameterTransformer.from_dict(params_extra)
        with pytest.raises(ValueError, match="Parameter sets do not match"):
            transformer.has_expanded_domain(transformer_extra)

    def test_has_expanded_domain_mismatched_types(self, transformer_fixture):
        transformer = transformer_fixture
        params_wrong_type = PARAMS_DICT.copy()
        params_wrong_type["layers"] = {"type": "continuous", "min": 1.0, "max": 5.0}
        transformer_wrong_type = ParameterTransformer.from_dict(params_wrong_type)
        with pytest.raises(TypeError, match="Param layers type mismatch"):
            transformer.has_expanded_domain(transformer_wrong_type)


# --- Test msgspec Union Decoding ---

def test_msgspec_predefined_config_union():
    """Test that msgspec correctly decodes into the Union type."""
    config_dict_cont = {"type": "continuous", "min": 0, "max": 1}
    decoded_cont = msgspec.msgpack.decode(msgspec.msgpack.encode(config_dict_cont), type=PredefinedParameterConfig)
    assert isinstance(decoded_cont, ContinuousParameterConfig)
    assert decoded_cont.min == 0 and decoded_cont.max == 1

    config_dict_cat = {"type": "categorical", "categories": ["a", "b"]}
    decoded_cat = msgspec.msgpack.decode(msgspec.msgpack.encode(config_dict_cat), type=PredefinedParameterConfig)
    assert isinstance(decoded_cat, CategoricalParameterConfig)
    assert decoded_cat.categories == ("a", "b")

    # Test decoding within the transformer context (implicitly tested by transformer tests)
    params_mixed = {
        "p1": {"type": "continuous", "min": 0, "max": 1},
        "p2": {"type": "categorical", "categories": ["a", "b"]}
    }
    transformer = ParameterTransformer.from_dict(params_mixed)
    assert isinstance(transformer.parameters["p1"], ContinuousParameterConfig)
    assert isinstance(transformer.parameters["p2"], CategoricalParameterConfig)