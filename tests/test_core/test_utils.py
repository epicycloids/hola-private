import numpy as np
import pytest

from hola.core.utils import MIN_FLOAT_TOLERANCE, uniform_to_category


def test_min_float_tolerance():
    """
    Ensure the global floating-point tolerance is set to a positive,
    sensible default.
    """
    assert MIN_FLOAT_TOLERANCE == 1e-6
    assert MIN_FLOAT_TOLERANCE > 0, "Tolerance should be positive."


@pytest.mark.parametrize(
    "u_sample, n_categories, expected",
    [
        (0.0, 3, 0),  # Minimum boundary
        (0.999, 3, 2),  # High but under 1.0
        (1.0, 3, 2),  # Exactly 1.0 maps to the last category
        (0.7, 3, 2),  # General case
        (0.3, 4, 1),  # Another general case
    ],
)
def test_uniform_to_category_scalar(u_sample, n_categories, expected):
    """
    Check uniform_to_category correctness for scalar inputs within [0,1].
    """
    result = uniform_to_category(u_sample, n_categories)
    assert isinstance(result, np.int64), "Scalar input should return np.int64."
    assert result == expected


@pytest.mark.parametrize(
    "u_samples, n_categories, expected",
    [
        ([0.1, 0.5, 0.9], 4, [0, 2, 3]),
        ([0.0, 0.999, 1.0], 2, [0, 1, 1]),
    ],
)
def test_uniform_to_category_array(u_samples, n_categories, expected):
    """
    Check uniform_to_category correctness for array inputs within [0,1].
    """
    u_array = np.array(u_samples)
    result = uniform_to_category(u_array, n_categories)
    assert isinstance(result, np.ndarray), "Array input should return a NumPy array."
    assert result.dtype == np.int64, "Output array should be of integer dtype."
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "u_sample, n_categories",
    [
        (-0.1, 3),  # Negative sample
        (1.1, 2),  # Sample above 1
    ],
)
def test_uniform_to_category_scalar_out_of_bounds(u_sample, n_categories):
    """
    Test that uniform_to_category raises ValueError for out-of-range scalar.
    """
    with pytest.raises(ValueError, match="u_sample must be in the range"):
        uniform_to_category(u_sample, n_categories)


def test_uniform_to_category_array_out_of_bounds():
    """
    Test that uniform_to_category raises ValueError if any array element is out of range.
    """
    invalid_samples = np.array([0.0, 0.5, 1.1])
    with pytest.raises(ValueError, match="u_sample must be in the range"):
        uniform_to_category(invalid_samples, 3)


def test_uniform_to_category_zero_or_negative_categories():
    """
    Test that uniform_to_category raises ValueError for non-positive n_categories.
    """
    with pytest.raises(ValueError, match="n_categories must be positive"):
        uniform_to_category(0.5, 0)
    with pytest.raises(ValueError, match="n_categories must be positive"):
        uniform_to_category(0.5, -2)
