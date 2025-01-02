"""Tests for parameter configuration and transformation utilities."""

from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from hola.core.params import (
    CategoricalParamConfig,
    ContinuousParamConfig,
    IntegerParamConfig,
    LatticeParamConfig,
    ParameterTransformer,
    ParamInfeasibleError,
    Scale,
    create_param_config,
)


class TestContinuousParamConfig:
    @pytest.mark.parametrize(
        "min_val, max_val, scale",
        [
            (0.0, 1.0, Scale.LINEAR),
            (1e-4, 1e-1, Scale.LOG),
        ],
    )
    def test_instantiation_valid(self, min_val: float, max_val: float, scale: Scale) -> None:
        config = ContinuousParamConfig(min=min_val, max=max_val, scale=scale)
        assert config.min == min_val
        assert config.max == max_val
        assert config.scale == scale

    @pytest.mark.parametrize(
        "min_val, max_val, scale",
        [
            (1.0, 0.0, Scale.LINEAR),
            (1e-1, 1e-4, Scale.LOG),
            (-1.0, 1.0, Scale.LOG),
            (1.0, 2.0, "invalid"),  # type: ignore
        ],
    )
    def test_instantiation_invalid(self, min_val: float, max_val: float, scale: Any) -> None:
        with pytest.raises(ValidationError):
            ContinuousParamConfig(min=min_val, max=max_val, scale=scale)

    @pytest.mark.parametrize(
        "u, expected",
        [
            (0.0, 0.0),
            (0.5, 0.5),
            (1.0, 1.0),
        ],
    )
    def test_transform_param_linear(self, u: float, expected: float) -> None:
        config = ContinuousParamConfig(min=0.0, max=1.0, scale=Scale.LINEAR)
        result = config.transform_param(u)
        assert result == pytest.approx(expected)

    @pytest.mark.parametrize(
        "u, expected",
        [
            (0.0, 1e-4),
            (1.0, 1e-1),
        ],
    )
    def test_transform_param_log_boundary(self, u: float, expected: float) -> None:
        config = ContinuousParamConfig(min=1e-4, max=1e-1, scale=Scale.LOG)
        result = config.transform_param(u)
        assert result == pytest.approx(expected)

    def test_transform_param_log_middle(self) -> None:
        config = ContinuousParamConfig(min=1e-4, max=1e-1, scale=Scale.LOG)
        assert config.transform_param(0.5) == pytest.approx(np.sqrt(1e-4 * 1e-1))

    def test_transform_param_invalid_u(self) -> None:
        config = ContinuousParamConfig(min=0.0, max=1.0)
        with pytest.raises(ValueError):
            config.transform_param(-0.1)
        with pytest.raises(ValueError):
            config.transform_param(1.1)

    def test_back_transform_param_valid_range_linear(self) -> None:
        config = ContinuousParamConfig(min=0.0, max=1.0, scale=Scale.LINEAR)
        x_values = [0.0, 0.5, 1.0]
        for x in x_values:
            u = config.back_transform_param(x)
            assert 0 <= u <= 1

    def test_back_transform_param_valid_range_log(self) -> None:
        config = ContinuousParamConfig(min=1e-4, max=1e-1, scale=Scale.LOG)
        x_values = [1e-4, np.sqrt(1e-4 * 1e-1), 1e-1]
        for x in x_values:
            u = config.back_transform_param(x)
            assert 0 <= u <= 1

    def test_back_transform_param_infeasible(self) -> None:
        config = ContinuousParamConfig(min=0.0, max=1.0)
        with pytest.raises(ParamInfeasibleError):
            config.back_transform_param(-0.1)
        with pytest.raises(ParamInfeasibleError):
            config.back_transform_param(1.1)

    def test_transform_back_transform_linear(self) -> None:
        config = ContinuousParamConfig(min=0.0, max=1.0)
        u_values = [0.1, 0.5, 0.9]
        for u in u_values:
            x = config.transform_param(u)
            u_back = config.back_transform_param(x)
            assert u_back == pytest.approx(u)

    def test_transform_back_transform_log(self) -> None:
        config = ContinuousParamConfig(min=1e-4, max=1e-1, scale=Scale.LOG)
        u_values = [0.1, 0.5, 0.9]
        for u in u_values:
            x = config.transform_param(u)
            u_back = config.back_transform_param(x)
            assert u_back == pytest.approx(u)


class TestCategoricalParamConfig:
    @pytest.mark.parametrize(
        "categories",
        [
            ["a", "b", "c"],
            [1, 2, 3],
            [True, False],
        ],
    )
    def test_instantiation_valid(self, categories: list[Any]) -> None:
        config = CategoricalParamConfig(categories=categories)
        assert config.categories == categories
        assert config.n_categories == len(categories)

    def test_instantiation_invalid(self) -> None:
        with pytest.raises(ValidationError):
            CategoricalParamConfig(categories=[])

    @pytest.mark.parametrize(
        "u, expected",
        [
            (0.0, "a"),
            (0.33, "a"),
            (0.34, "b"),
            (0.99, "c"),
        ],
    )
    def test_transform_param(self, u: float, expected: Any) -> None:
        config = CategoricalParamConfig(categories=["a", "b", "c"])
        result = config.transform_param(u)
        assert result == expected

    def test_transform_param_invalid_u(self) -> None:
        config = CategoricalParamConfig(categories=["a", "b"])
        with pytest.raises(ValueError):
            config.transform_param(-0.1)
        with pytest.raises(ValueError):
            config.transform_param(1.1)

    def test_back_transform_param_valid_range(self) -> None:
        config = CategoricalParamConfig(categories=["a", "b", "c"])
        x_values = ["a", "b", "c"]
        for x in x_values:
            u = config.back_transform_param(x)
            assert 0 <= u <= 1

    def test_back_transform_param_infeasible(self) -> None:
        config = CategoricalParamConfig(categories=["a", "b"])
        with pytest.raises(ParamInfeasibleError):
            config.back_transform_param("c")

    def test_transform_back_transform(self) -> None:
        config = CategoricalParamConfig(categories=["a", "b", "c"])
        u_values = [0.1, 0.5, 0.9]
        for u in u_values:
            x = config.transform_param(u)
            u_back = config.back_transform_param(x)
            assert 0 <= u_back <= 1


class TestIntegerParamConfig:
    @pytest.mark.parametrize(
        "min_val, max_val",
        [
            (0, 10),
            (-5, 5),
        ],
    )
    def test_instantiation_valid(self, min_val: int, max_val: int) -> None:
        config = IntegerParamConfig(min=min_val, max=max_val)
        assert config.min == min_val
        assert config.max == max_val
        assert config.num_values == max_val - min_val + 1

    def test_instantiation_invalid(self) -> None:
        with pytest.raises(ValidationError):
            IntegerParamConfig(min=10, max=0)

    @pytest.mark.parametrize(
        "u, expected",
        [
            (0.0, 0),
            (0.49, 4),
            (0.5, 5),
            (0.99, 9),
        ],
    )
    def test_transform_param(self, u: float, expected: int) -> None:
        config = IntegerParamConfig(min=0, max=9)
        result = config.transform_param(u)
        assert result == expected

    def test_transform_param_invalid_u(self) -> None:
        config = IntegerParamConfig(min=0, max=10)
        with pytest.raises(ValueError):
            config.transform_param(-0.1)
        with pytest.raises(ValueError):
            config.transform_param(1.1)

    def test_back_transform_param_valid_range(self) -> None:
        config = IntegerParamConfig(min=0, max=9)
        x_values = [0, 5, 9]
        for x in x_values:
            u = config.back_transform_param(x)
            assert 0 <= u <= 1

    def test_back_transform_param_infeasible(self) -> None:
        config = IntegerParamConfig(min=0, max=10)
        with pytest.raises(ParamInfeasibleError):
            config.back_transform_param(-1)
        with pytest.raises(ParamInfeasibleError):
            config.back_transform_param(11)

    def test_transform_back_transform(self) -> None:
        config = IntegerParamConfig(min=0, max=9)
        u_values = [0.1, 0.5, 0.9]
        for u in u_values:
            x = config.transform_param(u)
            u_back = config.back_transform_param(x)
            assert 0 <= u_back <= 1


class TestLatticeParamConfig:
    @pytest.mark.parametrize(
        "min_val, max_val, num_values",
        [
            (0.0, 1.0, 5),
            (-1.0, 1.0, 3),
        ],
    )
    def test_instantiation_valid(self, min_val: float, max_val: float, num_values: int) -> None:
        config = LatticeParamConfig(min=min_val, max=max_val, num_values=num_values)
        assert config.min == min_val
        assert config.max == max_val
        assert config.num_values == num_values
        assert config.step_size == (max_val - min_val) / (num_values - 1)

    @pytest.mark.parametrize(
        "min_val, max_val, num_values",
        [
            (1.0, 0.0, 5),
            (0.0, 1.0, 1),
        ],
    )
    def test_instantiation_invalid(self, min_val: float, max_val: float, num_values: int) -> None:
        with pytest.raises(ValidationError):
            LatticeParamConfig(min=min_val, max=max_val, num_values=num_values)

    @pytest.mark.parametrize(
        "u, expected",
        [
            (0.0, 0.0),
            (0.5, 0.5),
            (1.0, 1.0),
        ],
    )
    def test_transform_param(self, u: float, expected: float) -> None:
        config = LatticeParamConfig(min=0.0, max=1.0, num_values=3)
        result = config.transform_param(u)
        assert result == pytest.approx(expected)

    def test_transform_param_invalid_u(self) -> None:
        config = LatticeParamConfig(min=0.0, max=1.0, num_values=3)
        with pytest.raises(ValueError):
            config.transform_param(-0.1)
        with pytest.raises(ValueError):
            config.transform_param(1.1)

    def test_back_transform_param_valid_range(self) -> None:
        config = LatticeParamConfig(min=0.0, max=1.0, num_values=3)
        x_values = [0.0, 0.5, 1.0]
        for x in x_values:
            u = config.back_transform_param(x)
            assert 0 <= u <= 1

    def test_back_transform_param_infeasible(self) -> None:
        config = LatticeParamConfig(min=0.0, max=1.0, num_values=3)
        with pytest.raises(ParamInfeasibleError):
            config.back_transform_param(-0.1)
        with pytest.raises(ParamInfeasibleError):
            config.back_transform_param(1.1)

    def test_transform_back_transform(self) -> None:
        config = LatticeParamConfig(min=0.0, max=1.0, num_values=3)
        u_values = [0.1, 0.5, 0.9]
        for u in u_values:
            x = config.transform_param(u)
            u_back = config.back_transform_param(x)
            assert 0 <= u_back <= 1


class TestCreateParamConfig:
    def test_create_continuous(self) -> None:
        config = create_param_config("continuous", min=0.0, max=1.0)
        assert isinstance(config, ContinuousParamConfig)

    def test_create_integer(self) -> None:
        config = create_param_config("integer", min=0, max=10)
        assert isinstance(config, IntegerParamConfig)

    def test_create_categorical(self) -> None:
        config = create_param_config("categorical", categories=["a", "b"])
        assert isinstance(config, CategoricalParamConfig)

    def test_create_lattice(self) -> None:
        config = create_param_config("lattice", min=0.0, max=1.0, num_values=3)
        assert isinstance(config, LatticeParamConfig)

    def test_create_invalid_type(self) -> None:
        with pytest.raises(ValueError):
            create_param_config("invalid_type", min=0.0, max=1.0)  # type: ignore

    def test_create_missing_kwargs(self) -> None:
        with pytest.raises(ValueError):
            create_param_config("continuous", min=0.0)


class TestParameterTransformer:
    @pytest.fixture
    def example_params(self) -> dict[str, Any]:
        return {
            "lr": ContinuousParamConfig(min=1e-4, max=1e-1, scale=Scale.LOG),
            "bs": IntegerParamConfig(min=16, max=256),
            "opt": CategoricalParamConfig(categories=["adam", "sgd"]),
        }

    def test_init(self, example_params: dict[str, Any]) -> None:
        transformer = ParameterTransformer(example_params)
        assert transformer.param_configs == example_params

    def test_transform_normalized_params(self, example_params: dict[str, Any]) -> None:
        transformer = ParameterTransformer(example_params)
        normalized = [0.1, 0.5, 0.9]
        transformed = transformer.transform_normalized_params(normalized)
        assert isinstance(transformed, dict)
        assert isinstance(transformed["lr"], float)
        assert isinstance(transformed["bs"], int)
        assert isinstance(transformed["opt"], str)

    def test_transform_normalized_params_mismatch(self, example_params: dict[str, Any]) -> None:
        transformer = ParameterTransformer(example_params)
        normalized = [0.1, 0.5]  # Missing one parameter
        with pytest.raises(ValueError):
            transformer.transform_normalized_params(normalized)

    def test_back_transform_param_dict_valid_range(self, example_params: dict[str, Any]) -> None:
        transformer = ParameterTransformer(example_params)
        param_values = {
            "lr": 0.001,
            "bs": 32,
            "opt": "adam",
        }
        normalized = transformer.back_transform_param_dict(param_values)
        assert isinstance(normalized, list)
        assert len(normalized) == 3
        assert all(0 <= u <= 1 for u in normalized)

    def test_back_transform_param_dict_mismatch_keys(self, example_params: dict[str, Any]) -> None:
        transformer = ParameterTransformer(example_params)
        param_values = {"l": 0.001, "bs": 32, "opt": "adam"}  # Wrong key 'l' instead of 'lr'
        with pytest.raises(ValueError):
            transformer.back_transform_param_dict(param_values)

    def test_transform_back_transform_transformer(self, example_params: dict[str, Any]) -> None:
        transformer = ParameterTransformer(example_params)
        original_params = {
            "lr": 0.0005,
            "bs": 64,
            "opt": "adam",
        }
        normalized_params = transformer.back_transform_param_dict(original_params)
        reconstructed_params = transformer.transform_normalized_params(normalized_params)

        assert reconstructed_params["lr"] == pytest.approx(original_params["lr"])
        assert reconstructed_params["bs"] == original_params["bs"]
        assert reconstructed_params["opt"] == original_params["opt"]


if __name__ == "__main__":
    pytest.main([__file__])
