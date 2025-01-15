"""Tests for utility classes and functions."""

from typing import Type

import pytest
from pydantic import ConfigDict, Field, ValidationError

from hola.core.utils import BaseConfig, uniform_to_category


class MyConfig(BaseConfig):
    """Example config class for testing BaseConfig."""

    name: str
    count: int = 0


@pytest.fixture
def my_config_class() -> Type[MyConfig]:
    """Fixture providing the MyConfig class for testing."""
    return MyConfig


class TestBaseConfig:
    def test_instantiation_with_valid_data(self, my_config_class: Type[MyConfig]) -> None:
        config = my_config_class(name="test", count=42)
        assert config.name == "test"
        assert config.count == 42

    def test_instantiation_with_type_coercion(self, my_config_class: Type[MyConfig]) -> None:
        config = my_config_class(name="test", count="42")  # type: ignore
        assert config.count == 42

    def test_frozen_config(self, my_config_class: Type[MyConfig]) -> None:
        config = my_config_class(name="test")
        with pytest.raises(ValidationError):
            config.name = "new_test"  # type: ignore

    def test_extra_forbid(self, my_config_class: Type[MyConfig]) -> None:
        with pytest.raises(ValidationError):
            my_config_class(name="test", extra_field="value")  # type: ignore

    def test_str_strip_whitespace(self, my_config_class: Type[MyConfig]) -> None:
        config = my_config_class(name="  test  ")
        assert config.name == "test"

    def test_parse_from_dict(self, my_config_class: Type[MyConfig]) -> None:
        data = {"name": "test", "count": 42}
        config = my_config_class.parse(data)
        assert isinstance(config, my_config_class)
        assert config.name == "test"
        assert config.count == 42

    def test_parse_from_json_string(self, my_config_class: Type[MyConfig]) -> None:
        data = '{"name": "test", "count": 42}'
        config = my_config_class.parse(data)
        assert isinstance(config, my_config_class)
        assert config.name == "test"
        assert config.count == 42

    def test_parse_from_bytes(self, my_config_class: Type[MyConfig]) -> None:
        data = b'{"name": "test", "count": 42}'
        config = my_config_class.parse(data)
        assert isinstance(config, my_config_class)
        assert config.name == "test"
        assert config.count == 42

    def test_parse_from_existing_instance(self, my_config_class: Type[MyConfig]) -> None:
        original_config = my_config_class(name="test", count=42)
        parsed_config = my_config_class.parse(original_config)
        assert isinstance(parsed_config, my_config_class)
        assert parsed_config.name == "test"
        assert parsed_config.count == 42

    def test_parse_type_coercion(self, my_config_class: Type[MyConfig]) -> None:
        data = {"name": "test", "count": "42"}  # type: ignore
        config = my_config_class.parse(data)
        assert config.count == 42

    def test_parse_populate_by_name(self) -> None:
        class ConfigWithAlias(BaseConfig):
            my_name: str = Field("default", alias="MyName")
            model_config = ConfigDict(populate_by_name=True)

        data = {"MyName": "test"}
        config = ConfigWithAlias.parse(data)
        assert config.my_name == "test"

    def test_parse_raises_validation_error(self, my_config_class: Type[MyConfig]) -> None:
        data = {"name": "test", "count": "invalid"}
        with pytest.raises(ValidationError):
            my_config_class.parse(data)

    def test_parse_unsupported_data_type(self, my_config_class: Type[MyConfig]) -> None:
        with pytest.raises(TypeError):
            my_config_class.parse(123)  # type: ignore


class TestUniformToCategory:
    @pytest.mark.parametrize(
        "u_sample, n_categories, expected",
        [
            (0.7, 5, 3),
            (0.2, 5, 1),
            (0.0, 5, 0),
            (1.0, 5, 4),
        ],
    )
    def test_single_value(self, u_sample: float, n_categories: int, expected: int) -> None:
        assert uniform_to_category(u_sample, n_categories) == expected

    def test_boundary_values(self) -> None:
        assert uniform_to_category(0.0, 5) == 0
        assert uniform_to_category(1.0, 5) == 4

    @pytest.mark.parametrize(
        "u_sample, n_categories, expected",
        [(0.3, 2, 0), (0.7, 2, 1), (0.1, 10, 1), (0.9, 10, 9), (1.0, 10, 9)],
    )
    def test_different_number_of_categories(
        self, u_sample: float, n_categories: int, expected: int
    ) -> None:
        assert uniform_to_category(u_sample, n_categories) == expected

    def test_raises_type_error_for_invalid_u_sample_type(self) -> None:
        with pytest.raises(TypeError):
            uniform_to_category("0.5", 5)  # type: ignore

    def test_raises_type_error_for_invalid_n_categories_type(self) -> None:
        with pytest.raises(TypeError):
            uniform_to_category(0.5, "5")  # type: ignore

    @pytest.mark.parametrize("n_categories", [0, -1])
    def test_raises_value_error_for_non_positive_n_categories(self, n_categories: int) -> None:
        with pytest.raises(ValueError):
            uniform_to_category(0.5, n_categories)

    @pytest.mark.parametrize("u_sample", [-0.1, 1.1])
    def test_raises_value_error_for_u_sample_out_of_range(self, u_sample: float) -> None:
        with pytest.raises(ValueError):
            uniform_to_category(u_sample, 5)


if __name__ == "__main__":
    pytest.main([__file__])
