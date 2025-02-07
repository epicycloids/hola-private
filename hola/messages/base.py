from typing import Any

from msgspec import Struct

from hola.core.objectives import ObjectiveName
from hola.core.parameters import ParameterName


class ParameterSet(Struct):
    values: dict[ParameterName, Any]


class Result(Struct):
    parameters: dict[ParameterName, Any]
    objectives: dict[ObjectiveName, float]
