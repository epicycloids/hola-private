from typing import Any

import msgspec

from hola.core.objectives import ObjectiveName
from hola.core.parameters import ParameterName


class RESTGetSuggestionResponse(msgspec.Struct):
    """Response to GET /job containing parameter suggestions."""

    parameters: dict[ParameterName, Any] | None = None
    error: str | None = None


class RESTSubmitResult(msgspec.Struct):
    """Request body for POST /result containing evaluation results."""

    parameters: dict[ParameterName, Any]
    objectives: dict[ObjectiveName, float]


class RESTSubmitResponse(msgspec.Struct):
    """Response to POST /result indicating success/failure."""

    success: bool
    error: str | None = None
