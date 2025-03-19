"""Protocol message definitions for communication between components."""

from typing import Any, Union

from msgspec import Struct

from hola.core.objectives import ObjectiveName
from hola.core.parameters import ParameterName


class Result(Struct):
    """Complete result of a trial evaluation"""

    parameters: dict[ParameterName, Any]
    objectives: dict[ObjectiveName, float]


# Request Messages
class GetSuggestionRequest(Struct, tag="get_suggestion"):
    """Request for parameter suggestions."""
    worker_id: int


class SubmitResultRequest(Struct, tag="submit_result"):
    """Request to submit evaluation results."""
    worker_id: int
    result: Result


class ShutdownRequest(Struct, tag="shutdown"):
    """Request to shutdown the server."""
    pass


class StatusRequest(Struct, tag="status"):
    """Request for current optimization status."""
    pass


# Response Messages
class GetSuggestionResponse(Struct, tag="suggestion_response"):
    """Response containing parameter suggestions."""
    parameters: dict[ParameterName, Any] | None


class SubmitResultResponse(Struct, tag="result_response"):
    """Response to result submission."""
    success: bool
    error: str | None = None


class StatusResponse(Struct, tag="status_response"):
    """Response containing optimization status."""
    active_workers: int
    total_evaluations: int
    best_objectives: dict[ObjectiveName, float] | None = None


# Define the message union type
Message = Union[
    GetSuggestionRequest,
    SubmitResultRequest,
    ShutdownRequest,
    StatusRequest,
    GetSuggestionResponse,
    SubmitResultResponse,
    StatusResponse,
]