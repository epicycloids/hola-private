from typing import Any

from msgspec import Struct

from hola.core.parameters import ParameterName
from hola.messages.base import Result


# Request Messages
class GetSuggestionRequest(Struct, tag="get_suggestion"):
    worker_id: int


class SubmitResultRequest(Struct, tag="submit_result"):
    worker_id: int
    result: Result


class ShutdownRequest(Struct, tag="shutdown"):
    pass


# Response Messages
class GetSuggestionResponse(Struct, tag="suggestion_response"):
    parameters: dict[ParameterName, Any] | None


class SubmitResultResponse(Struct, tag="result_response"):
    success: bool
    error: str | None = None


# Define the message union type
Message = (
    GetSuggestionRequest
    | SubmitResultRequest
    | ShutdownRequest
    | GetSuggestionResponse
    | SubmitResultResponse
)
