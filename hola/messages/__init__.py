"""Message protocol definitions for HOLA communication."""

from hola.messages.protocol import (
    GetSuggestionRequest,
    GetSuggestionResponse,
    Message,
    Result,
    ShutdownRequest,
    StatusRequest,
    StatusResponse,
    SubmitResultRequest,
    SubmitResultResponse,
)

__all__ = [
    "GetSuggestionRequest",
    "GetSuggestionResponse",
    "Message",
    "ShutdownRequest",
    "StatusRequest",
    "StatusResponse",
    "SubmitResultRequest",
    "SubmitResultResponse",
    "Result",
]