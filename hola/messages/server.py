from typing import Any, TypeAlias, Union

from msgspec import Struct

from hola.core.parameters import ParameterName
from hola.messages.base import BaseMessage
from hola.messages.errors import StructuredError
from hola.server.status import OptimizationStatus


class SampleResponseContent(Struct, frozen=True, tag="samples"):
    samples: list[dict[ParameterName, Any]]


class StatusUpdateContent(Struct, frozen=True, tag="status"):
    status: OptimizationStatus


class ConfigUpdatedContent(Struct, frozen=True, tag="config_updated"):
    message: str | None = None


class ServerErrorContent(Struct, frozen=True, tag="error"):
    error: StructuredError


class OKContent(Struct, frozen=True, tag="ok"):
    pass


ServerMessageContent: TypeAlias = Union[
    SampleResponseContent, StatusUpdateContent, ConfigUpdatedContent, ServerErrorContent, OKContent
]


class ServerMessage(BaseMessage, frozen=True):
    content: ServerMessageContent
