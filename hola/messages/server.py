from typing import Any, TypeAlias, Union

from hola.core.coordinator import OptimizationState
from hola.core.parameters import ParameterName
from hola.messages.base import BaseMessage
from hola.messages.errors import StructuredError


class BaseServerMessage(BaseMessage, frozen=True):
    pass


class SampleResponse(BaseServerMessage, frozen=True, tag="samples"):
    samples: list[dict[ParameterName, Any]]


class StatusUpdate(BaseServerMessage, frozen=True, tag="status"):
    status: OptimizationState


class ConfigUpdated(BaseServerMessage, frozen=True, tag="config_updated"):
    message: str | None = None


class ServerError(BaseServerMessage, frozen=True, tag="error"):
    error: StructuredError


class OK(BaseServerMessage, frozen=True, tag="ok"):
    pass


ServerMessage: TypeAlias = Union[SampleResponse, StatusUpdate, ConfigUpdated, ServerError, OK]
