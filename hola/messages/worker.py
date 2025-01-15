from datetime import datetime
from typing import Any, TypeAlias, Union
from uuid import UUID

from msgspec import Struct

from hola.core.objectives import ObjectiveName
from hola.core.parameters import ParameterName
from hola.messages.base import BaseMessage
from hola.messages.errors import StructuredError
from hola.worker.status import WorkerStatus


class WorkerRegistrationContent(Struct, frozen=True, tag="register"):
    capabilities: dict[str, Any]


class WorkerDeregistrationContent(Struct, frozen=True, tag="deregister"):
    reason: str | None = None


class HeartbeatPingContent(Struct, frozen=True, tag="heartbeat"):
    last_active: datetime
    current_load: int


class SampleRequestContent(Struct, frozen=True, tag="sample"):
    n_samples: int


class EvaluationContent(Struct, frozen=True, tag="evaluation"):
    parameters: dict[ParameterName, Any]
    objectives: dict[ObjectiveName, float]


class WorkerErrorContent(Struct, frozen=True, tag="error"):
    error: StructuredError


class StatusUpdateContent(Struct, frozen=True, tag="status"):
    status: WorkerStatus


class BaseWorkerMessage(BaseMessage, frozen=True):
    worker_id: UUID


WorkerMessageContent: TypeAlias = Union[
    WorkerRegistrationContent,
    WorkerDeregistrationContent,
    HeartbeatPingContent,
    SampleRequestContent,
    EvaluationContent,
    WorkerErrorContent,
    StatusUpdateContent,
]


class WorkerMessage(BaseMessage, frozen=True):
    content: WorkerMessageContent
