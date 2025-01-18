from datetime import datetime
from typing import Any, TypeAlias, Union

from hola.core.objectives import ObjectiveName
from hola.core.parameters import ParameterName
from hola.messages.base import BaseMessage
from hola.messages.errors import StructuredError


class BaseWorkerMessage(BaseMessage, frozen=True):
    pass


class WorkerRegistration(BaseWorkerMessage, frozen=True, tag="register"):
    capabilities: dict[str, Any]


class WorkerDeregistration(BaseWorkerMessage, frozen=True, tag="deregister"):
    reason: str | None = None


class HeartbeatPing(BaseWorkerMessage, frozen=True, tag="heartbeat"):
    last_active: datetime
    current_load: int


class SampleRequest(BaseWorkerMessage, frozen=True, tag="sample"):
    n_samples: int


class Evaluation(BaseWorkerMessage, frozen=True, tag="evaluation"):
    parameters: dict[ParameterName, Any]
    objectives: dict[ObjectiveName, float]


class WorkerError(BaseWorkerMessage, frozen=True, tag="error"):
    error: StructuredError


class StatusUpdate(BaseWorkerMessage, frozen=True, tag="status"):
    status: None = None


WorkerMessage: TypeAlias = Union[
    WorkerRegistration,
    WorkerDeregistration,
    HeartbeatPing,
    SampleRequest,
    Evaluation,
    WorkerError,
    StatusUpdate,
]
