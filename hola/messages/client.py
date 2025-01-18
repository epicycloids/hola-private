from typing import Any, TypeAlias, Union

from hola.core.objectives import ObjectiveName
from hola.core.parameters import ParameterName
from hola.messages.base import BaseMessage
from hola.messages.errors import StructuredError


class BaseClientMessage(BaseMessage, frozen=True):
    pass


class InitializeRequest(BaseClientMessage, frozen=True, tag="initialize"):
    # TODO: Need some way to initialize the Sampler. It will currently
    # initialize an explore/exploit sampler with a Sobol' explorer and a GMM
    # exploiter.
    objectives_config: dict[ObjectiveName, dict[str, Any]]
    parameters_config: dict[ParameterName, dict[str, Any]]


class StatusRequest(BaseClientMessage, frozen=True, tag="status"):
    pass


class PauseRequest(BaseClientMessage, frozen=True, tag="pause"):
    pass


class ResumeRequest(BaseClientMessage, frozen=True, tag="resume"):
    pass


class UpdateObjectiveConfig(BaseClientMessage, frozen=True, tag="update_objective_config"):
    new_config: dict[ObjectiveName, dict[str, Any]]


class UpdateParameterConfig(BaseClientMessage, frozen=True, tag="update_parameter_config"):
    new_config: dict[ParameterName, dict[str, Any]]


class SaveStateRequest(BaseClientMessage, frozen=True, tag="save_state"):
    filepath: str


class LoadStateRequest(BaseClientMessage, frozen=True, tag="load_state"):
    filepath: str


class ClientError(BaseClientMessage, frozen=True, tag="error"):
    error: StructuredError


ClientMessage: TypeAlias = Union[
    InitializeRequest,
    StatusRequest,
    PauseRequest,
    ResumeRequest,
    UpdateObjectiveConfig,
    UpdateParameterConfig,
    SaveStateRequest,
    LoadStateRequest,
    ClientError,
]
