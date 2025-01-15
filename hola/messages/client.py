from struct import Struct
from typing import Any, TypeAlias, Union

from hola.core.objectives import ObjectiveConfig, ObjectiveName
from hola.core.parameters import ParameterName, PredefinedParameterConfig
from hola.messages.base import BaseMessage
from hola.messages.errors import StructuredError


class InitializeRequestContent(Struct, frozen=True, tag="initialize"):
    # TODO: Need some way to initialize the Sampler. It will currently
    # initialize an explore/exploit sampler with a Sobol' explorer and a GMM
    # exploiter.
    objectives_config: dict[ObjectiveName, dict[str, Any] | ObjectiveConfig]
    parameters_config: dict[ParameterName, dict[str, Any] | PredefinedParameterConfig]


class StatusRequestContent(Struct, frozen=True, tag="status"):
    pass


class PauseRequestContent(Struct, frozen=True, tag="pause"):
    pass


class ResumeRequestContent(Struct, frozen=True, tag="resume"):
    pass


class UpdateObjectiveConfigContent(Struct, frozen=True, tag="update_objective_config"):
    new_config: dict[ObjectiveName, dict[str, Any] | ObjectiveConfig]


class UpdateParameterConfigContent(Struct, frozen=True, tag="update_parameter_config"):
    new_config: dict[ParameterName, dict[str, Any] | PredefinedParameterConfig]


class SaveStateRequestContent(Struct, frozen=True, tag="save_state"):
    filepath: str


class LoadStateRequestContent(Struct, frozen=True, tag="load_state"):
    filepath: str


class ClientErrorContent(Struct, frozen=True, tag="error"):
    error: StructuredError


ClientMessageContent: TypeAlias = Union[
    InitializeRequestContent,
    StatusRequestContent,
    PauseRequestContent,
    ResumeRequestContent,
    UpdateObjectiveConfigContent,
    UpdateParameterConfigContent,
    SaveStateRequestContent,
    LoadStateRequestContent,
    ClientErrorContent,
]


class ClientMessage(BaseMessage, frozen=True):
    content: ClientMessageContent
