"""
Defines the message structures used for communication between distributed components
(Scheduler, Worker, Server) using msgspec for efficient serialization.
"""

from typing import Any, List, Dict, Optional, Union
import msgspec
from msgspec import Struct

# Assuming ParameterName and ObjectiveName are simple types like str for now
# If they are more complex, import them from hola.core
# from hola.core.parameters import ParameterName
# from hola.core.objectives import ObjectiveName
ParameterName = str
ObjectiveName = str

# ============================================================================
# Core ZMQ Message Types
# ============================================================================


class ParameterSet(Struct):
    """Parameters suggested by the coordinator for evaluation"""

    values: dict[ParameterName, Any]


class Result(Struct):
    """Complete result of a trial evaluation"""

    parameters: dict[ParameterName, Any]
    objectives: dict[ObjectiveName, float]


# ============================================================================
# ZMQ Request Messages
# ============================================================================


class GetSuggestionRequest(Struct, tag="get_suggestion", tag_field="tag"):
    worker_id: int


class SubmitResultRequest(Struct, tag="submit_result", tag_field="tag"):
    worker_id: int
    result: Result


class HeartbeatRequest(Struct, tag="heartbeat", tag_field="tag"):
    worker_id: int


class ShutdownRequest(Struct, tag="shutdown", tag_field="tag"):
    pass


class StatusRequest(Struct, tag="status", tag_field="tag"):
    pass


class GetTrialsRequest(Struct, tag="get_trials", tag_field="tag"):
    ranked_only: bool = True


class GetMetadataRequest(Struct, tag="get_metadata", tag_field="tag"):
    trial_ids: Optional[Union[int, List[int]]] = None


class GetTopKRequest(Struct, tag="get_top_k", tag_field="tag"):
    k: int = 1


class IsMultiGroupRequest(Struct, tag="is_multi_group", tag_field="tag"):
    pass


# ============================================================================
# ZMQ Response Messages
# ============================================================================


class GetSuggestionResponse(Struct, tag="suggestion_response", tag_field="tag"):
    parameters: dict[ParameterName, Any] | None


class SubmitResultResponse(Struct, tag="result_response", tag_field="tag"):
    success: bool
    is_best: bool = False
    error: str | None = None


class HeartbeatResponse(Struct, tag="heartbeat_response", tag_field="tag"):
    success: bool


class StatusResponse(Struct, tag="status_response", tag_field="tag"):
    active_workers: int
    total_evaluations: int
    best_objectives: dict[ObjectiveName, float] | None = None


class GetTrialsResponse(Struct, tag="trials_response", tag_field="tag"):
    trials: List[Dict[str, Any]]


class GetMetadataResponse(Struct, tag="metadata_response", tag_field="tag"):
    metadata: List[Dict[str, Any]]


class GetTopKResponse(Struct, tag="top_k_response", tag_field="tag"):
    trials: List[Dict[str, Any]]


class IsMultiGroupResponse(Struct, tag="multi_group_response", tag_field="tag"):
    is_multi_group: bool


# ============================================================================
# ZMQ Message Union Type
# ============================================================================

# Define the message union type for ZMQ communication
Message = (
    GetSuggestionRequest
    | SubmitResultRequest
    | HeartbeatRequest
    | ShutdownRequest
    | StatusRequest
    | GetTrialsRequest
    | GetMetadataRequest
    | GetTopKRequest
    | IsMultiGroupRequest
    | GetSuggestionResponse
    | SubmitResultResponse
    | HeartbeatResponse
    | StatusResponse
    | GetTrialsResponse
    | GetMetadataResponse
    | GetTopKResponse
    | IsMultiGroupResponse
)


# ============================================================================
# REST API Message Types
# ============================================================================


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


class RESTHeartbeatRequest(msgspec.Struct):
    """Request body for POST /heartbeat."""

    worker_id: int


class RESTHeartbeatResponse(msgspec.Struct):
    """Response to POST /heartbeat."""

    success: bool
    error: str | None = None


class RESTGetTrialsResponse(msgspec.Struct):
    """Response to GET /trials."""

    trials: List[Dict[str, Any]]
    error: str | None = None


class RESTGetMetadataResponse(msgspec.Struct):
    """Response to GET /metadata."""

    metadata: List[Dict[str, Any]]
    error: str | None = None


class RESTGetTopKResponse(msgspec.Struct):
    """Response to GET /top."""

    trials: List[Dict[str, Any]]
    error: str | None = None


class RESTIsMultiGroupResponse(msgspec.Struct):
    """Response to GET /is_multi_group."""

    is_multi_group: bool
    error: str | None = None