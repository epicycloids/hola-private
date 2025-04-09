"""Tests for hola.distributed.messages using msgspec."""

import pytest
import msgspec

# Assuming the tests directory is at the same level as hola/
# Adjust imports if the structure is different
from hola.distributed.messages import (
    Message, # Union type
    ParameterSet,
    Result,
    GetSuggestionRequest,
    SubmitResultRequest,
    HeartbeatRequest,
    ShutdownRequest,
    StatusRequest,
    GetTrialsRequest,
    GetMetadataRequest,
    GetTopKRequest,
    IsMultiGroupRequest,
    GetSuggestionResponse,
    SubmitResultResponse,
    HeartbeatResponse,
    StatusResponse,
    GetTrialsResponse,
    GetMetadataResponse,
    GetTopKResponse,
    IsMultiGroupResponse,
    RESTGetSuggestionResponse,
    RESTSubmitResult,
    RESTSubmitResponse,
    RESTHeartbeatRequest,
    RESTHeartbeatResponse,
    RESTGetTrialsResponse,
    RESTGetMetadataResponse,
    RESTGetTopKResponse,
    RESTIsMultiGroupResponse,
)

# --- Test Data ---

EXAMPLE_PARAMS = {"x": 1.0, "y": "category_a"}
EXAMPLE_OBJECTIVES = {"obj1": 10.5, "obj2": -3.0}
EXAMPLE_RESULT = Result(parameters=EXAMPLE_PARAMS, objectives=EXAMPLE_OBJECTIVES)
EXAMPLE_WORKER_ID = 123

# --- Test Cases ---

# List of ZMQ request types and example instances
zmq_request_cases = [
    (GetSuggestionRequest, GetSuggestionRequest(worker_id=EXAMPLE_WORKER_ID)),
    (SubmitResultRequest, SubmitResultRequest(worker_id=EXAMPLE_WORKER_ID, result=EXAMPLE_RESULT)),
    (HeartbeatRequest, HeartbeatRequest(worker_id=EXAMPLE_WORKER_ID)),
    (ShutdownRequest, ShutdownRequest()),
    (StatusRequest, StatusRequest()),
    (GetTrialsRequest, GetTrialsRequest(ranked_only=False)),
    (GetTrialsRequest, GetTrialsRequest(ranked_only=True)), # Default
    (GetMetadataRequest, GetMetadataRequest(trial_ids=None)), # Default
    (GetMetadataRequest, GetMetadataRequest(trial_ids=1)),
    (GetMetadataRequest, GetMetadataRequest(trial_ids=[1, 5, 10])),
    (GetTopKRequest, GetTopKRequest(k=5)),
    (GetTopKRequest, GetTopKRequest(k=1)), # Default
    (IsMultiGroupRequest, IsMultiGroupRequest()),
]

# List of ZMQ response types and example instances
zmq_response_cases = [
    (GetSuggestionResponse, GetSuggestionResponse(parameters=EXAMPLE_PARAMS)),
    (GetSuggestionResponse, GetSuggestionResponse(parameters=None)),
    (SubmitResultResponse, SubmitResultResponse(success=True, is_best=True, error=None)),
    (SubmitResultResponse, SubmitResultResponse(success=False, is_best=False, error="Something failed")),
    (HeartbeatResponse, HeartbeatResponse(success=True)),
    (StatusResponse, StatusResponse(active_workers=3, total_evaluations=100, best_objectives=EXAMPLE_OBJECTIVES)),
    (StatusResponse, StatusResponse(active_workers=0, total_evaluations=0, best_objectives=None)),
    (GetTrialsResponse, GetTrialsResponse(trials=[{"id": 1, "params": {}}])),
    (GetMetadataResponse, GetMetadataResponse(metadata=[{"id": 1, "meta": {}}])),
    (GetTopKResponse, GetTopKResponse(trials=[{"id": 1, "params": {}, "objectives": {}}])),
    (IsMultiGroupResponse, IsMultiGroupResponse(is_multi_group=True)),
]

# Combine ZMQ cases for testing with the Message union type
all_zmq_cases = zmq_request_cases + zmq_response_cases

# List of REST types and example instances
rest_cases = [
    (RESTGetSuggestionResponse, RESTGetSuggestionResponse(parameters=EXAMPLE_PARAMS, error=None)),
    (RESTGetSuggestionResponse, RESTGetSuggestionResponse(parameters=None, error="No suggestions")),
    (RESTSubmitResult, RESTSubmitResult(parameters=EXAMPLE_PARAMS, objectives=EXAMPLE_OBJECTIVES)),
    (RESTSubmitResponse, RESTSubmitResponse(success=True, error=None)),
    (RESTSubmitResponse, RESTSubmitResponse(success=False, error="Invalid data")),
    (RESTHeartbeatRequest, RESTHeartbeatRequest(worker_id=EXAMPLE_WORKER_ID)),
    (RESTHeartbeatResponse, RESTHeartbeatResponse(success=True, error=None)),
    (RESTHeartbeatResponse, RESTHeartbeatResponse(success=False, error="Unknown worker")),
    (RESTGetTrialsResponse, RESTGetTrialsResponse(trials=[{"id": 1}], error=None)),
    (RESTGetTrialsResponse, RESTGetTrialsResponse(trials=[], error="Failed to fetch")),
    (RESTGetMetadataResponse, RESTGetMetadataResponse(metadata=[{"id": 1}], error=None)),
    (RESTGetMetadataResponse, RESTGetMetadataResponse(metadata=[], error="Not found")),
    (RESTGetTopKResponse, RESTGetTopKResponse(trials=[{"id": 1}], error=None)),
    (RESTGetTopKResponse, RESTGetTopKResponse(trials=[], error="Not available")),
    (RESTIsMultiGroupResponse, RESTIsMultiGroupResponse(is_multi_group=False, error=None)),
    (RESTIsMultiGroupResponse, RESTIsMultiGroupResponse(is_multi_group=False, error="Scheduler unreachable")),
]


# --- Test Functions ---

@pytest.mark.parametrize("cls, instance", all_zmq_cases)
def test_zmq_message_msgpack_serialization(cls, instance):
    """Test msgpack serialization/deserialization for ZMQ messages."""
    encoder = msgspec.msgpack.Encoder()
    decoder = msgspec.msgpack.Decoder(Message) # Use union type for decoding

    encoded = encoder.encode(instance)
    decoded = decoder.decode(encoded)

    assert isinstance(decoded, cls)
    assert decoded == instance

@pytest.mark.parametrize("cls, instance", all_zmq_cases)
def test_zmq_message_json_serialization(cls, instance):
    """Test JSON serialization/deserialization for ZMQ messages."""
    encoder = msgspec.json.Encoder()
    decoder = msgspec.json.Decoder(Message) # Use union type for decoding

    encoded = encoder.encode(instance)
    decoded = decoder.decode(encoded)

    assert isinstance(decoded, cls)
    assert decoded == instance

@pytest.mark.parametrize("cls, instance", rest_cases)
def test_rest_message_json_serialization(cls, instance):
    """Test JSON serialization/deserialization for REST messages."""
    encoder = msgspec.json.Encoder()
    decoder = msgspec.json.Decoder(cls) # Decode with the specific REST type

    encoded = encoder.encode(instance)
    decoded = decoder.decode(encoded)

    assert isinstance(decoded, cls)
    assert decoded == instance

# Test core types separately if needed (though covered by SubmitResultRequest)
def test_result_serialization():
    """Test serialization of the core Result type."""
    instance = EXAMPLE_RESULT
    encoder_pack = msgspec.msgpack.Encoder()
    decoder_pack = msgspec.msgpack.Decoder(Result)
    encoder_json = msgspec.json.Encoder()
    decoder_json = msgspec.json.Decoder(Result)

    encoded_pack = encoder_pack.encode(instance)
    decoded_pack = decoder_pack.decode(encoded_pack)
    assert decoded_pack == instance

    encoded_json = encoder_json.encode(instance)
    decoded_json = decoder_json.decode(encoded_json)
    assert decoded_json == instance

# Add more tests for edge cases, invalid data, optional fields etc. if necessary