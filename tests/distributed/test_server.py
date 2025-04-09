"""Tests for hola.distributed.server (Server class with FastAPI)."""

import logging
import pytest
from unittest.mock import patch, MagicMock
import msgspec
import zmq
from fastapi.testclient import TestClient

# Assuming the tests directory is at the same level as hola/
from hola.distributed.server import Server
from hola.distributed.messages import (
    Message,
    Result, # Need Result for POST /result
    ParameterName, ObjectiveName, # Needed for Result
    StatusRequest, StatusResponse,
    GetSuggestionRequest, GetSuggestionResponse, RESTGetSuggestionResponse, # Suggestion
    SubmitResultRequest, SubmitResultResponse, RESTSubmitResult, RESTSubmitResponse, # Result
    ShutdownRequest, # Shutdown (uses SubmitResultResponse for ZMQ ack)
    HeartbeatRequest, HeartbeatResponse, RESTHeartbeatRequest, RESTHeartbeatResponse, # Heartbeat
    GetTrialsRequest, GetTrialsResponse, RESTGetTrialsResponse, # Trials
    GetMetadataRequest, GetMetadataResponse, RESTGetMetadataResponse, # Metadata
    GetTopKRequest, GetTopKResponse, RESTGetTopKResponse, # TopK
    IsMultiGroupRequest, IsMultiGroupResponse, RESTIsMultiGroupResponse # IsMultiGroup
)

# --- Fixtures ---

@pytest.fixture
def mock_zmq_server():
    """Fixture for mocking ZMQ context and socket used by Server."""
    with patch('zmq.Context', autospec=True) as mock_context_cls:
        mock_context_instance = mock_context_cls.return_value
        mock_socket_req = MagicMock(spec=zmq.Socket)
        mock_context_instance.socket.return_value = mock_socket_req
        yield mock_context_instance, mock_socket_req

@pytest.fixture
def mock_setup_logging_server():
    """Fixture for mocking setup_logging used by Server."""
    with patch('hola.distributed.server.setup_logging', autospec=True) as mock_setup:
        mock_logger = MagicMock(spec=logging.Logger)
        mock_setup.return_value = mock_logger
        yield mock_logger, mock_setup

@pytest.fixture
def server_instance(mock_zmq_server, mock_setup_logging_server):
    """Fixture to create a Server instance with mocked dependencies."""
    # Don't need to mock uvicorn/threading for TestClient usage
    # Patch os.path.abspath if needed by setup_logging internal logic
    with patch('hola.distributed.utils.os.path.abspath'):
        server = Server(host="test.local", port=8888)
        # Replace logger instance after init
        server.logger = mock_setup_logging_server[0]
        # Replace zmq socket instance used by endpoints
        server.context = mock_zmq_server[0]
        server.socket = mock_zmq_server[1]
        yield server # Provide the configured server instance

@pytest.fixture
def test_client(server_instance):
    """Fixture to create a FastAPI TestClient for the server instance."""
    client = TestClient(server_instance.rest_app)
    yield client

# --- Test Cases ---

def test_server_init(mock_setup_logging_server, mock_zmq_server):
    """Test basic Server initialization."""
    mock_logger, mock_setup = mock_setup_logging_server
    mock_context, mock_socket = mock_zmq_server

    server = Server(host="test.local", port=8888)

    assert server.host == "test.local"
    assert server.port == 8888
    assert server.context is mock_context
    mock_context.socket.assert_called_once_with(zmq.REQ)
    mock_socket.connect.assert_called_once_with("ipc:///tmp/scheduler.ipc")
    # Check logger setup was called
    # Note: setup_logging is called *after* zmq setup in the original code
    # This assertion depends on the mock fixture patching the correct path
    mock_setup.assert_called_once_with("Server")
    assert server.server_thread is None # Thread not started yet

def test_get_status_endpoint(test_client, server_instance): # Uses fixtures
    """Test the GET /status endpoint."""
    mock_socket = server_instance.socket # Get the mocked socket from server

    # Configure mock ZMQ socket response for StatusRequest
    status_data = {
        "active_workers": 2,
        "total_evaluations": 50,
        "best_objectives": {"o1": 10.0}
    }
    status_response_msg = StatusResponse(**status_data)
    encoded_response = msgspec.msgpack.encode(status_response_msg)
    mock_socket.recv.return_value = encoded_response

    # Make request using TestClient
    response = test_client.get("/status")

    # Assertions
    assert response.status_code == 200
    # Check ZMQ socket interactions
    expected_request_msg = StatusRequest()
    encoded_request = msgspec.msgpack.encode(expected_request_msg)
    mock_socket.send.assert_called_once_with(encoded_request)
    mock_socket.recv.assert_called_once()
    # Check JSON response body
    expected_json = {
        "active_workers": 2,
        "total_evaluations": 50,
        "best_result": {"objectives": {"o1": 10.0}}
    }
    assert response.json() == expected_json
    server_instance.logger.info.assert_any_call("Received status request")

def test_get_suggestion_endpoint(test_client, server_instance):
    """Test the GET /suggestion endpoint."""
    mock_socket = server_instance.socket
    params = {"p1": 1.0}

    # Mock ZMQ response
    zmq_response_msg = GetSuggestionResponse(parameters=params)
    encoded_zmq_response = msgspec.msgpack.encode(zmq_response_msg)
    mock_socket.recv.return_value = encoded_zmq_response

    # Make request
    response = test_client.get("/suggestion")

    # Assertions
    assert response.status_code == 200
    # Check ZMQ send (request worker_id is variable, use ANY)
    mock_socket.send.assert_called_once()
    sent_data = mock_socket.send.call_args[0][0]
    decoded_sent_req = msgspec.msgpack.decode(sent_data, type=GetSuggestionRequest)
    assert isinstance(decoded_sent_req, GetSuggestionRequest)
    assert isinstance(decoded_sent_req.worker_id, int)
    assert decoded_sent_req.worker_id < 0 # Server uses negative IDs
    mock_socket.recv.assert_called_once()
    # Check REST response
    expected_rest_response = RESTGetSuggestionResponse(parameters=params)
    assert response.json() == msgspec.to_builtins(expected_rest_response)

def test_post_result_endpoint(test_client, server_instance):
    """Test the POST /result endpoint."""
    mock_socket = server_instance.socket
    params = {"p1": 2.0}
    objectives = {"o1": 3.0}
    rest_request_body = RESTSubmitResult(parameters=params, objectives=objectives)

    # Mock ZMQ response
    zmq_response_msg = SubmitResultResponse(success=True, is_best=True, error=None)
    encoded_zmq_response = msgspec.msgpack.encode(zmq_response_msg)
    mock_socket.recv.return_value = encoded_zmq_response

    # Make request
    response = test_client.post("/result", content=msgspec.json.encode(rest_request_body))

    # Assertions
    assert response.status_code == 200
    # Check ZMQ send
    mock_socket.send.assert_called_once()
    sent_data = mock_socket.send.call_args[0][0]
    decoded_sent_req = msgspec.msgpack.decode(sent_data, type=SubmitResultRequest)
    assert isinstance(decoded_sent_req, SubmitResultRequest)
    assert decoded_sent_req.result.parameters == params
    assert decoded_sent_req.result.objectives == objectives
    assert decoded_sent_req.worker_id < 0 # Server uses negative IDs
    mock_socket.recv.assert_called_once()
    # Check REST response
    expected_rest_response = RESTSubmitResponse(success=True, error=None)
    assert response.json() == msgspec.to_builtins(expected_rest_response)

def test_post_shutdown_endpoint(test_client, server_instance):
    """Test the POST /shutdown endpoint."""
    mock_socket = server_instance.socket

    # Mock ZMQ response (Scheduler acks shutdown with SubmitResultResponse)
    zmq_response_msg = SubmitResultResponse(success=True)
    encoded_zmq_response = msgspec.msgpack.encode(zmq_response_msg)
    mock_socket.recv.return_value = encoded_zmq_response

    # Make request
    response = test_client.post("/shutdown")

    # Assertions
    assert response.status_code == 200
    # Check ZMQ send
    expected_zmq_request = ShutdownRequest()
    encoded_zmq_request = msgspec.msgpack.encode(expected_zmq_request)
    mock_socket.send.assert_called_once_with(encoded_zmq_request)
    mock_socket.recv.assert_called_once()
    # Check REST response
    expected_rest_response = RESTSubmitResponse(success=True)
    assert response.json() == msgspec.to_builtins(expected_rest_response)

def test_post_heartbeat_endpoint(test_client, server_instance):
    """Test the POST /heartbeat endpoint."""
    mock_socket = server_instance.socket
    worker_id = 987
    rest_request_body = RESTHeartbeatRequest(worker_id=worker_id)

    # Mock ZMQ response
    zmq_response_msg = HeartbeatResponse(success=True)
    encoded_zmq_response = msgspec.msgpack.encode(zmq_response_msg)
    mock_socket.recv.return_value = encoded_zmq_response

    # Make request
    response = test_client.post("/heartbeat", content=msgspec.json.encode(rest_request_body))

    # Assertions
    assert response.status_code == 200
    # Check ZMQ send
    expected_zmq_request = HeartbeatRequest(worker_id=worker_id)
    encoded_zmq_request = msgspec.msgpack.encode(expected_zmq_request)
    mock_socket.send.assert_called_once_with(encoded_zmq_request)
    mock_socket.recv.assert_called_once()
    # Check REST response
    expected_rest_response = RESTHeartbeatResponse(success=True)
    assert response.json() == msgspec.to_builtins(expected_rest_response)

def test_get_trials_endpoint(test_client, server_instance):
    """Test the GET /trials endpoint."""
    mock_socket = server_instance.socket
    trials_data = [{"id": 1, "p": "a"}, {"id": 2, "p": "b"}]

    # Mock ZMQ response
    zmq_response_msg = GetTrialsResponse(trials=trials_data)
    encoded_zmq_response = msgspec.msgpack.encode(zmq_response_msg)
    mock_socket.recv.return_value = encoded_zmq_response

    # Make request (default ranked_only=True)
    response = test_client.get("/trials")

    # Assertions
    assert response.status_code == 200
    # Check ZMQ send
    expected_zmq_request = GetTrialsRequest(ranked_only=True)
    encoded_zmq_request = msgspec.msgpack.encode(expected_zmq_request)
    mock_socket.send.assert_called_once_with(encoded_zmq_request)
    mock_socket.recv.assert_called_once()
    # Check REST response
    expected_rest_response = RESTGetTrialsResponse(trials=trials_data)
    assert response.json() == msgspec.to_builtins(expected_rest_response)

def test_get_metadata_endpoint(test_client, server_instance):
    """Test the GET /metadata endpoint."""
    mock_socket = server_instance.socket
    metadata_data = [{"id": 1, "m": "x"}, {"id": 2, "m": "y"}]
    trial_ids_param = "1,2"

    # Mock ZMQ response
    zmq_response_msg = GetMetadataResponse(metadata=metadata_data)
    encoded_zmq_response = msgspec.msgpack.encode(zmq_response_msg)
    mock_socket.recv.return_value = encoded_zmq_response

    # Make request
    response = test_client.get(f"/metadata?trial_ids={trial_ids_param}")

    # Assertions
    assert response.status_code == 200
    # Check ZMQ send (server parses param into list)
    expected_zmq_request = GetMetadataRequest(trial_ids=[1, 2])
    encoded_zmq_request = msgspec.msgpack.encode(expected_zmq_request)
    mock_socket.send.assert_called_once_with(encoded_zmq_request)
    mock_socket.recv.assert_called_once()
    # Check REST response
    expected_rest_response = RESTGetMetadataResponse(metadata=metadata_data)
    assert response.json() == msgspec.to_builtins(expected_rest_response)

def test_get_top_k_endpoint(test_client, server_instance):
    """Test the GET /top endpoint."""
    mock_socket = server_instance.socket
    top_k_data = [{"id": 10, "score": 0.1}]
    k_param = 3

    # Mock ZMQ response
    zmq_response_msg = GetTopKResponse(trials=top_k_data)
    encoded_zmq_response = msgspec.msgpack.encode(zmq_response_msg)
    mock_socket.recv.return_value = encoded_zmq_response

    # Make request
    response = test_client.get(f"/top?k={k_param}")

    # Assertions
    assert response.status_code == 200
    # Check ZMQ send
    expected_zmq_request = GetTopKRequest(k=k_param)
    encoded_zmq_request = msgspec.msgpack.encode(expected_zmq_request)
    mock_socket.send.assert_called_once_with(encoded_zmq_request)
    mock_socket.recv.assert_called_once()
    # Check REST response
    expected_rest_response = RESTGetTopKResponse(trials=top_k_data)
    assert response.json() == msgspec.to_builtins(expected_rest_response)

def test_is_multi_group_endpoint(test_client, server_instance):
    """Test the GET /is_multi_group endpoint."""
    mock_socket = server_instance.socket
    is_multi = False

    # Mock ZMQ response
    zmq_response_msg = IsMultiGroupResponse(is_multi_group=is_multi)
    encoded_zmq_response = msgspec.msgpack.encode(zmq_response_msg)
    mock_socket.recv.return_value = encoded_zmq_response

    # Make request
    response = test_client.get("/is_multi_group")

    # Assertions
    assert response.status_code == 200
    # Check ZMQ send
    expected_zmq_request = IsMultiGroupRequest()
    encoded_zmq_request = msgspec.msgpack.encode(expected_zmq_request)
    mock_socket.send.assert_called_once_with(encoded_zmq_request)
    mock_socket.recv.assert_called_once()
    # Check REST response
    expected_rest_response = RESTIsMultiGroupResponse(is_multi_group=is_multi)
    assert response.json() == msgspec.to_builtins(expected_rest_response)

def test_get_history_endpoint(test_client, server_instance):
    """Test the GET /history endpoint."""
    mock_socket = server_instance.socket

    # Mock ZMQ response for StatusRequest
    status_data = {
        "active_workers": 1,
        "total_evaluations": 25,
        "best_objectives": {"o2": 5.5}
    }
    status_response_msg = StatusResponse(**status_data)
    encoded_response = msgspec.msgpack.encode(status_response_msg)
    mock_socket.recv.return_value = encoded_response

    # Make request
    response = test_client.get("/history")

    # Assertions
    assert response.status_code == 200
    # Check ZMQ socket interactions
    expected_request_msg = StatusRequest()
    encoded_request = msgspec.msgpack.encode(expected_request_msg)
    mock_socket.send.assert_called_once_with(encoded_request)
    mock_socket.recv.assert_called_once()
    # Check JSON response body
    expected_json = {
        "history": {
            "total_evaluations": 25,
            "active_workers": 1,
            "best_objectives": {"o2": 5.5}
        }
    }
    assert response.json() == expected_json

# --- Error Handling Tests ---

def test_get_status_endpoint_zmq_send_error(test_client, server_instance):
    """Test GET /status when ZMQ send fails."""
    mock_socket = server_instance.socket
    mock_socket.send.side_effect = zmq.ZMQError("Send failed")

    response = test_client.get("/status")

    assert response.status_code == 200 # Endpoint handles error internally
    # Check that the error is the raw ZMQError message
    assert response.json() == {"error": "Send failed"}
    mock_socket.send.assert_called_once() # Send was attempted
    mock_socket.recv.assert_not_called() # Recv should not be called

def test_get_status_endpoint_zmq_recv_error(test_client, server_instance):
    """Test GET /status when ZMQ recv fails."""
    mock_socket = server_instance.socket
    mock_socket.recv.side_effect = zmq.ZMQError("Recv failed")

    response = test_client.get("/status")

    assert response.status_code == 200
    # Check that the error is the raw ZMQError message
    assert response.json() == {"error": "Recv failed"}
    mock_socket.send.assert_called_once()
    mock_socket.recv.assert_called_once() # Recv was attempted

def test_get_status_endpoint_decode_error(test_client, server_instance):
    """Test GET /status when ZMQ response decoding fails."""
    mock_socket = server_instance.socket
    # Return garbage bytes that cannot be decoded
    mock_socket.recv.return_value = b'\x01\x02\x03\xff'

    response = test_client.get("/status")

    assert response.status_code == 200
    # Check the actual error message format from msgspec
    error_msg = response.json()["error"]
    assert "Expected `object`" in error_msg # Check for characteristic msgspec error
    mock_socket.send.assert_called_once()
    mock_socket.recv.assert_called_once()

def test_post_result_endpoint_bad_body(test_client, server_instance):
    """Test POST /result with invalid request body."""
    mock_socket = server_instance.socket

    # Send invalid JSON content
    response = test_client.post("/result", content="this is not json")

    assert response.status_code == 200 # Endpoint handles error internally
    # Check the actual error message format
    error_msg = response.json()["error"]
    assert error_msg.startswith("Error submitting result: JSON is malformed")
    # ZMQ socket should not have been used
    mock_socket.send.assert_not_called()
    mock_socket.recv.assert_not_called()

def test_post_result_endpoint_zmq_error(test_client, server_instance):
    """Test POST /result when ZMQ interaction fails."""
    mock_socket = server_instance.socket
    params = {"p1": 2.0}
    objectives = {"o1": 3.0}
    rest_request_body = RESTSubmitResult(parameters=params, objectives=objectives)

    # Simulate ZMQ send error
    mock_socket.send.side_effect = zmq.ZMQError("Cannot reach scheduler")

    response = test_client.post("/result", content=msgspec.json.encode(rest_request_body))

    assert response.status_code == 200
    assert "Error submitting result: Cannot reach scheduler" in response.json()["error"]
    mock_socket.send.assert_called_once() # Send was attempted
    mock_socket.recv.assert_not_called()

# TODO: Add more tests for error cases (e.g., ZMQ socket error, bad request data)
