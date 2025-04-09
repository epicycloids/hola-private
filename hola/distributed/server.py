"""
Contains the Server class which provides a FastAPI interface to the scheduler.
"""

import logging
import threading
import time
from typing import Any, List, Dict, Optional

import msgspec
import zmq
import uvicorn
from fastapi import FastAPI, Request

# Use relative imports for messages and utils within the same package
from .messages import (
    Message,
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
    RESTGetSuggestionResponse, # REST specific
    RESTSubmitResult,          # REST specific
    RESTSubmitResponse,        # REST specific
    RESTHeartbeatRequest,      # REST specific
    RESTHeartbeatResponse,     # REST specific
    RESTGetTrialsResponse,     # REST specific
    RESTGetMetadataResponse,   # REST specific
    RESTGetTopKResponse,       # REST specific
    RESTIsMultiGroupResponse,  # REST specific
)
from .utils import setup_logging


class Server:
    """HTTP server providing REST API access to the optimization system."""

    def __init__(self, host: str = "localhost", port: int = 8000):
        self.host = host
        self.port = port

        self.context: zmq.Context = zmq.Context()
        self.socket: zmq.Socket = self.context.socket(zmq.REQ)
        self.socket.connect("ipc:///tmp/scheduler.ipc") # TODO: Make configurable

        self.rest_app = FastAPI()
        self.setup_rest_routes()

        self.running: bool = False
        # Use setup_logging from .utils
        self.logger = setup_logging("Server")

        # Server thread
        self.server_thread = None

    def setup_rest_routes(self) -> None:
        @self.rest_app.get("/suggestion")
        async def get_job():
            try:
                # Assign a unique negative worker ID for REST API clients
                # (to distinguish from local workers which have positive IDs)
                worker_id = -int(time.time() % 100000)
                # Use ZMQ request type from .messages
                request = GetSuggestionRequest(worker_id=worker_id)
                # Use msgpack for ZMQ socket
                self.socket.send(msgspec.msgpack.encode(request))

                # Use msgpack for ZMQ socket
                response = msgspec.msgpack.decode(self.socket.recv(), type=Message)
                match response:
                    # Use ZMQ response type from .messages
                    case GetSuggestionResponse(parameters=params):
                        rest_response = RESTGetSuggestionResponse(parameters=params)
                        return msgspec.to_builtins(rest_response)
                    case _:
                        rest_response = RESTGetSuggestionResponse(
                            parameters=None, error="Unexpected response from scheduler"
                        )
                        return msgspec.to_builtins(rest_response)

            except Exception as e:
                rest_response = RESTGetSuggestionResponse(
                    parameters=None, error=f"Error getting job: {str(e)}"
                )
                return msgspec.to_builtins(rest_response)

        @self.rest_app.post("/result")
        async def submit_result(request: Request):
            try:
                # Decode the raw request body using msgspec
                body = await request.body()
                # Use REST request type from .messages
                result_req = msgspec.json.decode(body, type=RESTSubmitResult)

                # Create result request with a unique negative worker ID
                worker_id = -int(time.time() % 100000)
                # Use core Result type from .messages
                result_obj = Result(
                    parameters=result_req.parameters, objectives=result_req.objectives
                )
                # Use ZMQ request type from .messages
                zmq_request = SubmitResultRequest(worker_id=worker_id, result=result_obj)
                # Use msgpack for ZMQ socket
                self.socket.send(msgspec.msgpack.encode(zmq_request))

                # Get response from scheduler
                # Use msgpack for ZMQ socket
                response = msgspec.msgpack.decode(self.socket.recv(), type=Message)

                match response:
                    # Use ZMQ response type from .messages
                    case SubmitResultResponse(
                        success=success, is_best=is_best, error=error
                    ):
                        rest_response = RESTSubmitResponse(success=success, error=error)
                        return msgspec.to_builtins(rest_response)
                    case _:
                        rest_response = RESTSubmitResponse(
                            success=False,
                            error="Unexpected response from scheduler",
                        )
                        return msgspec.to_builtins(rest_response)

            except Exception as e:
                self.logger.error(f"Error in submit_result: {e}")
                rest_response = RESTSubmitResponse(
                    success=False, error=f"Error submitting result: {str(e)}"
                )
                return msgspec.to_builtins(rest_response)

        @self.rest_app.post("/shutdown")
        async def shutdown_scheduler():
            try:
                # Use ZMQ request type from .messages
                request = ShutdownRequest()
                # Use msgpack for ZMQ socket
                self.socket.send(msgspec.msgpack.encode(request))

                # Use msgpack for ZMQ socket
                response = msgspec.msgpack.decode(self.socket.recv(), type=Message)

                match response:
                    # Use ZMQ response type from .messages
                    case SubmitResultResponse(success=success, error=error):
                        rest_response = RESTSubmitResponse(success=success, error=error)
                        return msgspec.to_builtins(rest_response)
                    case _:
                        rest_response = RESTSubmitResponse(
                            success=False,
                            error="Unexpected response from scheduler",
                        )
                        return msgspec.to_builtins(rest_response)
            except Exception as e:
                self.logger.error(f"Error in shutdown request: {e}")
                rest_response = RESTSubmitResponse(
                    success=False, error=f"Error shutting down: {str(e)}"
                )
                return msgspec.to_builtins(rest_response)

        @self.rest_app.post("/heartbeat")
        async def send_heartbeat(request: Request):
            try:
                body = await request.body()
                 # Use REST request type from .messages
                heartbeat_req = msgspec.json.decode(body, type=RESTHeartbeatRequest)

                worker_id = heartbeat_req.worker_id
                # Use ZMQ request type from .messages
                zmq_request = HeartbeatRequest(worker_id=worker_id)
                # Use msgpack for ZMQ socket
                self.socket.send(msgspec.msgpack.encode(zmq_request))

                # Use msgpack for ZMQ socket
                response = msgspec.msgpack.decode(self.socket.recv(), type=Message)

                match response:
                     # Use ZMQ response type from .messages
                    case HeartbeatResponse(success=success):
                        rest_response = RESTHeartbeatResponse(success=success)
                        return msgspec.to_builtins(rest_response)
                    case _:
                        rest_response = RESTHeartbeatResponse(
                            success=False,
                            error="Unexpected response from scheduler"
                        )
                        return msgspec.to_builtins(rest_response)
            except Exception as e:
                self.logger.error(f"Error in heartbeat: {e}")
                rest_response = RESTHeartbeatResponse(
                    success=False, error=f"Error sending heartbeat: {str(e)}"
                )
                return msgspec.to_builtins(rest_response)

        @self.rest_app.get("/trials")
        async def get_trials(ranked_only: bool = True):
            try:
                # Use ZMQ request type from .messages
                request = GetTrialsRequest(ranked_only=ranked_only)
                # Use msgpack for ZMQ socket
                self.socket.send(msgspec.msgpack.encode(request))

                # Use msgpack for ZMQ socket
                response = msgspec.msgpack.decode(self.socket.recv(), type=Message)

                match response:
                    # Use ZMQ response type from .messages
                    case GetTrialsResponse(trials=trials):
                        rest_response = RESTGetTrialsResponse(trials=trials)
                        return msgspec.to_builtins(rest_response)
                    case _:
                        rest_response = RESTGetTrialsResponse(
                            trials=[],
                            error="Unexpected response from scheduler"
                        )
                        return msgspec.to_builtins(rest_response)
            except Exception as e:
                self.logger.error(f"Error getting trials: {e}")
                rest_response = RESTGetTrialsResponse(trials=[], error=f"Error: {str(e)}")
                return msgspec.to_builtins(rest_response)

        @self.rest_app.get("/metadata")
        async def get_metadata(trial_ids: Optional[str] = None):
            try:
                # Convert string parameter to list of integers if provided
                parsed_trial_ids = None
                if trial_ids:
                    try:
                        # Handle both single int and comma-separated list
                        if ',' in trial_ids:
                            parsed_trial_ids = [int(id.strip()) for id in trial_ids.split(',')]
                        else:
                            parsed_trial_ids = int(trial_ids)
                    except ValueError:
                        rest_response = RESTGetMetadataResponse(
                            metadata=[],
                            error="Invalid trial_ids format: must be an integer or comma-separated integers"
                        )
                        return msgspec.to_builtins(rest_response)
                # Use ZMQ request type from .messages
                request = GetMetadataRequest(trial_ids=parsed_trial_ids)
                # Use msgpack for ZMQ socket
                self.socket.send(msgspec.msgpack.encode(request))

                # Use msgpack for ZMQ socket
                response = msgspec.msgpack.decode(self.socket.recv(), type=Message)

                match response:
                    # Use ZMQ response type from .messages
                    case GetMetadataResponse(metadata=metadata):
                        rest_response = RESTGetMetadataResponse(metadata=metadata)
                        return msgspec.to_builtins(rest_response)
                    case _:
                        rest_response = RESTGetMetadataResponse(
                            metadata=[],
                            error="Unexpected response from scheduler"
                        )
                        return msgspec.to_builtins(rest_response)
            except Exception as e:
                self.logger.error(f"Error getting metadata: {e}")
                rest_response = RESTGetMetadataResponse(metadata=[], error=f"Error: {str(e)}")
                return msgspec.to_builtins(rest_response)

        @self.rest_app.get("/top")
        async def get_top_k(k: int = 1):
            try:
                # Use ZMQ request type from .messages
                request = GetTopKRequest(k=k)
                # Use msgpack for ZMQ socket
                self.socket.send(msgspec.msgpack.encode(request))

                # Use msgpack for ZMQ socket
                response = msgspec.msgpack.decode(self.socket.recv(), type=Message)

                match response:
                    # Use ZMQ response type from .messages
                    case GetTopKResponse(trials=trials):
                        rest_response = RESTGetTopKResponse(trials=trials)
                        return msgspec.to_builtins(rest_response)
                    case _:
                        rest_response = RESTGetTopKResponse(
                            trials=[],
                            error="Unexpected response from scheduler"
                        )
                        return msgspec.to_builtins(rest_response)
            except Exception as e:
                self.logger.error(f"Error getting top k trials: {e}")
                rest_response = RESTGetTopKResponse(trials=[], error=f"Error: {str(e)}")
                return msgspec.to_builtins(rest_response)

        @self.rest_app.get("/is_multi_group")
        async def is_multi_group():
            try:
                # Use ZMQ request type from .messages
                request = IsMultiGroupRequest()
                # Use msgpack for ZMQ socket
                self.socket.send(msgspec.msgpack.encode(request))

                # Use msgpack for ZMQ socket
                response = msgspec.msgpack.decode(self.socket.recv(), type=Message)

                match response:
                    # Use ZMQ response type from .messages
                    case IsMultiGroupResponse(is_multi_group=is_multi):
                        rest_response = RESTIsMultiGroupResponse(is_multi_group=is_multi)
                        return msgspec.to_builtins(rest_response)
                    case _:
                        rest_response = RESTIsMultiGroupResponse(
                            is_multi_group=False,
                            error="Unexpected response from scheduler"
                        )
                        return msgspec.to_builtins(rest_response)
            except Exception as e:
                self.logger.error(f"Error checking multi group: {e}")
                rest_response = RESTIsMultiGroupResponse(is_multi_group=False, error=f"Error: {str(e)}")
                return msgspec.to_builtins(rest_response)

        @self.rest_app.get("/history")
        async def get_history():
            """Endpoint to get optimization history (simplified status)."""
            try:
                # Get current status from scheduler
                # Use ZMQ request type from .messages
                # Use msgpack for ZMQ socket
                self.socket.send(msgspec.msgpack.encode(StatusRequest()))
                 # Use msgpack for ZMQ socket
                status_response = msgspec.msgpack.decode(self.socket.recv(), type=Message)

                if not isinstance(status_response, StatusResponse):
                    return {"error": "Failed to get status from scheduler"}

                # Return just the current status information
                history_data = {
                    "total_evaluations": status_response.total_evaluations,
                    "active_workers": status_response.active_workers,
                    "best_objectives": status_response.best_objectives,
                }

                return {"history": history_data}
            except Exception as e:
                self.logger.error(f"Error handling history request: {e}")
                return {"error": str(e)}

        @self.rest_app.get("/status")
        async def get_status():
            """Endpoint to get detailed current status."""
            try:
                self.logger.info("Received status request")
                # Use ZMQ request type from .messages
                request = StatusRequest()
                # Use msgpack for ZMQ socket
                self.socket.send(msgspec.msgpack.encode(request))

                # Use msgpack for ZMQ socket
                response = msgspec.msgpack.decode(self.socket.recv(), type=Message)

                match response:
                    # Use ZMQ response type from .messages
                    case StatusResponse() as status:
                        return {
                            "active_workers": status.active_workers,
                            "total_evaluations": status.total_evaluations,
                            "best_result": {"objectives": status.best_objectives}
                            if status.best_objectives
                            else None,
                        }
                    case _:
                        return {"error": "Unexpected response type"}

            except Exception as e:
                self.logger.error(f"Error handling status request: {e}")
                return {"error": str(e)}

    def start(self):
        self.running = True

        def run_server():
            config = uvicorn.Config(
                app=self.rest_app, host=self.host, port=self.port, log_level="info"
            )
            server = uvicorn.Server(config)
            try:
                server.run()
            except Exception as e:
                self.logger.error(f"Server error: {e}")

        self.server_thread = threading.Thread(target=run_server)
        self.server_thread.daemon = True
        self.server_thread.start()

        self.logger.info(f"HTTP server started on http://{self.host}:{self.port}")

    def stop(self):
        self.running = False
        # TODO: Need a way to gracefully shut down the Uvicorn server
        # Currently, relies on the thread being a daemon.
        # Consider using server.should_exit flag or similar if Uvicorn supports it.
        self.socket.close()
        self.context.term()
        self.logger.info("Server stopped")