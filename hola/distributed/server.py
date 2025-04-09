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
        @self.rest_app.get("/suggestion", response_model=None)
        async def get_job() -> bytes:
            try:
                # Assign a unique negative worker ID for REST API clients
                # (to distinguish from local workers which have positive IDs)
                worker_id = -int(time.time() % 100000)
                # Use ZMQ request type from .messages
                request = GetSuggestionRequest(worker_id=worker_id)
                self.socket.send(msgspec.json.encode(request))

                # Use Message union type from .messages
                response = msgspec.json.decode(self.socket.recv(), type=Message)
                match response:
                    # Use ZMQ response type from .messages
                    case GetSuggestionResponse(parameters=params):
                        # Use REST response type from .messages
                        return msgspec.json.encode(
                            RESTGetSuggestionResponse(parameters=params)
                        )
                    case _:
                        return msgspec.json.encode(
                            RESTGetSuggestionResponse(
                                parameters=None,
                                error="Unexpected response from scheduler",
                            )
                        )

            except Exception as e:
                return msgspec.json.encode(
                    RESTGetSuggestionResponse(
                        parameters=None, error=f"Error getting job: {str(e)}"
                    )
                )

        @self.rest_app.post("/result", response_model=None)
        async def submit_result(request: Request) -> bytes:
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
                self.socket.send(msgspec.json.encode(zmq_request))

                # Get response from scheduler
                # Use Message union type from .messages
                response = msgspec.json.decode(self.socket.recv(), type=Message)

                match response:
                    # Use ZMQ response type from .messages
                    case SubmitResultResponse(
                        success=success, is_best=is_best, error=error
                    ):
                        # Use REST response type from .messages
                        return msgspec.json.encode(
                            RESTSubmitResponse(success=success, error=error)
                        )
                    case _:
                        return msgspec.json.encode(
                            RESTSubmitResponse(
                                success=False,
                                error="Unexpected response from scheduler",
                            )
                        )

            except Exception as e:
                self.logger.error(f"Error in submit_result: {e}")
                return msgspec.json.encode(
                    RESTSubmitResponse(
                        success=False, error=f"Error submitting result: {str(e)}"
                    )
                )

        @self.rest_app.post("/shutdown", response_model=None)
        async def shutdown_scheduler() -> bytes: # Renamed for clarity
            try:
                # Use ZMQ request type from .messages
                request = ShutdownRequest()
                self.socket.send(msgspec.json.encode(request))

                # Use Message union type from .messages
                response = msgspec.json.decode(self.socket.recv(), type=Message)

                match response:
                    # Use ZMQ response type from .messages
                    case SubmitResultResponse(success=success, error=error):
                         # Use REST response type from .messages
                        return msgspec.json.encode(
                            RESTSubmitResponse(success=success, error=error)
                        )
                    case _:
                        return msgspec.json.encode(
                            RESTSubmitResponse(
                                success=False,
                                error="Unexpected response from scheduler",
                            )
                        )
            except Exception as e:
                self.logger.error(f"Error in shutdown request: {e}")
                return msgspec.json.encode(
                    RESTSubmitResponse(
                        success=False, error=f"Error shutting down: {str(e)}"
                    )
                )

        @self.rest_app.post("/heartbeat", response_model=None)
        async def send_heartbeat(request: Request) -> bytes:
            try:
                body = await request.body()
                 # Use REST request type from .messages
                heartbeat_req = msgspec.json.decode(body, type=RESTHeartbeatRequest)

                worker_id = heartbeat_req.worker_id
                # Use ZMQ request type from .messages
                zmq_request = HeartbeatRequest(worker_id=worker_id)
                self.socket.send(msgspec.json.encode(zmq_request))

                # Use Message union type from .messages
                response = msgspec.json.decode(self.socket.recv(), type=Message)

                match response:
                     # Use ZMQ response type from .messages
                    case HeartbeatResponse(success=success):
                         # Use REST response type from .messages
                        return msgspec.json.encode(
                            RESTHeartbeatResponse(success=success)
                        )
                    case _:
                        return msgspec.json.encode(
                            RESTHeartbeatResponse(
                                success=False,
                                error="Unexpected response from scheduler"
                            )
                        )
            except Exception as e:
                self.logger.error(f"Error in heartbeat: {e}")
                return msgspec.json.encode(
                    RESTHeartbeatResponse(
                        success=False, error=f"Error sending heartbeat: {str(e)}"
                    )
                )

        @self.rest_app.get("/trials", response_model=None)
        async def get_trials(ranked_only: bool = True) -> bytes:
            try:
                # Use ZMQ request type from .messages
                request = GetTrialsRequest(ranked_only=ranked_only)
                self.socket.send(msgspec.json.encode(request))

                # Use Message union type from .messages
                response = msgspec.json.decode(self.socket.recv(), type=Message)

                match response:
                    # Use ZMQ response type from .messages
                    case GetTrialsResponse(trials=trials):
                         # Use REST response type from .messages
                        return msgspec.json.encode(
                            RESTGetTrialsResponse(trials=trials)
                        )
                    case _:
                        return msgspec.json.encode(
                            RESTGetTrialsResponse(
                                trials=[],
                                error="Unexpected response from scheduler"
                            )
                        )
            except Exception as e:
                self.logger.error(f"Error getting trials: {e}")
                return msgspec.json.encode(
                    RESTGetTrialsResponse(trials=[], error=f"Error: {str(e)}")
                )

        @self.rest_app.get("/metadata", response_model=None)
        async def get_metadata(trial_ids: Optional[str] = None) -> bytes:
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
                        return msgspec.json.encode(
                            RESTGetMetadataResponse(
                                metadata=[],
                                error="Invalid trial_ids format: must be an integer or comma-separated integers"
                            )
                        )
                # Use ZMQ request type from .messages
                request = GetMetadataRequest(trial_ids=parsed_trial_ids)
                self.socket.send(msgspec.json.encode(request))

                # Use Message union type from .messages
                response = msgspec.json.decode(self.socket.recv(), type=Message)

                match response:
                    # Use ZMQ response type from .messages
                    case GetMetadataResponse(metadata=metadata):
                         # Use REST response type from .messages
                        return msgspec.json.encode(
                            RESTGetMetadataResponse(metadata=metadata)
                        )
                    case _:
                        return msgspec.json.encode(
                            RESTGetMetadataResponse(
                                metadata=[],
                                error="Unexpected response from scheduler"
                            )
                        )
            except Exception as e:
                self.logger.error(f"Error getting metadata: {e}")
                return msgspec.json.encode(
                    RESTGetMetadataResponse(metadata=[], error=f"Error: {str(e)}")
                )

        @self.rest_app.get("/top", response_model=None)
        async def get_top_k(k: int = 1) -> bytes:
            try:
                # Use ZMQ request type from .messages
                request = GetTopKRequest(k=k)
                self.socket.send(msgspec.json.encode(request))

                # Use Message union type from .messages
                response = msgspec.json.decode(self.socket.recv(), type=Message)

                match response:
                    # Use ZMQ response type from .messages
                    case GetTopKResponse(trials=trials):
                         # Use REST response type from .messages
                        return msgspec.json.encode(
                            RESTGetTopKResponse(trials=trials)
                        )
                    case _:
                        return msgspec.json.encode(
                            RESTGetTopKResponse(
                                trials=[],
                                error="Unexpected response from scheduler"
                            )
                        )
            except Exception as e:
                self.logger.error(f"Error getting top k trials: {e}")
                return msgspec.json.encode(
                    RESTGetTopKResponse(trials=[], error=f"Error: {str(e)}")
                )

        @self.rest_app.get("/is_multi_group", response_model=None)
        async def is_multi_group() -> bytes:
            try:
                # Use ZMQ request type from .messages
                request = IsMultiGroupRequest()
                self.socket.send(msgspec.json.encode(request))

                # Use Message union type from .messages
                response = msgspec.json.decode(self.socket.recv(), type=Message)

                match response:
                    # Use ZMQ response type from .messages
                    case IsMultiGroupResponse(is_multi_group=is_multi):
                         # Use REST response type from .messages
                        return msgspec.json.encode(
                            RESTIsMultiGroupResponse(is_multi_group=is_multi)
                        )
                    case _:
                        return msgspec.json.encode(
                            RESTIsMultiGroupResponse(
                                is_multi_group=False,
                                error="Unexpected response from scheduler"
                            )
                        )
            except Exception as e:
                self.logger.error(f"Error checking multi group: {e}")
                return msgspec.json.encode(
                    RESTIsMultiGroupResponse(is_multi_group=False, error=f"Error: {str(e)}")
                )

        @self.rest_app.get("/history")
        async def get_history():
            """Endpoint to get optimization history (simplified status)."""
            try:
                # Get current status from scheduler
                # Use ZMQ request type from .messages
                self.socket.send(msgspec.json.encode(StatusRequest()))
                 # Use Message union type from .messages
                status_response = msgspec.json.decode(self.socket.recv(), type=Message)

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
                self.socket.send(msgspec.json.encode(request))

                # Use Message union type from .messages
                response = msgspec.json.decode(self.socket.recv(), type=Message)

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