"""REST API server for distributed optimization."""

import threading
from multiprocessing.sharedctypes import Synchronized
from typing import Any

import msgspec
import uvicorn
import zmq
from fastapi import FastAPI, Request

from hola.core.objectives import ObjectiveName
from hola.core.parameters import ParameterName
from hola.messages.protocol import (
    GetSuggestionRequest,
    GetSuggestionResponse,
    Message,
    Result,
    StatusRequest,
    StatusResponse,
    SubmitResultRequest,
    SubmitResultResponse,
)
from hola.utils.logging import setup_logging


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


class Server:
    """HTTP server providing REST API access to the optimization system."""

    def __init__(
        self, host: str = "localhost", port: int = 8000, active_workers: Synchronized | None = None
    ):
        self.host = host
        self.port = port
        self.active_workers = active_workers

        self.context: zmq.Context = zmq.Context()
        self.socket: zmq.Socket = self.context.socket(zmq.REQ)
        self.socket.connect("ipc:///tmp/scheduler.ipc")

        self.rest_app = FastAPI()
        self.setup_rest_routes()

        self.running: bool = False
        self.logger = setup_logging("Server")

    def setup_rest_routes(self) -> None:
        @self.rest_app.get("/suggestion", response_model=None)
        async def get_job() -> bytes:
            try:
                request = GetSuggestionRequest(worker_id=-1)
                self.socket.send(msgspec.json.encode(request))

                response = msgspec.json.decode(self.socket.recv(), type=Message)
                match response:
                    case GetSuggestionResponse(parameters=params):
                        return msgspec.json.encode(RESTGetSuggestionResponse(parameters=params))
                    case _:
                        return msgspec.json.encode(
                            RESTGetSuggestionResponse(
                                parameters=None, error="Unexpected response from scheduler"
                            )
                        )

            except Exception as e:
                return msgspec.json.encode(
                    RESTGetSuggestionResponse(parameters=None, error=f"Error getting job: {str(e)}")
                )

        @self.rest_app.post("/suggestion", response_model=None)
        async def submit_result(request: Request) -> bytes:
            try:
                # Decode the raw request body using msgspec
                body = await request.body()
                result = msgspec.json.decode(body, type=RESTSubmitResult)

                request = SubmitResultRequest(
                    worker_id=-1,
                    result=Result(parameters=result.parameters, objectives=result.objectives),
                )
                self.socket.send(msgspec.json.encode(request))

                response = msgspec.json.decode(self.socket.recv(), type=Message)
                match response:
                    case SubmitResultResponse(success=success, error=error):
                        return msgspec.json.encode(RESTSubmitResponse(success=success, error=error))
                    case _:
                        return msgspec.json.encode(
                            RESTSubmitResponse(
                                success=False, error="Unexpected response from scheduler"
                            )
                        )

            except Exception as e:
                return msgspec.json.encode(
                    RESTSubmitResponse(success=False, error=f"Error submitting result: {str(e)}")
                )

        @self.rest_app.get("/status")
        async def get_status():
            try:
                self.logger.info("Received status request")

                request = StatusRequest()
                encoded_request = msgspec.json.encode(request)
                self.logger.info(f"Sending request to scheduler: {encoded_request}")

                self.socket.send(encoded_request)

                raw_response = self.socket.recv()
                self.logger.info(f"Received raw response from scheduler: {raw_response}")

                response = msgspec.json.decode(raw_response, type=Message)
                self.logger.info(f"Decoded response: {response}")

                match response:
                    case StatusResponse() as status:
                        return {
                            "active_workers": status.active_workers,
                            "total_evaluations": status.total_evaluations,
                            "best_result": {
                                "objectives": status.best_objectives
                            } if status.best_objectives else None
                        }
                    case _:
                        return {"error": "Unexpected response type"}

            except Exception as e:
                self.logger.error(f"Error handling status request: {e}")
                return {"error": str(e)}

    def start(self):
        self.running = True

        rest_thread = threading.Thread(
            target=uvicorn.run,
            args=(self.rest_app,),
            kwargs={"host": self.host, "port": self.port, "log_level": "info"},
        )
        rest_thread.daemon = True
        rest_thread.start()

        self.logger.info(f"HTTP server started on http://{self.host}:{self.port}")

    def stop(self):
        self.running = False
        self.socket.close()
        self.context.term()