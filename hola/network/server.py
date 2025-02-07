import threading

import msgspec
import uvicorn
import zmq
from fastapi import FastAPI

from hola.messages.base import Result
from hola.messages.rest import RESTGetSuggestionResponse, RESTSubmitResponse, RESTSubmitResult
from hola.messages.scheduler import (
    GetSuggestionRequest,
    GetSuggestionResponse,
    Message,
    SubmitResultRequest,
    SubmitResultResponse,
)
from hola.utils.logging import setup_logging


class Server:
    """HTTP server providing REST API access to the optimization system."""

    def __init__(self, host="localhost", port=8000, active_workers=None):
        self.host = host
        self.port = port
        self.active_workers = active_workers

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect("ipc:///tmp/scheduler.ipc")

        self.rest_app = FastAPI()
        self.setup_rest_routes()

        self.running = False
        self.logger = setup_logging("Server")

    def setup_rest_routes(self):
        @self.rest_app.get("/suggestion", response_model=RESTGetSuggestionResponse)
        async def get_job():
            try:
                request = GetSuggestionRequest(worker_id=-1)  # Use -1 for REST API requests
                self.socket.send(msgspec.json.encode(request))

                response = msgspec.json.decode(self.socket.recv(), type=Message)
                match response:
                    case GetSuggestionResponse(parameters=params):
                        return RESTGetSuggestionResponse(parameters=params)
                    case _:
                        return RESTGetSuggestionResponse(
                            parameters=None, error="Unexpected response from scheduler"
                        )

            except Exception as e:
                return RESTGetSuggestionResponse(
                    parameters=None, error=f"Error getting job: {str(e)}"
                )

        @self.rest_app.post("/result", response_model=RESTSubmitResponse)
        async def submit_result(result: RESTSubmitResult):
            try:
                request = SubmitResultRequest(
                    worker_id=-1,  # Use -1 for REST API requests
                    result=Result(parameters=result.parameters, objectives=result.objectives),
                )
                self.socket.send(msgspec.json.encode(request))

                response = msgspec.json.decode(self.socket.recv(), type=Message)
                match response:
                    case SubmitResultResponse(success=success, error=error):
                        return RESTSubmitResponse(success=success, error=error)
                    case _:
                        return RESTSubmitResponse(
                            success=False, error="Unexpected response from scheduler"
                        )

            except Exception as e:
                return RESTSubmitResponse(success=False, error=f"Error submitting result: {str(e)}")

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
