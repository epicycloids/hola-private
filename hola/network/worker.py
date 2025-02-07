from typing import Any

import msgspec
import zmq

from hola.core.parameters import ParameterName
from hola.messages.base import Result
from hola.messages.scheduler import (
    GetSuggestionRequest,
    GetSuggestionResponse,
    Message,
    SubmitResultRequest,
    SubmitResultResponse,
)
from hola.utils.logging import setup_logging


class LocalWorker:
    """Worker process that executes objective function evaluations."""

    def __init__(
        self,
        worker_id: int,
        active_workers,
        evaluation_fn: callable,  # New parameter
        use_ipc: bool = True,
    ):
        self.worker_id = worker_id
        self.active_workers = active_workers
        self.evaluation_fn = evaluation_fn  # Store the evaluation function
        self.address = "ipc:///tmp/scheduler.ipc" if use_ipc else "tcp://localhost:5555"
        self.logger = setup_logging(f"Worker-{worker_id}")

    def run(self):
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(self.address)

        self.logger.info(f"Started worker {self.worker_id} using {self.address}")

        with self.active_workers.get_lock():
            self.active_workers.value += 1
            self.logger.info(
                f"Worker {self.worker_id} registered. Active workers: {self.active_workers.value}"
            )

        try:
            while True:
                request = GetSuggestionRequest(worker_id=self.worker_id)
                socket.send(msgspec.json.encode(request))

                response = msgspec.json.decode(socket.recv(), type=Message)
                match response:
                    case GetSuggestionResponse(parameters=None):
                        self.logger.info(
                            f"Worker {self.worker_id}: No more parameter suggestions available"
                        )
                        return

                    case GetSuggestionResponse(parameters=params):
                        result = self.evaluate_parameters(params)
                        request = SubmitResultRequest(worker_id=self.worker_id, result=result)
                        socket.send(msgspec.json.encode(request))

                        response = msgspec.json.decode(socket.recv(), type=Message)
                        match response:
                            case SubmitResultResponse(success=True):
                                pass
                            case SubmitResultResponse(success=False, error=error):
                                self.logger.error(f"Error submitting result: {error}")

        except Exception as e:
            self.logger.error(f"Worker {self.worker_id} error: {e}")
        finally:
            with self.active_workers.get_lock():
                self.active_workers.value -= 1
                self.logger.info(
                    f"Worker {self.worker_id} deregistered. Active workers: {self.active_workers.value}"
                )

            socket.close()
            context.term()
            self.logger.info(f"Worker {self.worker_id} shutdown complete")

    def evaluate_parameters(self, params: dict[ParameterName, Any]) -> Result:
        self.logger.info(f"Worker {self.worker_id} processing parameters {params}")

        try:
            # Call the evaluation function with unpacked parameters
            objectives = self.evaluation_fn(**params)

            # Validate that the return value is a dictionary
            if not isinstance(objectives, dict):
                raise ValueError(f"Evaluation function must return a dict, got {type(objectives)}")

            return Result(parameters=params, objectives=objectives)

        except Exception as e:
            self.logger.error(f"Error evaluating function: {e}")
            raise
