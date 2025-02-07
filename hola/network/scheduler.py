import multiprocessing as mp
from typing import Any

import msgspec
import zmq

from hola.core.coordinator import OptimizationCoordinator
from hola.core.parameters import ParameterName
from hola.messages.base import Result
from hola.messages.scheduler import (
    GetSuggestionRequest,
    GetSuggestionResponse,
    Message,
    ShutdownRequest,
    SubmitResultRequest,
    SubmitResultResponse,
)
from hola.utils.logging import setup_logging


class SchedulerProcess:
    """Central scheduler that coordinates optimization trials and worker assignments."""

    def __init__(self, coordinator: OptimizationCoordinator):
        self.coordinator = coordinator
        self.running = False
        self.active_workers = mp.Value("i", 0)  # Shared counter for active workers
        self.logger = setup_logging("Scheduler")

    def run(self):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.setsockopt(zmq.LINGER, 0)  # Prevents hanging on context termination

        socket.bind("ipc:///tmp/scheduler.ipc")
        socket.bind("tcp://*:5555")

        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)

        self.running = True
        while self.running:
            try:
                # Poll with timeout
                if poller.poll(100):  # 100ms timeout
                    message = msgspec.json.decode(socket.recv(), type=Message)

                    match message:
                        case GetSuggestionRequest():
                            params = self.suggest_parameters()
                            response = GetSuggestionResponse(parameters=params)
                            socket.send(msgspec.json.encode(response))

                        case SubmitResultRequest():
                            self.register_completed(message.result)
                            response = SubmitResultResponse(success=True)
                            socket.send(msgspec.json.encode(response))

                        case ShutdownRequest():
                            self.running = False
                            socket.send(msgspec.json.encode(SubmitResultResponse(success=True)))

            except Exception as e:
                self.logger.error(f"Scheduler error: {e}")
                try:
                    socket.send_pyobj(None)
                except:
                    pass

        self.logger.info("Scheduler cleaning up...")
        socket.close()
        context.term()
        self.logger.info("Scheduler shutdown complete")

    def suggest_parameters(self) -> dict[ParameterName, Any] | None:
        suggestions = self.coordinator.suggest_parameters(n_samples=1)
        if not suggestions:
            return None
        return suggestions[0]

    def register_completed(self, result: Result):
        self.coordinator.record_evaluation(result.parameters, result.objectives)
        state = self.coordinator.get_state()
        self.logger.info(
            f"Trial completed: objectives={result.objectives}, "
            f"total_evaluations={state.total_evaluations}, "
            f"best_result={state.best_result}"
        )
