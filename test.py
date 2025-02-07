import logging
import multiprocessing as mp
import sys
import threading
import time
from datetime import datetime
from typing import Any

import msgspec
import uvicorn
import zmq
from fastapi import FastAPI
from msgspec import Struct

from hola.core.coordinator import OptimizationCoordinator
from hola.core.objectives import ObjectiveName
from hola.core.parameters import ParameterName
from hola.core.samplers import SobolSampler

# ============================================================================
# Logging Setup
# ============================================================================


def setup_logging(name, level=logging.INFO):
    """Configure logging for a component with console and file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create file handler
    file_handler = logging.FileHandler(f'scheduler_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


# ============================================================================
# Message Types
# ============================================================================


class ParameterSet(Struct):
    """Parameters suggested by the coordinator for evaluation"""

    values: dict[ParameterName, Any]


class Result(Struct):
    """Complete result of a trial evaluation"""

    parameters: dict[ParameterName, Any]
    objectives: dict[ObjectiveName, float]


# Request Messages
class GetSuggestionRequest(Struct, tag="get_suggestion"):
    worker_id: int


class SubmitResultRequest(Struct, tag="submit_result"):
    worker_id: int
    result: Result


class ShutdownRequest(Struct, tag="shutdown"):
    pass


# Response Messages
class GetSuggestionResponse(Struct, tag="suggestion_response"):
    parameters: dict[ParameterName, Any] | None


class SubmitResultResponse(Struct, tag="result_response"):
    success: bool
    error: str | None = None


# Define the message union type
Message = (
    GetSuggestionRequest
    | SubmitResultRequest
    | ShutdownRequest
    | GetSuggestionResponse
    | SubmitResultResponse
)


# ============================================================================
# Scheduler Implementation
# ============================================================================


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


# ============================================================================
# Worker Implementation
# ============================================================================


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


# ============================================================================
# REST API Messages
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


# ============================================================================
# REST API Server
# ============================================================================


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


# ============================================================================
# System Control Functions
# ============================================================================


def spawn_local_worker(
    worker_id: int, active_workers, evaluation_fn: callable, use_ipc: bool = True
):
    """Spawn a new worker process."""
    worker = LocalWorker(worker_id, active_workers, evaluation_fn, use_ipc)
    worker.run()


def shutdown_system(scheduler_process, server, active_workers):
    """Gracefully shutdown all system components."""
    logger = setup_logging("Shutdown")
    logger.info("Initiating shutdown...")

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect("ipc:///tmp/scheduler.ipc")

    try:
        socket.send_multipart([Message.SHUTDOWN])
        if socket.poll(1000, zmq.POLLIN):
            socket.recv_pyobj()
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
    finally:
        socket.close()
        context.term()

        logger.info("Waiting for scheduler process to terminate...")
        scheduler_process.terminate()
        scheduler_process.join(timeout=2)

        logger.info("Stopping server...")
        server.stop()

        logger.info("Shutdown complete")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    # Setup logging
    main_logger = setup_logging("Main")
    active_workers = mp.Value("i", 0)

    # Create and configure OptimizationCoordinator
    hypercube_sampler = SobolSampler(dimension=2)
    objectives_dict = {"objective1": {"direction": "minimize", "target": 0.2, "limit": 0.9}}
    parameters_dict = {
        "param1": {"type": "continuous", "min": 0.0, "max": 1.0},
        "param2": {"type": "continuous", "min": 0.0, "max": 1.0},
    }

    coordinator = OptimizationCoordinator.from_dict(
        hypercube_sampler=hypercube_sampler,
        objectives_dict=objectives_dict,
        parameters_dict=parameters_dict,
    )

    # Example evaluation function
    def example_evaluation_fn(param1: float, param2: float) -> dict[str, float]:
        objective1 = param1 + param2
        return {"objective1": objective1}

    # Initialize and start system components
    scheduler = SchedulerProcess(coordinator)
    scheduler.active_workers = active_workers
    scheduler_process = mp.Process(target=scheduler.run)
    scheduler_process.start()

    time.sleep(0.5)  # Give scheduler time to start

    # Initialize and start server
    server = Server(active_workers=active_workers)
    server.start()

    # Start workers
    processes = []
    num_workers = 4
    for i in range(num_workers):
        use_ipc = i < num_workers // 2
        p = mp.Process(
            target=spawn_local_worker, args=(i, active_workers, example_evaluation_fn, use_ipc)
        )
        p.start()
        processes.append(p)

    # Main loop and shutdown handling
    try:
        while True:
            with active_workers.get_lock():
                current_workers = active_workers.value
                main_logger.info(
                    f"Main loop: {current_workers} active workers, "
                    f"total evaluations: {coordinator.get_total_evaluations()}"
                )
                if current_workers <= 0:
                    main_logger.info("All workers finished")
                    break
            time.sleep(1)

        # Clean up and report results
        time.sleep(0.5)
        final_state = coordinator.get_state()
        main_logger.info(f"Optimization completed:")
        main_logger.info(f"Total evaluations: {final_state.total_evaluations}")
        main_logger.info(f"Best result: {final_state.best_result}")

        main_logger.info("Initiating final shutdown sequence")
        shutdown_system(scheduler_process, server, active_workers)

    except KeyboardInterrupt:
        main_logger.info("\nReceived interrupt signal")
        shutdown_system(scheduler_process, server, active_workers)

    # Wait for all worker processes to finish
    for p in processes:
        p.join()

    main_logger.info("Main process exiting")
