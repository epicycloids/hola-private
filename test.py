import logging
import multiprocessing as mp
import sys
import threading
import time
import asyncio
from datetime import datetime
from multiprocessing.sharedctypes import Synchronized
from typing import Any, Callable, List, Dict

import msgspec
import uvicorn
import zmq
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from msgspec import Struct

from hola.core.coordinator import OptimizationCoordinator
from hola.core.objectives import ObjectiveName
from hola.core.parameters import ParameterName
from hola.core.samplers import SobolSampler

# ============================================================================
# Logging Setup
# ============================================================================


def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
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


class StatusRequest(Struct, tag="status"):
    pass


# Response Messages
class GetSuggestionResponse(Struct, tag="suggestion_response"):
    parameters: dict[ParameterName, Any] | None


class SubmitResultResponse(Struct, tag="result_response"):
    success: bool
    is_best: bool = False
    error: str | None = None


class StatusResponse(Struct, tag="status_response"):
    active_workers: int
    total_evaluations: int
    best_objectives: dict[ObjectiveName, float] | None = None


# Define the message union type
Message = (
    GetSuggestionRequest
    | SubmitResultRequest
    | ShutdownRequest
    | StatusRequest
    | GetSuggestionResponse
    | SubmitResultResponse
    | StatusResponse
)


# ============================================================================
# SchedulerProcess Implementation
# ============================================================================


class SchedulerProcess:
    """Central scheduler that coordinates optimization trials and worker assignments."""

    def __init__(self, coordinator: OptimizationCoordinator):
        self.coordinator = coordinator
        self.running: bool = False
        self.active_workers: Synchronized[int] = mp.Value(
            "i", 0
        )  # Shared counter for active workers
        self.logger = setup_logging("Scheduler")
        # Track results for faster state retrieval
        self.result_history: List[Result] = []

    def run(self):
        context: zmq.Context = zmq.Context()
        socket: zmq.Socket = context.socket(zmq.REP)
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
                            is_best = self.register_completed(message.result)
                            # Add is_best flag to the response
                            response = SubmitResultResponse(success=True, is_best=is_best)
                            socket.send(msgspec.json.encode(response))

                        case ShutdownRequest():
                            self.running = False
                            socket.send(msgspec.json.encode(SubmitResultResponse(success=True)))

                        case StatusRequest():
                            try:
                                response = StatusResponse(
                                    active_workers=self.active_workers.value,
                                    total_evaluations=self.coordinator.get_total_evaluations(),
                                    best_objectives=self.coordinator.get_best_objectives()
                                )
                                socket.send(msgspec.json.encode(response))
                            except Exception as e:
                                self.logger.error(f"Error creating status response: {e}")
                                error_response = StatusResponse(
                                    active_workers=self.active_workers.value,
                                    total_evaluations=0,
                                    best_objectives=None
                                )
                                socket.send(msgspec.json.encode(error_response))

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

    def register_completed(self, result: Result) -> bool:
        """Register a completed evaluation and return whether it's the best result."""
        # Add to history
        self.result_history.append(result)

        # Use coordinator to determine if this is the best result
        is_best = self.coordinator.evaluate_and_determine_if_best(
            result.parameters, result.objectives
        )

        self.logger.info(
            f"Trial completed: objectives={result.objectives}, "
            f"total_evaluations={self.coordinator.get_total_evaluations()}, "
            f"is_best={is_best}"
        )

        return is_best


# ============================================================================
# Worker Implementation
# ============================================================================


class LocalWorker:
    """Worker process that executes objective function evaluations."""

    def __init__(
        self,
        worker_id: int,
        active_workers: Synchronized,
        evaluation_fn: Callable[..., dict[ObjectiveName, float]],
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

        # WebSocket connections manager for clients
        self.active_connections: List[WebSocket] = []

        self.running: bool = False
        self.logger = setup_logging("Server")

        # Server thread
        self.server_thread = None

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
        async def submit_result(request: Request) -> bytes:  # Use FastAPI's Request
            try:
                # Decode the raw request body using msgspec
                body = await request.body()
                result = msgspec.json.decode(body, type=RESTSubmitResult)

                # Create result request
                result_obj = Result(parameters=result.parameters, objectives=result.objectives)
                request = SubmitResultRequest(worker_id=-1, result=result_obj)
                self.socket.send(msgspec.json.encode(request))

                # Get response from scheduler
                response = msgspec.json.decode(self.socket.recv(), type=Message)

                match response:
                    case SubmitResultResponse(success=success, is_best=is_best, error=error):
                        # If successful, get current state to broadcast to clients
                        if success:
                            # Request current status to get total_evaluations and active_workers
                            self.socket.send(msgspec.json.encode(StatusRequest()))
                            status_response = msgspec.json.decode(self.socket.recv(), type=Message)

                            if isinstance(status_response, StatusResponse):
                                # Create result data for clients
                                result_data = {
                                    "parameters": result.parameters,
                                    "objectives": result.objectives,
                                    "is_best": is_best,
                                    "active_workers": status_response.active_workers,
                                    "total_evaluations": status_response.total_evaluations
                                }

                                # Broadcast to WebSocket clients
                                if self.active_connections:
                                    asyncio.create_task(self.broadcast_result(result_data))

                        return msgspec.json.encode(RESTSubmitResponse(success=success, error=error))
                    case _:
                        return msgspec.json.encode(
                            RESTSubmitResponse(
                                success=False, error="Unexpected response from scheduler"
                            )
                        )

            except Exception as e:
                self.logger.error(f"Error in submit_result: {e}")
                return msgspec.json.encode(
                    RESTSubmitResponse(success=False, error=f"Error submitting result: {str(e)}")
                )

        @self.rest_app.get("/history")
        async def get_history():
            """Endpoint to get optimization history"""
            try:
                # Get current status from scheduler
                self.socket.send(msgspec.json.encode(StatusRequest()))
                status_response = msgspec.json.decode(self.socket.recv(), type=Message)

                if not isinstance(status_response, StatusResponse):
                    return {"error": "Failed to get status from scheduler"}

                # Return just the current status information
                history_data = {
                    "total_evaluations": status_response.total_evaluations,
                    "active_workers": status_response.active_workers,
                    "best_objectives": status_response.best_objectives
                }

                return {
                    "history": history_data
                }
            except Exception as e:
                self.logger.error(f"Error handling history request: {e}")
                return {"error": str(e)}

        @self.rest_app.get("/status")
        async def get_status():
            try:
                self.logger.info("Received status request")

                request = StatusRequest()
                self.socket.send(msgspec.json.encode(request))

                response = msgspec.json.decode(self.socket.recv(), type=Message)

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

        @self.rest_app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.active_connections.append(websocket)
            self.logger.info(f"WebSocket client connected. Total connections: {len(self.active_connections)}")

            try:
                # Send initial status to the new client
                self.socket.send(msgspec.json.encode(StatusRequest()))
                status_response = msgspec.json.decode(self.socket.recv(), type=Message)

                if isinstance(status_response, StatusResponse):
                    initial_data = {
                        "type": "status",
                        "data": {
                            "active_workers": status_response.active_workers,
                            "total_evaluations": status_response.total_evaluations,
                            "best_objectives": status_response.best_objectives
                        }
                    }
                    await websocket.send_json(initial_data)

                # Keep connection open
                while True:
                    # Just keep the connection alive
                    await websocket.receive_text()

            except WebSocketDisconnect:
                self.active_connections.remove(websocket)
                self.logger.info(f"WebSocket client disconnected. Remaining: {len(self.active_connections)}")
            except Exception as e:
                self.logger.error(f"WebSocket error: {e}")
                if websocket in self.active_connections:
                    self.active_connections.remove(websocket)

    async def broadcast_result(self, result_data: Dict):
        """Broadcast optimization result to all connected WebSocket clients"""
        if not self.active_connections:
            return

        # Format the message
        message = {
            "type": "result",
            "data": result_data
        }

        # Send to all active connections
        for connection in self.active_connections.copy():
            try:
                await connection.send_json(message)
            except Exception as e:
                self.logger.error(f"Error broadcasting to client: {e}")
                if connection in self.active_connections:
                    self.active_connections.remove(connection)

    def start(self):
        self.running = True

        def run_server():
            config = uvicorn.Config(
                app=self.rest_app,
                host=self.host,
                port=self.port,
                log_level="info"
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
        self.socket.close()
        self.context.term()
        self.logger.info("Server stopped")


# ============================================================================
# System Control Functions
# ============================================================================


def spawn_local_worker(
    worker_id: int,
    active_workers: Synchronized,
    evaluation_fn: Callable[..., dict[ObjectiveName, str]],
    use_ipc: bool = True,
):
    """Spawn a new worker process."""
    worker = LocalWorker(worker_id, active_workers, evaluation_fn, use_ipc)
    worker.run()


def shutdown_system(scheduler_process: mp.Process, server: Server, active_workers: Synchronized):
    """Gracefully shutdown all system components."""
    logger = setup_logging("Shutdown")
    logger.info("Initiating shutdown...")

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect("ipc:///tmp/scheduler.ipc")

    try:
        # Use the proper ShutdownRequest message type
        shutdown_request = ShutdownRequest()
        socket.send(msgspec.json.encode(shutdown_request))

        # Wait for response with timeout
        if socket.poll(1000, zmq.POLLIN):
            response = msgspec.json.decode(socket.recv(), type=Message)
            match response:
                case SubmitResultResponse(success=True):
                    logger.info("Scheduler acknowledged shutdown request")
                case _:
                    logger.warning("Unexpected response to shutdown request")
        else:
            logger.warning("No response received from scheduler during shutdown")

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

        # Ensure we're really giving the server time to stop
        time.sleep(1)

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
    objectives_dict = {"objective1": {"direction": "minimize", "target": 1e-6, "limit": 0.9}}
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
    processes: list[mp.Process] = []
    num_workers = 16
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
                    f"total_evaluations: {coordinator.get_total_evaluations()}"
                )
                if current_workers <= 0:
                    main_logger.info("All workers finished")
                    break
            time.sleep(1)

        # Clean up and report results
        time.sleep(0.5)
        main_logger.info(f"Optimization completed:")
        main_logger.info(f"Total evaluations: {coordinator.get_total_evaluations()}")
        best_objectives = coordinator.get_best_objectives()
        main_logger.info(f"Best result: {best_objectives}")

        main_logger.info("Initiating final shutdown sequence")
        shutdown_system(scheduler_process, server, active_workers)

    except KeyboardInterrupt:
        main_logger.info("\nReceived interrupt signal")
        shutdown_system(scheduler_process, server, active_workers)

    # Wait for all worker processes to finish
    for p in processes:
        p.join()

    main_logger.info("Main process exiting")
