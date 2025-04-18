import logging
import multiprocessing as mp
import os
import random
import sys
import threading
import time
from datetime import datetime
from typing import Any, Callable, List, Dict, Optional, Union

import msgspec
import uvicorn
import zmq
from fastapi import FastAPI, Request
from msgspec import Struct
import numpy as np

from hola.core.coordinator import OptimizationCoordinator
from hola.core.objectives import ObjectiveName
from hola.core.parameters import ParameterName
from hola.core.samplers import SobolSampler, ClippedGaussianMixtureSampler, ExploreExploitSampler

# ============================================================================
# Logging Setup
# ============================================================================


def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """Configure logging for a component with console and file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    # Create file handler in the logs directory
    log_file_path = os.path.join(logs_dir, f'scheduler_{name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    file_handler = logging.FileHandler(log_file_path)
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
class GetSuggestionRequest(Struct, tag="get_suggestion", tag_field="tag"):
    worker_id: int


class SubmitResultRequest(Struct, tag="submit_result", tag_field="tag"):
    worker_id: int
    result: Result


class HeartbeatRequest(Struct, tag="heartbeat", tag_field="tag"):
    worker_id: int


class ShutdownRequest(Struct, tag="shutdown", tag_field="tag"):
    pass


class StatusRequest(Struct, tag="status", tag_field="tag"):
    pass


class GetTrialsRequest(Struct, tag="get_trials", tag_field="tag"):
    ranked_only: bool = True


class GetMetadataRequest(Struct, tag="get_metadata", tag_field="tag"):
    trial_ids: Optional[Union[int, List[int]]] = None


class GetTopKRequest(Struct, tag="get_top_k", tag_field="tag"):
    k: int = 1


class IsMultiGroupRequest(Struct, tag="is_multi_group", tag_field="tag"):
    pass


# Response Messages
class GetSuggestionResponse(Struct, tag="suggestion_response", tag_field="tag"):
    parameters: dict[ParameterName, Any] | None


class SubmitResultResponse(Struct, tag="result_response", tag_field="tag"):
    success: bool
    is_best: bool = False
    error: str | None = None


class HeartbeatResponse(Struct, tag="heartbeat_response", tag_field="tag"):
    success: bool


class StatusResponse(Struct, tag="status_response", tag_field="tag"):
    active_workers: int
    total_evaluations: int
    best_objectives: dict[ObjectiveName, float] | None = None


class GetTrialsResponse(Struct, tag="trials_response", tag_field="tag"):
    trials: List[Dict[str, Any]]


class GetMetadataResponse(Struct, tag="metadata_response", tag_field="tag"):
    metadata: List[Dict[str, Any]]


class GetTopKResponse(Struct, tag="top_k_response", tag_field="tag"):
    trials: List[Dict[str, Any]]


class IsMultiGroupResponse(Struct, tag="multi_group_response", tag_field="tag"):
    is_multi_group: bool


# Define the message union type
Message = (
    GetSuggestionRequest
    | SubmitResultRequest
    | HeartbeatRequest
    | ShutdownRequest
    | StatusRequest
    | GetTrialsRequest
    | GetMetadataRequest
    | GetTopKRequest
    | IsMultiGroupRequest
    | GetSuggestionResponse
    | SubmitResultResponse
    | HeartbeatResponse
    | StatusResponse
    | GetTrialsResponse
    | GetMetadataResponse
    | GetTopKResponse
    | IsMultiGroupResponse
)


# ============================================================================
# SchedulerProcess Implementation
# ============================================================================


class WorkerState:
    """Class to track the state of a worker."""

    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self.current_parameters: Optional[dict[ParameterName, Any]] = None
        self.start_time: Optional[float] = None
        self.last_heartbeat: float = time.time()
        self.retry_count: int = 0

    def assign_parameters(self, parameters: dict[ParameterName, Any]):
        """Assign parameters to this worker."""
        self.current_parameters = parameters
        self.start_time = time.time()
        self.last_heartbeat = time.time()

    def update_heartbeat(self):
        """Update the last heartbeat time."""
        self.last_heartbeat = time.time()

    def is_timed_out(self, timeout_seconds: float) -> bool:
        """Check if the worker has timed out."""
        if self.start_time is None:
            return False
        return (time.time() - self.last_heartbeat) > timeout_seconds


class SchedulerProcess:
    """Central scheduler that coordinates optimization trials and worker assignments."""

    def __init__(
        self,
        coordinator: OptimizationCoordinator,
        max_retries: int = 3,
        worker_timeout_seconds: float = 300.0,
        save_interval: int = 10,  # Save coordinator state every 10 trials by default
        save_dir: str = "optimization_results",  # Directory for saving results
    ):
        self.coordinator = coordinator
        self.running: bool = False
        self.logger = setup_logging("Scheduler")
        # Set log level for less verbosity
        self.logger.setLevel(logging.INFO)
        # Track results for faster state retrieval
        self.result_history: List[Result] = []

        # Worker management
        self.workers: Dict[int, WorkerState] = {}
        self.max_retries: int = max_retries
        self.worker_timeout_seconds: float = worker_timeout_seconds
        self.retry_queue: List[dict[ParameterName, Any]] = []

        # Start timeout checker thread
        self.timeout_checker_thread = None

        # Worker ID tracking - helps ensure uniqueness
        self.last_seen_worker_ids = set()

        # For tracking the current best objective values
        self.best_objectives = None

        # Save configuration
        self.save_interval = save_interval

        # Base directory for saving results
        self.base_save_dir = save_dir
        # Create base save directory if it doesn't exist
        os.makedirs(self.base_save_dir, exist_ok=True)

        # Create a timestamped subdirectory for this optimization run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join(self.base_save_dir, f"run_{timestamp}")
        os.makedirs(self.save_dir, exist_ok=True)
        self.logger.info(f"Created optimization run directory: {self.save_dir}")

        # Save README with initial information
        self.save_readme()

        # Track the last trial count when coordinator was saved
        self.last_save_count = 0

    def save_readme(self) -> None:
        """
        Save a README file with basic information about the optimization run.
        """
        try:
            readme_path = os.path.join(self.save_dir, "README.txt")

            # Get information about objectives and parameters
            # Using safer approach to get info in case methods don't exist
            try:
                if hasattr(self.coordinator.leaderboard._objective_scorer, 'get_info'):
                    objective_info = self.coordinator.leaderboard._objective_scorer.get_info()
                else:
                    # Fallback to a simple representation
                    try:
                        objective_names = self.coordinator.leaderboard._objective_scorer.objective_names
                    except AttributeError:
                        # If objective_names attribute doesn't exist, try to infer from another source
                        try:
                            objective_names = list(self.coordinator.leaderboard._objective_scorer.objective_directions.keys())
                        except (AttributeError, KeyError):
                            objective_names = ["unknown_objective"]

                    objective_info = {name: {"direction": "unknown"}
                                     for name in objective_names}
            except Exception as e:
                self.logger.warning(f"Could not get objective info: {e}")
                objective_info = {}

            try:
                if hasattr(self.coordinator.parameter_transformer, 'get_info'):
                    param_info = self.coordinator.parameter_transformer.get_info()
                else:
                    # Fallback to a simple representation
                    try:
                        parameter_names = self.coordinator.parameter_transformer.parameter_names
                    except AttributeError:
                        # If parameter_names attribute doesn't exist, try to infer from another source
                        try:
                            parameter_names = list(self.coordinator.parameter_transformer.parameters.keys())
                        except (AttributeError, KeyError):
                            parameter_names = ["unknown_parameter"]

                    param_info = {name: {"type": "unknown"}
                                 for name in parameter_names}
            except Exception as e:
                self.logger.warning(f"Could not get parameter info: {e}")
                param_info = {}

            with open(readme_path, 'w') as f:
                f.write("Optimization Run Information\n")
                f.write("===========================\n\n")
                f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                f.write("Objectives:\n")
                if objective_info:
                    for obj_name, obj_config in objective_info.items():
                        direction = obj_config.get('direction', 'unknown')
                        f.write(f"  - {obj_name}: {direction}\n")
                else:
                    f.write("  (Objective information not available)\n")

                f.write("\nParameters:\n")
                if param_info:
                    for param_name, param_config in param_info.items():
                        param_type = param_config.get('type', 'unknown')
                        if param_type == 'continuous':
                            min_val = param_config.get('min', 'unknown')
                            max_val = param_config.get('max', 'unknown')
                            f.write(f"  - {param_name}: {param_type} [{min_val}, {max_val}]\n")
                        else:
                            f.write(f"  - {param_name}: {param_type}\n")
                else:
                    f.write("  (Parameter information not available)\n")

                f.write("\nFiles:\n")
                f.write("  - coordinator_state.json: Complete optimizer state\n")
                f.write("  - README.txt: This file\n")

            self.logger.info(f"Saved README file to {readme_path}")
        except Exception as e:
            self.logger.error(f"Error saving README file: {e}")
            # Continue even if README creation fails

    def run(self):
        # Create context and main socket
        context = zmq.Context()
        main_socket = context.socket(zmq.REP)
        main_socket.setsockopt(zmq.LINGER, 0)  # Prevents hanging on context termination
        main_socket.bind("ipc:///tmp/scheduler.ipc")
        main_socket.bind("tcp://*:5555")

        # Create separate socket for heartbeats
        heartbeat_socket = context.socket(zmq.REP)
        heartbeat_socket.setsockopt(zmq.LINGER, 0)
        heartbeat_socket.bind("ipc:///tmp/scheduler_heartbeat.ipc")
        heartbeat_socket.bind("tcp://*:5556")

        # Create poller for both sockets
        poller = zmq.Poller()
        poller.register(main_socket, zmq.POLLIN)
        poller.register(heartbeat_socket, zmq.POLLIN)

        self.running = True

        # Start timeout checker thread
        self.timeout_checker_thread = threading.Thread(
            target=self.check_worker_timeouts
        )
        self.timeout_checker_thread.daemon = True
        self.timeout_checker_thread.start()

        while self.running:
            try:
                # Poll with timeout - check both sockets
                socks = dict(poller.poll(100))  # 100ms timeout

                # Handle heartbeat messages
                if heartbeat_socket in socks and socks[heartbeat_socket] == zmq.POLLIN:
                    message_bytes = heartbeat_socket.recv()
                    try:
                        message = msgspec.json.decode(message_bytes, type=Message)

                        if isinstance(message, HeartbeatRequest):
                            worker_id = message.worker_id

                            if worker_id in self.workers:
                                self.workers[worker_id].update_heartbeat()
                                self.logger.debug(
                                    f"Received heartbeat from worker {worker_id}"
                                )
                                heartbeat_socket.send(
                                    msgspec.json.encode(HeartbeatResponse(success=True))
                                )
                            else:
                                # Worker not recognized - might have been removed due to timeout
                                self.logger.warning(
                                    f"Heartbeat from unknown worker {worker_id}"
                                )
                                heartbeat_socket.send(
                                    msgspec.json.encode(
                                        HeartbeatResponse(success=False)
                                    )
                                )
                        else:
                            # Not a heartbeat message
                            self.logger.warning(
                                f"Received non-heartbeat message on heartbeat socket: {type(message)}"
                            )
                            heartbeat_socket.send(
                                msgspec.json.encode(
                                    {"error": "Expected heartbeat message"}
                                )
                            )
                    except Exception as e:
                        self.logger.error(f"Error processing heartbeat: {e}")
                        heartbeat_socket.send(msgspec.json.encode({"error": str(e)}))

                # Handle main messages
                if main_socket in socks and socks[main_socket] == zmq.POLLIN:
                    message_bytes = main_socket.recv()
                    try:
                        message = msgspec.json.decode(message_bytes, type=Message)
                    except ValueError:
                        # Try to parse as generic JSON if structured parsing fails
                        try:
                            message_dict = msgspec.json.decode(message_bytes)
                            tag = message_dict.get("tag", "unknown")
                            if tag == "get_suggestion":
                                message = GetSuggestionRequest(
                                    worker_id=message_dict.get("worker_id", -1)
                                )
                            elif tag == "submit_result":
                                result_dict = message_dict.get("result", {})
                                result = Result(
                                    parameters=result_dict.get("parameters", {}),
                                    objectives=result_dict.get("objectives", {}),
                                )
                                message = SubmitResultRequest(
                                    worker_id=message_dict.get("worker_id", -1),
                                    result=result,
                                )
                            elif tag == "shutdown":
                                message = ShutdownRequest()
                            elif tag == "status":
                                message = StatusRequest()
                            elif tag == "get_trials":
                                message = GetTrialsRequest(
                                    ranked_only=message_dict.get("ranked_only", True)
                                )
                            elif tag == "get_metadata":
                                message = GetMetadataRequest(
                                    trial_ids=message_dict.get("trial_ids")
                                )
                            elif tag == "get_top_k":
                                message = GetTopKRequest(k=message_dict.get("k", 1))
                            elif tag == "is_multi_group":
                                message = IsMultiGroupRequest()
                            else:
                                # Unknown message type, reply with error
                                main_socket.send(
                                    msgspec.json.encode(
                                        {"error": f"Unknown message type: {tag}"}
                                    )
                                )
                                continue
                        except Exception as e:
                            self.logger.error(f"Failed to parse message: {e}")
                            main_socket.send(
                                msgspec.json.encode(
                                    {"error": f"Failed to parse message: {str(e)}"}
                                )
                            )
                            continue

                    match message:
                        case GetSuggestionRequest():
                            # Register or update worker state
                            worker_id = message.worker_id

                            # Track and log new workers
                            if worker_id not in self.last_seen_worker_ids:
                                self.last_seen_worker_ids.add(worker_id)
                                self.logger.info(
                                    f"First contact from worker {worker_id}"
                                )

                            if worker_id not in self.workers:
                                self.workers[worker_id] = WorkerState(worker_id)
                                self.logger.info(
                                    f"New worker registered: {worker_id}. Total active workers: {len(self.workers)}"
                                )
                            else:
                                self.workers[worker_id].update_heartbeat()

                            # Check retry queue first
                            params = None
                            if self.retry_queue:
                                params = self.retry_queue.pop(0)
                                self.logger.info(
                                    f"Assigning retry parameters to worker {worker_id}: {params}"
                                )
                            else:
                                params = self.suggest_parameters()

                            if params:
                                self.workers[worker_id].assign_parameters(params)

                            response = GetSuggestionResponse(parameters=params)
                            main_socket.send(msgspec.json.encode(response))

                        case SubmitResultRequest():
                            worker_id = message.worker_id
                            if worker_id in self.workers:
                                # Clear worker's current task
                                worker_state = self.workers[worker_id]
                                worker_state.current_parameters = None
                                worker_state.update_heartbeat()
                                worker_state.retry_count = (
                                    0  # Reset retry count on successful submission
                                )

                                is_best = self.register_completed(message.result)

                                # Save coordinator state if we've reached the save interval
                                total_evaluations = self.coordinator.get_total_evaluations()
                                if total_evaluations - self.last_save_count >= self.save_interval:
                                    self.save_coordinator_state()
                                    self.last_save_count = total_evaluations

                                # Add is_best flag to the response
                                response = SubmitResultResponse(
                                    success=True, is_best=is_best
                                )
                            else:
                                # Worker not recognized
                                self.logger.warning(
                                    f"Result submitted from unknown worker {worker_id}"
                                )
                                response = SubmitResultResponse(
                                    success=False, error="Worker not recognized"
                                )

                            main_socket.send(msgspec.json.encode(response))

                        case ShutdownRequest():
                            # Save coordinator state before shutting down
                            self.save_coordinator_state()

                            self.running = False
                            main_socket.send(
                                msgspec.json.encode(SubmitResultResponse(success=True))
                            )

                        case StatusRequest():
                            try:
                                best_trial = self.coordinator.get_best_trial()
                                best_objectives = (
                                    best_trial.objectives if best_trial else None
                                )

                                response = StatusResponse(
                                    active_workers=len(self.workers),
                                    total_evaluations=self.coordinator.get_total_evaluations(),
                                    best_objectives=best_objectives,
                                )
                                main_socket.send(msgspec.json.encode(response))
                            except Exception as e:
                                self.logger.error(
                                    f"Error creating status response: {e}"
                                )
                                error_response = StatusResponse(
                                    active_workers=len(self.workers),
                                    total_evaluations=0,
                                    best_objectives=None,
                                )
                                main_socket.send(msgspec.json.encode(error_response))

                        case GetTrialsRequest():
                            try:
                                if message.ranked_only:
                                    df = self.coordinator.get_trials_dataframe(
                                        ranked_only=True
                                    )
                                else:
                                    df = self.coordinator.get_all_trials_dataframe()

                                # Convert DataFrame to list of dicts
                                trials_list = df.reset_index().to_dict(orient="records")
                                response = GetTrialsResponse(trials=trials_list)
                                main_socket.send(msgspec.json.encode(response))
                            except Exception as e:
                                self.logger.error(f"Error getting trials: {e}")
                                main_socket.send(
                                    msgspec.json.encode(GetTrialsResponse(trials=[]))
                                )

                        case GetMetadataRequest():
                            try:
                                metadata_df = self.coordinator.get_trials_metadata(
                                    trial_ids=message.trial_ids
                                )
                                # Convert to list of dicts with trial_id included
                                metadata_list = []
                                for trial_id, row in metadata_df.iterrows():
                                    metadata_dict = row.to_dict()
                                    metadata_dict["trial_id"] = trial_id
                                    metadata_list.append(metadata_dict)

                                response = GetMetadataResponse(metadata=metadata_list)
                                main_socket.send(msgspec.json.encode(response))
                            except Exception as e:
                                self.logger.error(f"Error getting metadata: {e}")
                                main_socket.send(
                                    msgspec.json.encode(
                                        GetMetadataResponse(metadata=[])
                                    )
                                )

                        case GetTopKRequest():
                            try:
                                top_trials = self.coordinator.get_top_k_trials(
                                    k=message.k
                                )

                                # Convert Trial objects to dictionaries
                                trial_dicts = []
                                for trial in top_trials:
                                    trial_dict = {
                                        "trial_id": trial.trial_id,
                                        "parameters": trial.parameters,
                                        "objectives": trial.objectives,
                                        "is_feasible": trial.is_feasible,
                                    }
                                    trial_dicts.append(trial_dict)

                                response = GetTopKResponse(trials=trial_dicts)
                                main_socket.send(msgspec.json.encode(response))
                            except Exception as e:
                                self.logger.error(f"Error getting top k trials: {e}")
                                main_socket.send(
                                    msgspec.json.encode(GetTopKResponse(trials=[]))
                                )

                        case IsMultiGroupRequest():
                            try:
                                is_multi = self.coordinator.is_multi_group()
                                response = IsMultiGroupResponse(is_multi_group=is_multi)
                                main_socket.send(msgspec.json.encode(response))
                            except Exception as e:
                                self.logger.error(f"Error checking multi group: {e}")
                                main_socket.send(
                                    msgspec.json.encode(
                                        IsMultiGroupResponse(is_multi_group=False)
                                    )
                                )

                        case _:
                            self.logger.error(f"Unknown message type: {type(message)}")
                            main_socket.send(
                                msgspec.json.encode(
                                    {"error": f"Unknown message type: {type(message)}"}
                                )
                            )

            except Exception as e:
                self.logger.error(f"Scheduler error: {e}")
                try:
                    # If we're handling the main socket, respond there
                    if "main_socket" in locals() and "message" in locals():
                        main_socket.send(msgspec.json.encode({"error": str(e)}))
                except:
                    pass

        # Save final state before closing
        self.save_coordinator_state()

        self.logger.info("Scheduler cleaning up...")
        main_socket.close()
        heartbeat_socket.close()
        context.term()
        self.logger.info("Scheduler shutdown complete")

    def check_worker_timeouts(self):
        """Background thread to check for timed out workers."""
        while self.running:
            try:
                current_time = time.time()
                timed_out_workers = []

                for worker_id, worker_state in self.workers.items():
                    if worker_state.is_timed_out(self.worker_timeout_seconds):
                        timed_out_workers.append(worker_id)
                        # Queue parameters for retry if they exist and haven't exceeded max retries
                        if worker_state.current_parameters is not None:
                            if worker_state.retry_count < self.max_retries:
                                self.logger.info(
                                    f"Worker {worker_id} timed out. Queueing parameters for retry: {worker_state.current_parameters}"
                                )
                                self.retry_queue.append(worker_state.current_parameters)
                                worker_state.retry_count += 1
                            else:
                                self.logger.warning(
                                    f"Parameters from worker {worker_id} exceeded max retries: {worker_state.current_parameters}"
                                )

                # Remove timed out workers
                for worker_id in timed_out_workers:
                    self.logger.info(f"Removing timed out worker: {worker_id}")
                    del self.workers[worker_id]

                time.sleep(10)  # Check every 10 seconds
            except Exception as e:
                self.logger.error(f"Error in timeout checker: {e}")
                time.sleep(30)  # Wait longer on error

    def suggest_parameters(self) -> dict[ParameterName, Any] | None:
        suggestions, _ = self.coordinator.suggest_parameters(n_samples=1)
        if not suggestions:
            return None
        return suggestions[0]

    def register_completed(self, result: Result) -> bool:
        """Register a completed evaluation and return whether it's the best result."""
        # Add to history
        self.result_history.append(result)

        # Record the new evaluation in the coordinator
        current_trial = self.coordinator.record_evaluation(
            result.parameters, result.objectives, metadata={"source": "worker"}
        )

        # Get the current best trial after adding the new one
        current_best = self.coordinator.get_best_trial()

        # Determine if this is the new best trial
        is_best = False
        if current_best is not None and current_trial is not None and current_best.trial_id == current_trial.trial_id:
            # The best trial ID changed, so this must be the new best
            is_best = True

        # Only log at INFO level if it's the best result, otherwise debug
        if is_best:
            self.logger.info(
                f"Best trial found: objectives={result.objectives}, "
                f"total_evaluations={self.coordinator.get_total_evaluations()}"
            )
            # Update our tracked best objectives
            self.best_objectives = current_best.objectives if current_best else result.objectives
        else:
            self.logger.debug(
                f"Trial completed: objectives={result.objectives}, "
                f"total_evaluations={self.coordinator.get_total_evaluations()}"
            )

        return is_best

    def save_coordinator_state(self) -> None:
        """
        Save the entire coordinator state to a fixed file using a temporary file approach.
        """
        try:
            trial_count = self.coordinator.get_total_evaluations()

            # Use simple filenames without timestamps since we have a timestamped directory
            filename = "coordinator_state.json"
            filepath = os.path.join(self.save_dir, filename)
            temp_filepath = f"{filepath}.temp"

            # Save coordinator state to a temporary file
            self.coordinator.save_to_file(temp_filepath)

            # Atomically rename the temporary file to the final filename
            os.replace(temp_filepath, filepath)

            self.logger.info(f"Updated coordinator state at {filepath} (trials: {trial_count})")
        except Exception as e:
            self.logger.error(f"Error saving coordinator state: {e}")
            # Clean up the temporary file if it exists
            if 'temp_filepath' in locals() and os.path.exists(temp_filepath):
                try:
                    os.remove(temp_filepath)
                except:
                    pass


# ============================================================================
# Worker Implementation
# ============================================================================


class LocalWorker:
    """Worker process that executes objective function evaluations."""

    def __init__(
        self,
        worker_id: int,
        evaluation_fn: Callable[..., dict[ObjectiveName, float]],
        use_ipc: bool = True,
        heartbeat_interval: float = 2.0,
    ):
        self.worker_id = worker_id
        self.evaluation_fn = evaluation_fn  # Store the evaluation function
        self.base_address = "ipc:///tmp/scheduler" if use_ipc else "tcp://localhost:555"
        self.main_address = (
            f"{self.base_address}.ipc" if use_ipc else f"{self.base_address}5"
        )
        self.heartbeat_address = (
            f"{self.base_address}_heartbeat.ipc" if use_ipc else f"{self.base_address}6"
        )
        self.logger = setup_logging(f"Worker-{worker_id}")
        self.running = True
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_thread = None
        # Lock for thread safety
        self.lock = threading.Lock()

    def run(self):
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(self.main_address)

        self.logger.info(f"Started worker {self.worker_id} using {self.main_address}")

        # Start heartbeat thread
        self.heartbeat_thread = threading.Thread(target=self.send_heartbeats)
        self.heartbeat_thread.daemon = True
        self.heartbeat_thread.start()

        consecutive_errors = 0
        MAX_CONSECUTIVE_ERRORS = 5

        try:
            while self.running:
                try:
                    with self.lock:  # Use lock to prevent race conditions
                        request = GetSuggestionRequest(worker_id=self.worker_id)
                        socket.send(msgspec.json.encode(request))

                        # Set a timeout for receiving the response
                        poller = zmq.Poller()
                        poller.register(socket, zmq.POLLIN)

                        if poller.poll(10000):  # 10 second timeout
                            response_bytes = socket.recv()
                            try:
                                response = msgspec.json.decode(response_bytes, type=Message)
                            except msgspec.ValidationError as ve:
                                # Try to decode as a more generic type
                                self.logger.warning(f"Worker {self.worker_id}: Validation error: {ve}, trying generic decode")
                                try:
                                    response_dict = msgspec.json.decode(response_bytes)
                                    if "parameters" in response_dict:
                                        response = GetSuggestionResponse(parameters=response_dict.get("parameters"))
                                    else:
                                        self.logger.error(f"Worker {self.worker_id}: Cannot parse response: {response_dict}")
                                        raise ValueError(f"Cannot parse response: {response_dict}")
                                except Exception as e:
                                    self.logger.error(f"Worker {self.worker_id}: Failed to decode response: {e}")
                                    raise
                        else:
                            self.logger.warning(
                                f"Worker {self.worker_id}: Response timeout, retrying..."
                            )
                            # Recreate socket on timeout
                            socket.close()
                            socket = context.socket(zmq.REQ)
                            socket.setsockopt(zmq.LINGER, 0)
                            socket.connect(self.main_address)
                            consecutive_errors += 1
                            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                                self.logger.error(
                                    f"Worker {self.worker_id}: Too many consecutive errors, shutting down"
                                )
                                self.running = False
                                break
                            continue

                    # Validate response type
                    if not isinstance(response, GetSuggestionResponse):
                        self.logger.error(
                            f"Worker {self.worker_id}: Received unexpected response type: {type(response)}"
                        )
                        consecutive_errors += 1
                        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                            self.logger.error(
                                f"Worker {self.worker_id}: Too many consecutive errors, shutting down"
                            )
                            self.running = False
                            break
                        continue

                    # Reset error counter on successful response
                    consecutive_errors = 0

                    match response:
                        case GetSuggestionResponse(parameters=None):
                            self.logger.info(
                                f"Worker {self.worker_id}: No more parameter suggestions available"
                            )
                            self.running = False
                            break

                        case GetSuggestionResponse(parameters=params):
                            try:
                                result = self.evaluate_parameters(params)

                                with self.lock:  # Use lock again
                                    request = SubmitResultRequest(
                                        worker_id=self.worker_id, result=result
                                    )
                                    socket.send(msgspec.json.encode(request))

                                    # Set a timeout for receiving the response
                                    if poller.poll(10000):  # 10 second timeout
                                        response_bytes = socket.recv()
                                        try:
                                            response = msgspec.json.decode(
                                                response_bytes, type=Message
                                            )
                                        except msgspec.ValidationError as ve:
                                            # Try to decode as a more generic type
                                            self.logger.warning(f"Worker {self.worker_id}: Result submission validation error: {ve}, trying generic decode")
                                            try:
                                                response_dict = msgspec.json.decode(response_bytes)
                                                if "success" in response_dict:
                                                    response = SubmitResultResponse(
                                                        success=response_dict.get("success", False),
                                                        is_best=response_dict.get("is_best", False),
                                                        error=response_dict.get("error")
                                                    )
                                                else:
                                                    self.logger.error(f"Worker {self.worker_id}: Cannot parse result response: {response_dict}")
                                                    raise ValueError(f"Cannot parse result response: {response_dict}")
                                            except Exception as e:
                                                self.logger.error(f"Worker {self.worker_id}: Failed to decode result response: {e}")
                                                raise
                                    else:
                                        self.logger.warning(
                                            f"Worker {self.worker_id}: Result submission timeout"
                                        )
                                        # Recreate socket on timeout
                                        socket.close()
                                        socket = context.socket(zmq.REQ)
                                        socket.setsockopt(zmq.LINGER, 0)
                                        socket.connect(self.main_address)
                                        continue

                                # Validate response type
                                if not isinstance(response, SubmitResultResponse):
                                    self.logger.error(
                                        f"Worker {self.worker_id}: Received unexpected result response type: {type(response)}"
                                    )
                                    consecutive_errors += 1
                                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                                        self.logger.error(
                                            f"Worker {self.worker_id}: Too many result submission errors, shutting down"
                                        )
                                        self.running = False
                                        break
                                    continue

                                match response:
                                    case SubmitResultResponse(
                                        success=True, is_best=is_best
                                    ):
                                        if is_best:
                                            self.logger.info(
                                                f"Worker {self.worker_id}: Found new best result!"
                                            )
                                    case SubmitResultResponse(
                                        success=False, error=error
                                    ):
                                        self.logger.error(
                                            f"Error submitting result: {error}"
                                        )
                            except Exception as e:
                                self.logger.error(
                                    f"Error during evaluation: {e}", exc_info=True
                                )
                                consecutive_errors += 1
                                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                                    self.logger.error(
                                        f"Worker {self.worker_id}: Too many evaluation errors, shutting down"
                                    )
                                    self.running = False
                                    break
                                # Wait before retrying to avoid rapid failure loops
                                time.sleep(5)
                except zmq.ZMQError as e:
                    self.logger.error(f"Worker {self.worker_id} ZMQ error: {e}")
                    # Recreate socket on ZMQ errors
                    try:
                        socket.close()
                    except:
                        pass
                    socket = context.socket(zmq.REQ)
                    socket.setsockopt(zmq.LINGER, 0)
                    socket.connect(self.main_address)
                    consecutive_errors += 1
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        self.logger.error(
                            f"Worker {self.worker_id}: Too many ZMQ errors, shutting down"
                        )
                        self.running = False
                        break
                    time.sleep(2)  # Wait before reconnecting
                except Exception as e:
                    self.logger.error(
                        f"Worker {self.worker_id} unexpected error: {e}", exc_info=True
                    )
                    consecutive_errors += 1
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        self.logger.error(
                            f"Worker {self.worker_id}: Too many general errors, shutting down"
                        )
                        self.running = False
                        break
                    time.sleep(2)  # Wait before retrying
        except Exception as e:
            self.logger.error(f"Worker {self.worker_id} error: {e}")
        finally:
            self.running = False
            socket.close()
            context.term()

            # Wait for heartbeat thread to complete
            if self.heartbeat_thread and self.heartbeat_thread.is_alive():
                self.heartbeat_thread.join(timeout=1.0)

            self.logger.info(f"Worker {self.worker_id} shutdown complete")

    def send_heartbeats(self):
        """Send regular heartbeats to the scheduler."""
        # Create a separate socket for heartbeats
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(self.heartbeat_address)

        self.logger.info(
            f"Worker {self.worker_id}: Started heartbeat thread using {self.heartbeat_address}"
        )

        consecutive_errors = 0
        MAX_HEARTBEAT_ERRORS = 10  # More forgiving for heartbeats

        while self.running:
            try:
                time.sleep(self.heartbeat_interval)
                if not self.running:
                    break

                request = HeartbeatRequest(worker_id=self.worker_id)
                socket.send(msgspec.json.encode(request))

                # Set a timeout for receiving the response
                poller = zmq.Poller()
                poller.register(socket, zmq.POLLIN)

                if poller.poll(5000):  # 5 second timeout
                    response = msgspec.json.decode(socket.recv(), type=Message)
                    if (
                        not isinstance(response, HeartbeatResponse)
                        or not response.success
                    ):
                        self.logger.warning(
                            f"Worker {self.worker_id}: Received invalid heartbeat response: {type(response)}"
                        )
                        consecutive_errors += 1
                    else:
                        # Reset error counter on successful heartbeat
                        consecutive_errors = 0
                else:
                    self.logger.warning(
                        f"Worker {self.worker_id}: Heartbeat response timeout"
                    )
                    # Recreate socket on timeout
                    socket.close()
                    socket = context.socket(zmq.REQ)
                    socket.setsockopt(zmq.LINGER, 0)
                    socket.connect(self.heartbeat_address)
                    consecutive_errors += 1

                # If we've had too many errors, log a warning but keep trying
                # (we don't stop the heartbeat thread, just log the issue)
                if consecutive_errors >= MAX_HEARTBEAT_ERRORS:
                    self.logger.error(
                        f"Worker {self.worker_id}: Heartbeat experiencing persistent errors ({consecutive_errors}), but will continue trying"
                    )
                    # Don't reset counter, but slow down heartbeats when having issues
                    time.sleep(5)

            except zmq.ZMQError as e:
                self.logger.error(f"Worker {self.worker_id}: Heartbeat ZMQ error: {e}")
                # Always recreate socket on ZMQ errors
                try:
                    socket.close()
                except:
                    pass
                socket = context.socket(zmq.REQ)
                socket.setsockopt(zmq.LINGER, 0)
                socket.connect(self.heartbeat_address)
                consecutive_errors += 1
                time.sleep(2)  # Extra sleep on error
            except Exception as e:
                self.logger.error(
                    f"Worker {self.worker_id}: Heartbeat error: {e}", exc_info=True
                )
                consecutive_errors += 1
                time.sleep(5)  # Wait longer on unexpected errors

        socket.close()
        context.term()
        self.logger.info(f"Worker {self.worker_id}: Heartbeat thread stopped")

    def evaluate_parameters(self, params: dict[ParameterName, Any]) -> Result:
        self.logger.info(f"Worker {self.worker_id} processing parameters {params}")

        try:
            # Call the evaluation function with unpacked parameters
            objectives = self.evaluation_fn(**params)

            # Validate that the return value is a dictionary
            if not isinstance(objectives, dict):
                raise ValueError(
                    f"Evaluation function must return a dict, got {type(objectives)}"
                )

            return Result(parameters=params, objectives=objectives)

        except Exception as e:
            self.logger.error(f"Error evaluating function: {e}", exc_info=True)
            # Create a fallback result with error information
            error_message = f"Evaluation failed: {str(e)}"
            objectives = {
                "error": 999.0
            }  # Using a high value for minimization objectives

            # Re-raise to allow the worker to handle this failure
            # The heartbeat mechanism will ensure the scheduler knows this worker is still alive
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


class RESTHeartbeatRequest(msgspec.Struct):
    """Request body for POST /heartbeat."""

    worker_id: int


class RESTHeartbeatResponse(msgspec.Struct):
    """Response to POST /heartbeat."""

    success: bool
    error: str | None = None


class RESTGetTrialsResponse(msgspec.Struct):
    """Response to GET /trials."""

    trials: List[Dict[str, Any]]
    error: str | None = None


class RESTGetMetadataResponse(msgspec.Struct):
    """Response to GET /metadata."""

    metadata: List[Dict[str, Any]]
    error: str | None = None


class RESTGetTopKResponse(msgspec.Struct):
    """Response to GET /top."""

    trials: List[Dict[str, Any]]
    error: str | None = None


class RESTIsMultiGroupResponse(msgspec.Struct):
    """Response to GET /is_multi_group."""

    is_multi_group: bool
    error: str | None = None


# ============================================================================
# REST API Server
# ============================================================================


class Server:
    """HTTP server providing REST API access to the optimization system."""

    def __init__(self, host: str = "localhost", port: int = 8000):
        self.host = host
        self.port = port

        self.context: zmq.Context = zmq.Context()
        self.socket: zmq.Socket = self.context.socket(zmq.REQ)
        self.socket.connect("ipc:///tmp/scheduler.ipc")

        self.rest_app = FastAPI()
        self.setup_rest_routes()

        self.running: bool = False
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
                request = GetSuggestionRequest(worker_id=worker_id)
                self.socket.send(msgspec.json.encode(request))

                response = msgspec.json.decode(self.socket.recv(), type=Message)
                match response:
                    case GetSuggestionResponse(parameters=params):
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
                result = msgspec.json.decode(body, type=RESTSubmitResult)

                # Create result request with a unique negative worker ID
                worker_id = -int(time.time() % 100000)
                result_obj = Result(
                    parameters=result.parameters, objectives=result.objectives
                )
                request = SubmitResultRequest(worker_id=worker_id, result=result_obj)
                self.socket.send(msgspec.json.encode(request))

                # Get response from scheduler
                response = msgspec.json.decode(self.socket.recv(), type=Message)

                match response:
                    case SubmitResultResponse(
                        success=success, is_best=is_best, error=error
                    ):
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
        async def shutdown_system() -> bytes:
            try:
                request = ShutdownRequest()
                self.socket.send(msgspec.json.encode(request))

                response = msgspec.json.decode(self.socket.recv(), type=Message)

                match response:
                    case SubmitResultResponse(success=success, error=error):
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
                heartbeat_req = msgspec.json.decode(body, type=RESTHeartbeatRequest)

                worker_id = heartbeat_req.worker_id
                request = HeartbeatRequest(worker_id=worker_id)
                self.socket.send(msgspec.json.encode(request))

                response = msgspec.json.decode(self.socket.recv(), type=Message)

                match response:
                    case HeartbeatResponse(success=success):
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
                request = GetTrialsRequest(ranked_only=ranked_only)
                self.socket.send(msgspec.json.encode(request))

                response = msgspec.json.decode(self.socket.recv(), type=Message)

                match response:
                    case GetTrialsResponse(trials=trials):
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

                request = GetMetadataRequest(trial_ids=parsed_trial_ids)
                self.socket.send(msgspec.json.encode(request))

                response = msgspec.json.decode(self.socket.recv(), type=Message)

                match response:
                    case GetMetadataResponse(metadata=metadata):
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
                request = GetTopKRequest(k=k)
                self.socket.send(msgspec.json.encode(request))

                response = msgspec.json.decode(self.socket.recv(), type=Message)

                match response:
                    case GetTopKResponse(trials=trials):
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
                request = IsMultiGroupRequest()
                self.socket.send(msgspec.json.encode(request))

                response = msgspec.json.decode(self.socket.recv(), type=Message)

                match response:
                    case IsMultiGroupResponse(is_multi_group=is_multi):
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
                    "best_objectives": status_response.best_objectives,
                }

                return {"history": history_data}
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
        self.socket.close()
        self.context.term()
        self.logger.info("Server stopped")


# ============================================================================
# System Control Functions
# ============================================================================


def spawn_local_worker(
    worker_id: int,
    evaluation_fn: Callable[..., dict[ObjectiveName, str]],
    use_ipc: bool = True,
):
    """Spawn a new worker process."""
    worker = LocalWorker(worker_id, evaluation_fn, use_ipc)
    worker.run()


def shutdown_system(scheduler_process: mp.Process, server: Server):
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
        logger.info("Sending shutdown request to scheduler (will trigger coordinator save)...")
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

    # Create and configure OptimizationCoordinator
    # Simple parameter space with just two variables
    parameters_dict = {
        "x": {"type": "continuous", "min": -5.0, "max": 5.0, "scale": "linear"},
        "y": {"type": "continuous", "min": -5.0, "max": 5.0, "scale": "linear"},
    }

    # Define three objectives in two comparison groups
    objectives_dict = {
        "objective1": {
            "direction": "maximize",
            "target": 1.0,
            "limit": 0.0,
            "priority": 1.0,
            "comparison_group": 0
        },
        "objective2": {
            "direction": "minimize",
            "target": 0.0,
            "limit": 10.0,
            "priority": 0.8,
            "comparison_group": 0
        },
        "objective3": {
            "direction": "minimize",
            "target": 0.0,
            "limit": 10.0,
            "priority": 0.7,
            "comparison_group": 1
        }
    }

    # Create samplers for exploration and exploitation
    dimension = len(parameters_dict)
    explore_sampler = SobolSampler(dimension=dimension)
    exploit_sampler = ClippedGaussianMixtureSampler(dimension=dimension, n_components=2)

    # Create the explore-exploit sampler
    hypercube_sampler = ExploreExploitSampler(
        explore_sampler=explore_sampler,
        exploit_sampler=exploit_sampler
    )

    coordinator = OptimizationCoordinator.from_dict(
        hypercube_sampler=hypercube_sampler,
        objectives_dict=objectives_dict,
        parameters_dict=parameters_dict,
    )

    # Simple evaluation function with 3 objectives
    def example_evaluation_fn(x: float, y: float) -> dict[str, float]:
        # Add a small random delay to simulate computation time
        time.sleep(random.uniform(0.5, 1.5))

        # Calculate objectives
        # objective1: Higher is better (maximize), peak at x=0, y=0
        objective1 = np.exp(-(x**2 + y**2)/10)

        # objective2: Lower is better (minimize), valley along y=x
        objective2 = (x - y)**2

        # objective3: Lower is better (minimize), valley at origin
        objective3 = np.sqrt(x**2 + y**2)

        # Convert numpy types to Python types to avoid encoding issues
        return {
            "objective1": float(objective1),
            "objective2": float(objective2),
            "objective3": float(objective3)
        }

    # Initialize and start system components
    scheduler = SchedulerProcess(
        coordinator,
        max_retries=2,
        worker_timeout_seconds=5.0,
        save_interval=5,  # Save coordinator state every 5 trials
        save_dir="optimization_results"
    )
    scheduler_process = mp.Process(target=scheduler.run)
    scheduler_process.start()

    time.sleep(0.5)  # Give scheduler time to start

    # Initialize and start server
    server = Server()
    server.start()

    # Start workers
    processes: list[mp.Process] = []
    num_workers = 4  # Reduced number of workers for the simpler problem
    for i in range(num_workers):
        use_ipc = i < num_workers // 2
        p = mp.Process(
            target=spawn_local_worker, args=(i, example_evaluation_fn, use_ipc)
        )
        p.start()
        processes.append(p)

    # Main loop and shutdown handling
    try:
        # Poll scheduler for status
        status_socket = zmq.Context().socket(zmq.REQ)
        status_socket.connect("ipc:///tmp/scheduler.ipc")

        start_time = time.time()
        max_runtime = 600  # Run for at most 600 seconds (10 minutes)
        target_evaluations = 2000 # Target number of evaluations

        while time.time() - start_time < max_runtime:
            # Request status from scheduler
            status_socket.send(msgspec.json.encode(StatusRequest()))

            if status_socket.poll(1000, zmq.POLLIN):
                status_response = msgspec.json.decode(
                    status_socket.recv(), type=Message
                )

                if isinstance(status_response, StatusResponse):
                    current_workers = status_response.active_workers
                    total_evaluations = status_response.total_evaluations
                    main_logger.info(
                        f"Main loop: {current_workers} active workers, "
                        f"total_evaluations: {total_evaluations}/{target_evaluations}, "
                        f"time elapsed: {time.time() - start_time:.1f}s / {max_runtime}s"
                    )

                    # End early if we've done enough evaluations
                    if total_evaluations >= target_evaluations:
                        main_logger.info(f"Reached target number of {target_evaluations} evaluations")
                        break
            else:
                main_logger.warning("Status request timeout")

            time.sleep(1)
        else: # Executed if the loop finished due to timeout
            main_logger.warning(f"Reached maximum runtime of {max_runtime} seconds.")

        # Clean up and report results
        time.sleep(0.5)
        main_logger.info(f"Optimization completed:")

        # Request final status using the existing socket
        status_socket.send(msgspec.json.encode(StatusRequest()))
        if status_socket.poll(5000):  # 5 second timeout
            status_response = msgspec.json.decode(status_socket.recv(), type=Message)

            if isinstance(status_response, StatusResponse):
                main_logger.info(
                    f"Total evaluations: {status_response.total_evaluations}"
                )
                main_logger.info(f"Best result: {status_response.best_objectives}")
            else:
                main_logger.error(
                    f"Unexpected response type when requesting final status"
                )
        else:
            main_logger.error("Timeout when requesting final status")

        status_socket.close()

        main_logger.info("Initiating final shutdown sequence")
        shutdown_system(scheduler_process, server)

        # --- Add loading test ---
        main_logger.info("Attempting to reload the saved coordinator state...")
        try:
            # Find the latest run directory in optimization_results
            base_save_dir = "optimization_results"
            run_dirs = [d for d in os.listdir(base_save_dir) if os.path.isdir(os.path.join(base_save_dir, d)) and d.startswith("run_")]
            if not run_dirs:
                main_logger.error("No run directories found in optimization_results.")
            else:
                latest_run_dir = max(run_dirs, key=lambda d: os.path.getmtime(os.path.join(base_save_dir, d)))
                load_path = os.path.join(base_save_dir, latest_run_dir, "coordinator_state.json")

                if os.path.exists(load_path):
                    main_logger.info(f"Found coordinator state file: {load_path}")

                    # Recreate the sampler needed for loading (though state is in file)
                    # The specific state of the sampler doesn't matter for just loading the leaderboard data
                    dimension = len(parameters_dict)
                    explore_sampler_load = SobolSampler(dimension=dimension)
                    exploit_sampler_load = ClippedGaussianMixtureSampler(dimension=dimension, n_components=2)
                    hypercube_sampler_load = ExploreExploitSampler(
                        explore_sampler=explore_sampler_load,
                        exploit_sampler=exploit_sampler_load
                    )

                    start_load_time = time.time()
                    loaded_coordinator = OptimizationCoordinator.load_from_file(
                        filepath=load_path,
                    )
                    load_duration = time.time() - start_load_time
                    main_logger.info(f"Successfully loaded coordinator state in {load_duration:.2f} seconds.")
                    main_logger.info(f"Loaded leaderboard contains {loaded_coordinator.get_total_evaluations()} trials.")

                    # Optional: Perform a basic check on the loaded data
                    if loaded_coordinator.get_total_evaluations() > 0:
                        best_trial = loaded_coordinator.get_best_trial()
                        if best_trial:
                             main_logger.info(f"Best trial from loaded state: ID {best_trial.trial_id}, Objectives {best_trial.objectives}")
                        else:
                             main_logger.info("Loaded state has trials but no single best trial (possibly multigroup).")
                    else:
                        main_logger.info("Loaded state has 0 trials.")

                else:
                    main_logger.error(f"Coordinator state file not found at expected path: {load_path}")

        except Exception as e:
            main_logger.error(f"Failed to load coordinator state: {e}", exc_info=True)
        # --- End loading test ---

    except KeyboardInterrupt:
        main_logger.info("\nReceived interrupt signal")
        shutdown_system(scheduler_process, server)
    except Exception as e:
        main_logger.error(f"Error in main loop: {e}")
        shutdown_system(scheduler_process, server)

    # Wait for all worker processes to finish
    for p in processes:
        p.join(timeout=1)
        if p.is_alive():
            p.terminate()

    main_logger.info("Main process exiting")
