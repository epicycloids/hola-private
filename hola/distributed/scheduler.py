"""
Contains the SchedulerProcess and WorkerState classes for coordinating distributed optimization.
"""

import logging
import os
import threading
import time
from datetime import datetime
from typing import Any, Callable, List, Dict, Optional, Union

import msgspec
import zmq

from hola.core.coordinator import OptimizationCoordinator
# Use relative imports for messages and utils within the same package
from .messages import (
    Message,
    ParameterName,
    ObjectiveName,
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
)
from .utils import setup_logging


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
        # Use the setup_logging function from .utils
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
                        # Use Message union type from .messages
                        # Decode using msgpack for ZMQ
                        message = msgspec.msgpack.decode(message_bytes, type=Message)

                        if isinstance(message, HeartbeatRequest):
                            worker_id = message.worker_id

                            if worker_id in self.workers:
                                self.workers[worker_id].update_heartbeat()
                                self.logger.debug(
                                    f"Received heartbeat from worker {worker_id}"
                                )
                                # Use HeartbeatResponse from .messages
                                # Encode using msgpack for ZMQ
                                heartbeat_socket.send(
                                    msgspec.msgpack.encode(HeartbeatResponse(success=True))
                                )
                            else:
                                # Worker not recognized - might have been removed due to timeout
                                self.logger.warning(
                                    f"Heartbeat from unknown worker {worker_id}"
                                )
                                # Encode using msgpack for ZMQ
                                heartbeat_socket.send(
                                    msgspec.msgpack.encode(
                                        HeartbeatResponse(success=False)
                                    )
                                )
                        else:
                            # Not a heartbeat message
                            self.logger.warning(
                                f"Received non-heartbeat message on heartbeat socket: {type(message)}"
                            )
                             # Encode using msgpack for ZMQ (though this is an error case)
                            heartbeat_socket.send(
                                msgspec.msgpack.encode(
                                    {"error": "Expected heartbeat message"}
                                )
                            )
                    except Exception as e:
                        self.logger.error(f"Error processing heartbeat: {e}")
                        # Encode using msgpack for ZMQ (error case)
                        heartbeat_socket.send(msgspec.msgpack.encode({"error": str(e)}))

                # Handle main messages
                if main_socket in socks and socks[main_socket] == zmq.POLLIN:
                    message_bytes = main_socket.recv()
                    try:
                        # Use Message union type from .messages
                        # Decode using msgpack for ZMQ
                        message = msgspec.msgpack.decode(message_bytes, type=Message)
                    except (msgspec.DecodeError, ValueError) as e: # Catch msgpack decode errors
                        # Try to parse as generic JSON if structured parsing fails (Maybe keep this for flexibility? Or enforce msgpack?)
                        # For now, let's assume ZMQ uses msgpack consistently.
                        self.logger.error(f"Failed to decode msgpack message: {e}")
                        # Encode using msgpack for ZMQ
                        main_socket.send(
                            msgspec.msgpack.encode(
                                {"error": f"Failed to decode msgpack message: {str(e)}"}
                            )
                        )
                        continue
                        # Original JSON fallback logic commented out:
                        # try:
                        #     message_dict = msgspec.json.decode(message_bytes)
                        #     # ... (rest of JSON fallback) ...
                        # except Exception as json_e:
                        #     self.logger.error(f"Failed to parse message as msgpack or JSON: {e}; {json_e}")
                        #     main_socket.send(
                        #         msgspec.msgpack.encode(
                        #             {"error": f"Failed to parse message: {str(e)}; {str(json_e)}"}
                        #         )
                        #     )
                        #     continue

                    match message:
                        case GetSuggestionRequest():
                            # Register or update worker state
                            worker_id = message.worker_id

                            # Check if worker is truly new before logging and adding
                            if worker_id not in self.workers:
                                # Log first contact only for truly new workers
                                self.last_seen_worker_ids.add(worker_id) # Keep track if needed elsewhere
                                self.logger.info(
                                    f"First contact from worker {worker_id}"
                                )
                                self.workers[worker_id] = WorkerState(worker_id)
                                self.logger.info(
                                    f"New worker registered: {worker_id}. Total active workers: {len(self.workers)}"
                                )
                            else:
                                # Existing worker, just update heartbeat
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

                            # Use GetSuggestionResponse from .messages
                            response = GetSuggestionResponse(parameters=params)
                            main_socket.send(msgspec.msgpack.encode(response))

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
                                # Use SubmitResultResponse from .messages
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

                            main_socket.send(msgspec.msgpack.encode(response))

                        case ShutdownRequest():
                            # Save coordinator state before shutting down
                            self.save_coordinator_state()

                            self.running = False
                            # Use SubmitResultResponse from .messages (though content doesn't matter much)
                            main_socket.send(
                                msgspec.msgpack.encode(SubmitResultResponse(success=True))
                            )

                        case StatusRequest():
                            try:
                                best_trial = self.coordinator.get_best_trial()
                                best_objectives = (
                                    best_trial.objectives if best_trial else None
                                )
                                # Use StatusResponse from .messages
                                response = StatusResponse(
                                    active_workers=len(self.workers),
                                    total_evaluations=self.coordinator.get_total_evaluations(),
                                    best_objectives=best_objectives,
                                )
                                main_socket.send(msgspec.msgpack.encode(response))
                            except Exception as e:
                                self.logger.error(
                                    f"Error creating status response: {e}"
                                )
                                error_response = StatusResponse(
                                    active_workers=len(self.workers),
                                    total_evaluations=0,
                                    best_objectives=None,
                                )
                                main_socket.send(msgspec.msgpack.encode(error_response))

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
                                # Use GetTrialsResponse from .messages
                                response = GetTrialsResponse(trials=trials_list)
                                main_socket.send(msgspec.msgpack.encode(response))
                            except Exception as e:
                                self.logger.error(f"Error getting trials: {e}")
                                main_socket.send(
                                    msgspec.msgpack.encode(GetTrialsResponse(trials=[]))
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
                                # Use GetMetadataResponse from .messages
                                response = GetMetadataResponse(metadata=metadata_list)
                                main_socket.send(msgspec.msgpack.encode(response))
                            except Exception as e:
                                self.logger.error(f"Error getting metadata: {e}")
                                main_socket.send(
                                    msgspec.msgpack.encode(
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
                                # Use GetTopKResponse from .messages
                                response = GetTopKResponse(trials=trial_dicts)
                                main_socket.send(msgspec.msgpack.encode(response))
                            except Exception as e:
                                self.logger.error(f"Error getting top k trials: {e}")
                                main_socket.send(
                                    msgspec.msgpack.encode(GetTopKResponse(trials=[]))
                                )

                        case IsMultiGroupRequest():
                            try:
                                is_multi = self.coordinator.is_multi_group()
                                # Use IsMultiGroupResponse from .messages
                                response = IsMultiGroupResponse(is_multi_group=is_multi)
                                main_socket.send(msgspec.msgpack.encode(response))
                            except Exception as e:
                                self.logger.error(f"Error checking multi group: {e}")
                                main_socket.send(
                                    msgspec.msgpack.encode(
                                        IsMultiGroupResponse(is_multi_group=False)
                                    )
                                )

                        case _:
                            self.logger.error(f"Unknown message type: {type(message)}")
                            # Ensure response is msgpack encoded
                            main_socket.send(
                                msgspec.msgpack.encode(
                                    {"error": f"Unknown message type: {type(message)}"}
                                )
                            )

            except Exception as e:
                self.logger.error(f"Scheduler error: {e}")
                try:
                    # If we're handling the main socket, respond there
                    if "main_socket" in locals() and "message" in locals():
                         # Ensure response is msgpack encoded
                        main_socket.send(msgspec.msgpack.encode({"error": str(e)}))
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