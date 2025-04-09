"""
Example script for running a distributed hyperparameter optimization locally.

This script sets up:
- An OptimizationCoordinator with a simple 2D parameter space and 3 objectives.
- A SchedulerProcess to manage the optimization.
- A Server (FastAPI) to potentially allow remote workers (though not used in this example).
- Multiple LocalWorker processes to perform the evaluations.

It runs the optimization for a fixed duration or until a target number of evaluations
is reached, then shuts down the system and attempts to reload the saved state.
"""

import logging
import multiprocessing as mp
import os
import random
import time
from typing import Callable, Dict, List

import msgspec
import numpy as np
import zmq

# Imports from the HOLA library (assuming examples/ is at the same level as hola/)
# Adjust sys.path if necessary, or run as a module `python -m examples.run_distributed`
import sys
# Add the parent directory (containing hola/) to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from hola.core.coordinator import OptimizationCoordinator
from hola.core.samplers import (
    SobolSampler,
    ClippedGaussianMixtureSampler,
    ExploreExploitSampler,
)
from hola.distributed.scheduler import SchedulerProcess
from hola.distributed.server import Server
from hola.distributed.worker import LocalWorker
from hola.distributed.utils import setup_logging
from hola.distributed.messages import (
    Message,
    ObjectiveName, # Assuming ObjectiveName=str is sufficient from messages.py
    StatusRequest,
    ShutdownRequest,
    SubmitResultResponse,
    StatusResponse
)

# Define ObjectiveName explicitly if needed (or ensure messages.py defines it properly)
# ObjectiveName = str

# ============================================================================
# Helper Functions (Moved from original test.py)
# ============================================================================

def spawn_local_worker(
    worker_id: int,
    evaluation_fn: Callable[..., Dict[ObjectiveName, float]], # Corrected type hint
    use_ipc: bool = True,
):
    """Spawn a new worker process."""
    # Note: LocalWorker now uses setup_logging internally
    worker = LocalWorker(worker_id, evaluation_fn, use_ipc)
    worker.run()

def shutdown_system(scheduler_process: mp.Process, server: Server):
    """Gracefully shutdown all system components."""
    # Use setup_logging from hola.distributed.utils
    logger = setup_logging("Shutdown")
    logger.info("Initiating shutdown...")

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    # TODO: Make socket address configurable
    socket.connect("ipc:///tmp/scheduler.ipc")

    try:
        # Use the proper ShutdownRequest message type from .messages
        shutdown_request = ShutdownRequest()
        logger.info("Sending shutdown request to scheduler (will trigger coordinator save)...")
        socket.send(msgspec.json.encode(shutdown_request))

        # Wait for response with timeout
        if socket.poll(1000, zmq.POLLIN):
            # Use Message union type from .messages
            response = msgspec.json.decode(socket.recv(), type=Message)
            match response:
                # Use SubmitResultResponse from .messages
                case SubmitResultResponse(success=True):
                    logger.info("Scheduler acknowledged shutdown request")
                case _:
                    logger.warning(f"Unexpected response to shutdown request: {response}")
        else:
            logger.warning("No response received from scheduler during shutdown")

    except Exception as e:
        logger.error(f"Error during shutdown communication: {e}")
    finally:
        socket.close()
        context.term()

    logger.info("Waiting for scheduler process to terminate...")
    # Attempt graceful termination first
    scheduler_process.terminate() # Sends SIGTERM
    scheduler_process.join(timeout=5)
    if scheduler_process.is_alive():
        logger.warning("Scheduler process did not terminate gracefully, forcing kill...")
        scheduler_process.kill() # Sends SIGKILL
        scheduler_process.join(timeout=1)

    logger.info("Stopping server...")
    server.stop() # Ensure server.stop() handles Uvicorn shutdown if possible

    # Ensure we're really giving the server time to stop
    time.sleep(1)

    logger.info("Shutdown complete")


# ============================================================================
# Main Execution Block (Moved from original test.py)
# ============================================================================

if __name__ == "__main__":
    # Setup logging for the main script
    main_logger = setup_logging("Main")
    main_logger.setLevel(logging.INFO) # Set desired level for main script

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
        time.sleep(random.uniform(0.1, 0.5)) # Shorter delay for faster testing

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
    # Use SchedulerProcess from hola.distributed.scheduler
    scheduler = SchedulerProcess(
        coordinator,
        max_retries=2,
        worker_timeout_seconds=15.0, # Reduced timeout for testing
        save_interval=5,  # Save coordinator state every 5 trials
        save_dir="optimization_results" # Relative path for results
    )
    scheduler_process = mp.Process(target=scheduler.run)
    scheduler_process.start()

    main_logger.info("Waiting for scheduler to initialize...")
    time.sleep(2)  # Give scheduler more time to bind sockets

    # Initialize and start server
    # Use Server from hola.distributed.server
    server = Server()
    server.start()

    # Start workers
    processes: List[mp.Process] = []
    num_workers = 4  # Reduced number of workers for the simpler problem
    main_logger.info(f"Starting {num_workers} local workers...")
    for i in range(num_workers):
        # Ensure workers connect using the same method (IPC or TCP)
        # IPC is generally preferred for local inter-process communication
        p = mp.Process(
            target=spawn_local_worker, args=(i, example_evaluation_fn, True) # Use IPC
        )
        p.start()
        processes.append(p)
        main_logger.info(f"Started worker {i}")

    # Main loop and shutdown handling
    status_socket = None # Initialize to None
    try:
        # Poll scheduler for status
        main_logger.info("Connecting status socket...")
        status_context = zmq.Context()
        status_socket = status_context.socket(zmq.REQ)
        status_socket.setsockopt(zmq.LINGER, 0)
        # TODO: Make configurable
        status_socket.connect("ipc:///tmp/scheduler.ipc")
        main_logger.info("Status socket connected.")

        start_time = time.time()
        max_runtime = 60  # Run for a shorter time (e.g., 60 seconds)
        target_evaluations = 100 # Lower target for quicker example run

        while time.time() - start_time < max_runtime:
            # Request status from scheduler
            try:
                status_socket.send(msgspec.json.encode(StatusRequest()))
            except zmq.ZMQError as e:
                main_logger.error(f"Failed to send status request: {e}")
                time.sleep(2) # Wait before retrying
                continue

            if status_socket.poll(2000, zmq.POLLIN): # 2 second timeout
                try:
                    status_response_bytes = status_socket.recv()
                    status_response = msgspec.json.decode(
                        status_response_bytes, type=Message
                    )
                except Exception as e:
                     main_logger.error(f"Failed to receive or decode status response: {e}")
                     time.sleep(1)
                     continue

                if isinstance(status_response, StatusResponse):
                    current_workers = status_response.active_workers
                    total_evaluations = status_response.total_evaluations
                    main_logger.info(
                        f"Status: {current_workers} active workers, "
                        f"evals: {total_evaluations}/{target_evaluations}, "
                        f"time: {time.time() - start_time:.1f}s / {max_runtime}s"
                    )

                    # End early if we've done enough evaluations
                    if total_evaluations >= target_evaluations:
                        main_logger.info(f"Reached target number of {target_evaluations} evaluations.")
                        break
                    # Optional: End if no workers are active for a while
                    # if total_evaluations > 0 and current_workers == 0:
                    #     main_logger.warning("No active workers detected, shutting down.")
                    #     break
                else:
                    main_logger.warning(f"Received unexpected status response type: {type(status_response)}")
            else:
                main_logger.warning("Status request timed out.")
                # Attempt to reconnect status socket if needed
                try:
                    status_socket.close()
                    status_socket = status_context.socket(zmq.REQ)
                    status_socket.setsockopt(zmq.LINGER, 0)
                    status_socket.connect("ipc:///tmp/scheduler.ipc")
                    main_logger.info("Reconnected status socket.")
                except Exception as e:
                    main_logger.error(f"Failed to reconnect status socket: {e}")
                    break # Exit loop if status connection fails persistently

            time.sleep(5) # Check status less frequently
        else: # Executed if the loop finished due to timeout
            main_logger.warning(f"Reached maximum runtime of {max_runtime} seconds.")

        # --- Final Status and Report ---
        main_logger.info("Optimization loop finished.")
        time.sleep(1) # Give final results time to register

        if status_socket and not status_socket.closed:
            try:
                main_logger.info("Requesting final status...")
                status_socket.send(msgspec.json.encode(StatusRequest()))
                if status_socket.poll(5000):  # 5 second timeout
                    status_response_bytes = status_socket.recv()
                    status_response = msgspec.json.decode(status_response_bytes, type=Message)

                    if isinstance(status_response, StatusResponse):
                        main_logger.info(
                            f"Final Total Evaluations: {status_response.total_evaluations}"
                        )
                        if status_response.best_objectives:
                            main_logger.info(f"Final Best Result Objectives: {status_response.best_objectives}")
                        else:
                             main_logger.info("No best result found (or multi-group). Retrieve full data if needed.")
                    else:
                        main_logger.error(
                            f"Unexpected response type when requesting final status: {type(status_response)}"
                        )
                else:
                    main_logger.error("Timeout when requesting final status.")
            except Exception as e:
                 main_logger.error(f"Error getting final status: {e}")
        else:
             main_logger.warning("Status socket closed before final status check.")

    except KeyboardInterrupt:
        main_logger.info("\nReceived interrupt signal. Initiating shutdown...")
    except Exception as e:
        main_logger.error(f"Error in main loop: {e}", exc_info=True)
    finally:
        # --- Shutdown Sequence ---
        main_logger.info("Initiating system shutdown...")
        # Pass the actual server object
        shutdown_system(scheduler_process, server)

        # Close the status socket if it exists and is open
        if status_socket and not status_socket.closed:
            status_socket.close()
        if 'status_context' in locals() and not status_context.closed:
             status_context.term()

        # --- Wait for Worker Processes ---
        main_logger.info("Waiting for worker processes to terminate...")
        for i, p in enumerate(processes):
            p.join(timeout=2)
            if p.is_alive():
                main_logger.warning(f"Worker process {i} did not terminate, forcing kill...")
                p.kill()
                p.join(timeout=1)

        # --- Loading Test (Optional) ---
        main_logger.info("Attempting to reload the saved coordinator state...")
        try:
            # Find the latest run directory dynamically
            base_save_dir = "optimization_results"
            if not os.path.isdir(base_save_dir):
                 main_logger.warning(f"Save directory '{base_save_dir}' not found. Skipping load test.")
            else:
                run_dirs = [d for d in os.listdir(base_save_dir) if os.path.isdir(os.path.join(base_save_dir, d)) and d.startswith("run_")]
                if not run_dirs:
                    main_logger.warning("No run directories found in optimization_results. Skipping load test.")
                else:
                    # Find the most recently modified run directory
                    latest_run_dir = max(run_dirs, key=lambda d: os.path.getmtime(os.path.join(base_save_dir, d)))
                    load_path = os.path.join(base_save_dir, latest_run_dir, "coordinator_state.json")

                    if os.path.exists(load_path):
                        main_logger.info(f"Found coordinator state file: {load_path}")

                        # Sampler state is saved within the file, no need to recreate it for loading
                        start_load_time = time.time()
                        loaded_coordinator = OptimizationCoordinator.load_from_file(
                            filepath=load_path,
                        )
                        load_duration = time.time() - start_load_time
                        main_logger.info(f"Successfully loaded coordinator state in {load_duration:.2f} seconds.")
                        main_logger.info(f"Loaded coordinator contains {loaded_coordinator.get_total_evaluations()} trials.")

                        # Optional: Basic check on loaded data
                        if loaded_coordinator.get_total_evaluations() > 0:
                            best_trial = loaded_coordinator.get_best_trial()
                            if best_trial:
                                main_logger.info(f"Best trial from loaded state: ID {best_trial.trial_id}, Objectives {best_trial.objectives}")
                            else:
                                main_logger.info("Loaded state has trials but no single best trial (possibly multigroup or infeasible). Retrieve full data if needed.")
                        else:
                            main_logger.info("Loaded state has 0 trials.")
                    else:
                        main_logger.error(f"Coordinator state file not found at expected path: {load_path}")

        except Exception as e:
            main_logger.error(f"Failed to load coordinator state: {e}", exc_info=True)
        # --- End Loading Test ---

    main_logger.info("Main process exiting.")