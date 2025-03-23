"""
Example of using the distributed optimization system in HOLA with multiprocessing.

This example sets up a server, scheduler, and multiple remote workers in separate
processes to solve a simple multi-objective optimization problem, allowing for
true parallel execution across CPU cores.
"""

import time
import multiprocessing
from multiprocessing import Process, Event
import threading
import random
import logging
import requests

from hola.core.coordinator import OptimizationCoordinator
from hola.core.samplers import ExploreExploitSampler, SobolSampler, ClippedGaussianMixtureSampler
from hola.distributed.scheduler import OptimizationScheduler, SchedulerConfig
from hola.distributed.server import OptimizationServer, ServerConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("hola.distributed.multiproc")


def example_objective_function(params):
    """
    Example multi-objective function that computes two objectives.

    Simulates a potentially CPU-intensive computation.

    :param params: Parameters to evaluate
    :type params: dict
    :return: Dictionary of objective values
    :rtype: dict
    """
    x = params["x"]
    y = params["y"]

    # Simulate computation time (random between 0.5 and 2 seconds)
    # In a real application, this would be actual CPU-intensive computation
    time.sleep(0.5 + random.random() * 1.5)

    # Compute objectives
    f1 = x**2 + y**2
    f2 = (x-2)**2 + (y-2)**2

    return {"f1": f1, "f2": f2}


def worker_process(
    worker_id: str,
    server_url: str,
    objective_function,
    heartbeat_interval: float = 5.0,
    stop_event = None
):
    """
    Worker process that communicates with the server via HTTP.

    :param worker_id: Unique ID for this worker
    :param server_url: URL of the optimization server
    :param objective_function: Function to evaluate objective values
    :param heartbeat_interval: Interval between heartbeats in seconds
    :param stop_event: Event to signal process termination
    """
    if stop_event is None:
        stop_event = Event()

    # Create a local stop event for threads in this process
    local_stop_event = threading.Event()

    logger.info(f"Worker {worker_id} starting")

    # Register with server
    try:
        register_data = {"worker_id": worker_id}
        response = requests.post(f"{server_url}/api/register", json=register_data)
        response.raise_for_status()
        logger.info(f"Worker {worker_id} registered with server")
    except Exception as e:
        logger.error(f"Failed to register worker {worker_id}: {str(e)}")
        return

    # Start heartbeat thread (not process)
    heartbeat_thread = threading.Thread(
        target=heartbeat_loop,
        args=(worker_id, server_url, heartbeat_interval, local_stop_event)
    )
    heartbeat_thread.daemon = True
    heartbeat_thread.start()
    logger.info(f"Started heartbeat thread for worker {worker_id}")

    # Main worker loop
    consecutive_errors = 0
    while not stop_event.is_set():
        try:
            # Get job from server
            response = requests.get(f"{server_url}/api/job", params={"worker_id": worker_id})
            response.raise_for_status()

            job_data = response.json()
            job_id = job_data["job_id"]
            parameters = job_data["parameters"]

            logger.info(f"Worker {worker_id} received job {job_id}")

            # Evaluate objective function
            try:
                objectives = objective_function(parameters)
                success = True
                logger.info(f"Worker {worker_id} evaluated job {job_id}: {objectives}")
            except Exception as e:
                logger.error(f"Error evaluating job {job_id}: {str(e)}")
                objectives = {}
                success = False

            # Submit results
            result_data = {
                "worker_id": worker_id,
                "job_id": job_id,
                "objectives": objectives,
                "success": success
            }
            response = requests.post(f"{server_url}/api/result", json=result_data)
            response.raise_for_status()
            logger.info(f"Worker {worker_id} submitted results for job {job_id}")

            # Reset error counter on success
            consecutive_errors = 0

        except Exception as e:
            logger.error(f"Error in worker {worker_id} loop: {str(e)}")
            consecutive_errors += 1

            # Add backoff with increasing delay on consecutive errors
            backoff_time = min(5.0 * (2 ** (consecutive_errors - 1)), 60.0)
            logger.info(f"Worker {worker_id} backing off for {backoff_time:.1f} seconds")

            # Wait for backoff time or until stopped
            stop_event.wait(backoff_time)

    # Signal local threads to stop
    local_stop_event.set()

    # Wait for heartbeat thread to finish
    heartbeat_thread.join(timeout=1.0)

    logger.info(f"Worker {worker_id} stopping")


def heartbeat_loop(
    worker_id: str,
    server_url: str,
    heartbeat_interval: float,
    stop_event: threading.Event
):
    """
    Send regular heartbeats to the server.

    :param worker_id: Worker ID
    :param server_url: Server URL
    :param heartbeat_interval: Interval between heartbeats in seconds
    :param stop_event: Event to signal thread termination
    """
    logger.info(f"Heartbeat loop started for worker {worker_id}")

    while not stop_event.is_set():
        try:
            heartbeat_data = {"worker_id": worker_id}
            response = requests.post(f"{server_url}/api/heartbeat", json=heartbeat_data)
            response.raise_for_status()
            logger.debug(f"Worker {worker_id} sent heartbeat")
        except Exception as e:
            logger.error(f"Error sending heartbeat for worker {worker_id}: {str(e)}")

        # Wait for next heartbeat or until stopped
        stop_event.wait(heartbeat_interval)

    logger.info(f"Heartbeat loop stopped for worker {worker_id}")


def run_multiproc_optimization(n_workers=2, max_evaluations=10):
    """
    Run distributed optimization with workers in separate processes.

    :param n_workers: Number of worker processes to create
    :param max_evaluations: Maximum number of evaluations to perform
    """
    # Define parameters
    parameters = {
        "x": {"type": "continuous", "min": 0.0, "max": 10.0},
        "y": {"type": "continuous", "min": 0.0, "max": 10.0}
    }

    # Define objectives
    objectives = {
        "f1": {
            "target": 0.0,
            "limit": 100.0,
            "direction": "minimize",
            "priority": 1.0,
            "comparison_group": 0
        },
        "f2": {
            "target": 0.0,
            "limit": 100.0,
            "direction": "minimize",
            "priority": 0.8,
            "comparison_group": 0
        }
    }

    # Create samplers for exploration and exploitation
    explore_sampler = SobolSampler(dimension=2)
    exploit_sampler = ClippedGaussianMixtureSampler(dimension=2, n_components=2)

    # Create an explore-exploit sampler
    sampler = ExploreExploitSampler(
        explore_sampler=explore_sampler,
        exploit_sampler=exploit_sampler
    )

    # Create coordinator
    coordinator = OptimizationCoordinator.from_dict(
        hypercube_sampler=sampler,
        objectives_dict=objectives,
        parameters_dict=parameters,
        minimum_fit_samples=5
    )

    # Create scheduler
    scheduler_config = SchedulerConfig(max_retries=3, retry_delay=5.0)
    scheduler = OptimizationScheduler(coordinator=coordinator, config=scheduler_config)

    # Create server with HTTP only
    server_config = ServerConfig(
        zmq_ipc_endpoint=None,  # Disable ZMQ
        zmq_tcp_endpoint=None,  # Disable ZMQ
        http_host="0.0.0.0",
        http_port=8080,
        job_cleanup_interval=10.0,
        max_job_age=60.0
    )
    server = OptimizationServer(scheduler=scheduler, config=server_config)

    # Start server
    server.start()
    print(f"Server started with HTTP API at http://localhost:8080")

    # Sleep briefly to let server initialize
    time.sleep(1.0)

    # Create shared stop event for all processes
    stop_event = multiprocessing.Event()

    # Create and start worker processes
    workers = []
    for i in range(n_workers):
        worker_id = f"worker-{i+1}"
        worker_proc = Process(
            target=worker_process,
            args=(
                worker_id,                   # worker_id
                "http://localhost:8080",     # server_url
                example_objective_function,  # objective_function
                5.0,                         # heartbeat_interval
                stop_event                   # stop_event
            )
        )
        worker_proc.daemon = True
        worker_proc.start()
        workers.append(worker_proc)
        print(f"Started worker process {worker_id} (PID: {worker_proc.pid})")

    # Monitor progress
    try:
        start_time = time.time()
        last_eval_count = 0

        while coordinator.get_total_evaluations() < max_evaluations:
            time.sleep(1.0)
            curr_eval_count = coordinator.get_total_evaluations()

            if curr_eval_count > last_eval_count:
                print(f"Progress: {curr_eval_count}/{max_evaluations} evaluations completed")
                last_eval_count = curr_eval_count

                # Get current best
                best_trial = coordinator.get_best_trial()
                if best_trial:
                    print(f"Current best: parameters={best_trial.parameters}, objectives={best_trial.objectives}")

            # Check for timeout
            if time.time() - start_time > 60:  # 60 second timeout
                print("Optimization timed out")
                break

    except KeyboardInterrupt:
        print("Optimization interrupted by user")

    finally:
        # Signal workers to stop
        print("Stopping worker processes...")
        stop_event.set()

        # Give workers time to clean up
        time.sleep(2.0)

        # Terminate any remaining workers
        for i, worker in enumerate(workers):
            if worker.is_alive():
                print(f"Terminating worker-{i+1}")
                worker.terminate()

        # Wait for all workers to finish
        for worker in workers:
            worker.join(timeout=2.0)

        # Stop server
        server.stop()

        # Final results
        best_trial = coordinator.get_best_trial()
        print("\nOptimization completed:")
        print(f"Total evaluations: {coordinator.get_total_evaluations()}")

        if best_trial:
            print(f"Best parameters: {best_trial.parameters}")
            print(f"Best objectives: {best_trial.objectives}")
        else:
            print("No feasible solutions found")


if __name__ == "__main__":
    # Set multiprocessing start method to 'spawn' for better cross-platform compatibility
    multiprocessing.set_start_method('spawn', force=True)
    run_multiproc_optimization(n_workers=4, max_evaluations=20)