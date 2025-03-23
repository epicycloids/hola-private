"""
Example of using the distributed optimization system in HOLA with ZMQ + multiprocessing.

This example sets up a server and multiple local workers in separate processes to solve
a simple multi-objective optimization problem using ZMQ for communication.
"""

import os
import time
import multiprocessing
from multiprocessing import Process
import random
import logging

from hola.core.coordinator import OptimizationCoordinator
from hola.core.samplers import ExploreExploitSampler, SobolSampler, ClippedGaussianMixtureSampler
from hola.distributed.scheduler import OptimizationScheduler, SchedulerConfig
from hola.distributed.server import OptimizationServer, ServerConfig
from hola.distributed.worker import LocalWorker, WorkerConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("hola.distributed.zmq_example")


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
    zmq_ipc_endpoint: str,
    objective_function,
    heartbeat_interval: float = 5.0
):
    """
    Worker process that communicates with the server via ZMQ IPC.

    :param worker_id: Unique ID for this worker
    :param zmq_ipc_endpoint: ZMQ IPC endpoint for server communication
    :param objective_function: Function to evaluate objective values
    :param heartbeat_interval: Interval between heartbeats in seconds
    """
    try:
        # Configure worker
        worker_config = WorkerConfig(
            worker_id=worker_id,
            heartbeat_interval=heartbeat_interval,
            zmq_timeout=10000  # Increase timeout to 10 seconds for reliability
        )

        # Create and start the worker
        worker = LocalWorker(
            objective_function=objective_function,
            zmq_ipc_endpoint=zmq_ipc_endpoint,
            config=worker_config
        )

        logger.info(f"Starting worker {worker_id}...")
        worker.start()
        logger.info(f"Worker {worker_id} started successfully")

        # Wait indefinitely - the worker runs in its own threads
        while True:
            time.sleep(1.0)

    except KeyboardInterrupt:
        logger.info(f"Worker {worker_id} interrupted by user")
    except Exception as e:
        logger.error(f"Worker {worker_id} encountered an error: {str(e)}", exc_info=True)
    finally:
        try:
            # Try to stop the worker gracefully
            worker.stop()
            logger.info(f"Worker {worker_id} stopped")
        except:
            pass


def run_zmq_optimization(n_workers=2, max_evaluations=20):
    """
    Run distributed optimization with ZMQ workers in separate processes.

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

    # Set default ZMQ IPC endpoint
    zmq_ipc_endpoint = "ipc:///tmp/hola-optimization.ipc"

    # Try to clean up any existing socket file
    ipc_path = zmq_ipc_endpoint.replace("ipc://", "")
    if os.path.exists(ipc_path):
        try:
            os.remove(ipc_path)
            logger.info(f"Removed existing IPC socket file: {ipc_path}")
        except Exception as e:
            logger.warning(f"Could not remove existing IPC socket file: {str(e)}")

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

    # Create server with ZMQ only (no HTTP), and faster polling
    server_config = ServerConfig(
        zmq_ipc_endpoint=zmq_ipc_endpoint,  # Enable ZMQ IPC
        zmq_tcp_endpoint=None,              # Disable ZMQ TCP
        http_port=None,                     # Disable HTTP
        job_cleanup_interval=10.0,
        max_job_age=60.0,
        zmq_poll_timeout=100                # 100ms polling for faster response
    )
    server = OptimizationServer(scheduler=scheduler, config=server_config)

    # Start server
    server.start()
    logger.info(f"Server started with ZMQ IPC endpoint: {zmq_ipc_endpoint}")

    # Sleep briefly to let server initialize
    time.sleep(1.0)

    # Create and start worker processes
    workers = []
    for i in range(n_workers):
        worker_id = f"worker-{i+1}"
        worker_proc = Process(
            target=worker_process,
            args=(
                worker_id,                # worker_id
                zmq_ipc_endpoint,         # zmq_ipc_endpoint
                example_objective_function,  # objective_function
                5.0,                      # heartbeat_interval
            )
        )
        worker_proc.daemon = True
        worker_proc.start()
        workers.append(worker_proc)
        logger.info(f"Started worker process {worker_id} (PID: {worker_proc.pid})")

    # Monitor progress
    try:
        start_time = time.time()
        last_eval_count = 0

        while coordinator.get_total_evaluations() < max_evaluations:
            time.sleep(1.0)
            curr_eval_count = coordinator.get_total_evaluations()

            if curr_eval_count > last_eval_count:
                logger.info(f"Progress: {curr_eval_count}/{max_evaluations} evaluations completed")
                last_eval_count = curr_eval_count

                # Get current best
                best_trial = coordinator.get_best_trial()
                if best_trial:
                    logger.info(f"Current best: parameters={best_trial.parameters}, objectives={best_trial.objectives}")

            # Check for timeout
            if time.time() - start_time > 120:  # 2 minute timeout
                logger.warning("Optimization timed out")
                break

    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user")

    finally:
        # Stop server before workers to avoid connection errors
        logger.info("Stopping server...")
        server.stop()

        # Give server time to clean up
        time.sleep(1.0)

        # Terminate worker processes
        logger.info("Stopping worker processes...")
        for i, worker in enumerate(workers):
            if worker.is_alive():
                logger.info(f"Terminating worker-{i+1}")
                worker.terminate()

        # Wait for all workers to finish
        for worker in workers:
            worker.join(timeout=2.0)

        # Final results
        best_trial = coordinator.get_best_trial()
        logger.info("\nOptimization completed:")
        logger.info(f"Total evaluations: {coordinator.get_total_evaluations()}")

        if best_trial:
            logger.info(f"Best parameters: {best_trial.parameters}")
            logger.info(f"Best objectives: {best_trial.objectives}")
        else:
            logger.info("No feasible solutions found")


if __name__ == "__main__":
    # Set multiprocessing start method to 'spawn' for better cross-platform compatibility
    multiprocessing.set_start_method('spawn', force=True)
    run_zmq_optimization(n_workers=4, max_evaluations=20)