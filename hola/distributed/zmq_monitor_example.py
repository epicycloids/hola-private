"""
Example of using the distributed optimization system with the monitoring dashboard.

This example sets up a server, multiple workers, and the monitoring dashboard
to demonstrate the complete HOLA distributed optimization system.
"""

import os
import time
import multiprocessing
from multiprocessing import Process
import random
import logging
import threading
import argparse
import signal
import webbrowser
import socket
import zmq

from hola.core.coordinator import OptimizationCoordinator
from hola.core.samplers import ExploreExploitSampler, SobolSampler, ClippedGaussianMixtureSampler
from hola.distributed.scheduler import OptimizationScheduler, SchedulerConfig
from hola.distributed.server import OptimizationServer, ServerConfig
from hola.distributed.worker import LocalWorker, WorkerConfig
from hola.distributed.monitor import OptimizationMonitor, run_monitor

# Configure logging with less verbose level
logging.basicConfig(
    level=logging.INFO,  # Changed from DEBUG to INFO
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
# Set third-party loggers to a higher level to reduce noise
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("zmq").setLevel(logging.WARNING)

logger = logging.getLogger("hola.distributed.zmq_monitor_example")


def example_objective_function(params, slow_demo=False):
    """
    Example multi-objective function that computes three objectives.

    Simulates a potentially CPU-intensive computation.

    :param params: Parameters to evaluate
    :type params: dict
    :param slow_demo: If True, add extra delay for demonstration purposes
    :type slow_demo: bool
    :return: Dictionary of objective values
    :rtype: dict
    """
    x = params["x"]
    y = params["y"]

    # Simulate computation time
    if slow_demo:
        # Slower demo mode for better dashboard visualization
        sleep_time = 2.0 + random.random() * 3.0  # 2-5 seconds
    else:
        # Normal mode
        sleep_time = 0.5 + random.random() * 1.5  # 0.5-2 seconds

    # In a real application, this would be actual CPU-intensive computation
    time.sleep(sleep_time)

    # Compute objectives
    f1 = x**2 + y**2
    f2 = (x-2)**2 + (y-2)**2
    f3 = (x-4)**2 + (y-4)**2

    return {"f1": f1, "f2": f2, "f3": f3}


def worker_process(
    worker_id: str,
    zmq_ipc_endpoint: str,
    objective_function,
    heartbeat_interval: float = 5.0,
    slow_demo=False
):
    """
    Worker process that communicates with the server via ZMQ IPC.

    :param worker_id: Unique ID for this worker
    :param zmq_ipc_endpoint: ZMQ IPC endpoint for server communication
    :param objective_function: Function to evaluate objective values
    :param heartbeat_interval: Interval between heartbeats in seconds
    :param slow_demo: Whether to use slower evaluation times for demonstration
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
            objective_function=lambda params: objective_function(params, slow_demo),
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


def is_port_in_use(port):
    """Check if a port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def monitor_process(zmq_tcp_endpoint: str, open_browser: bool = True):
    """
    Run the monitoring dashboard in a separate process.

    :param zmq_tcp_endpoint: ZMQ TCP endpoint for server communication
    :param open_browser: Whether to automatically open the browser
    """
    try:
        logger.info(f"Starting monitor with endpoint {zmq_tcp_endpoint}")

        # Start Streamlit server
        import subprocess
        import sys

        # Find an available port starting from 8501
        port = 8501
        max_attempts = 5

        for attempt in range(max_attempts):
            if not is_port_in_use(port):
                break
            logger.info(f"Port {port} is already in use, trying port {port + 1}")
            port += 1

        if attempt == max_attempts - 1 and is_port_in_use(port):
            logger.error(f"Failed to find an available port after {max_attempts} attempts")
            return

        logger.info(f"Using port {port} for Streamlit dashboard")

        # Prepare command to run Streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            "--server.port", str(port),
            "--server.headless", "true",
            "--browser.serverAddress", "localhost",
            "--browser.gatherUsageStats", "false",
            # Reduce Streamlit's own logging
            "--logger.level", "warning",
        ]

        # Add the monitor module path
        monitor_path = os.path.join(os.path.dirname(__file__), "monitor.py")
        cmd.append(monitor_path)

        # Add arguments for the monitor
        cmd.extend(["--", "--zmq_endpoint", zmq_tcp_endpoint])

        # Start Streamlit process
        process = subprocess.Popen(cmd)
        logger.info("Monitor process started")

        # Open browser if requested
        if open_browser:
            time.sleep(2)  # Give Streamlit a moment to start
            dashboard_url = f"http://localhost:{port}"
            webbrowser.open(dashboard_url)
            logger.info(f"Opened monitor dashboard in browser at {dashboard_url}")

        # Wait for the process to complete
        process.wait()

    except KeyboardInterrupt:
        logger.info("Monitor interrupted by user")
    except Exception as e:
        logger.error(f"Monitor encountered an error: {str(e)}", exc_info=True)


def run_zmq_optimization_with_monitor(n_workers=4, max_evaluations=50, open_browser=True, verbose=False, slow_demo=False, wait_for_dashboard=False):
    """
    Run distributed optimization with ZMQ workers and monitoring dashboard.

    :param n_workers: Number of worker processes to create
    :param max_evaluations: Maximum number of evaluations to perform
    :param open_browser: Whether to automatically open the browser for the dashboard
    :param verbose: Whether to use verbose logging
    :param slow_demo: Whether to use slower evaluation times for demonstration
    :param wait_for_dashboard: Whether to wait for dashboard connection before starting workers
    """
    # Set log levels based on verbosity
    if verbose:
        logging.getLogger("hola").setLevel(logging.DEBUG)
        logging.getLogger("hola.distributed.server").setLevel(logging.DEBUG)
        logging.getLogger("hola.distributed.scheduler").setLevel(logging.DEBUG)
        logging.getLogger("hola.distributed.worker").setLevel(logging.DEBUG)
    else:
        logging.getLogger("hola").setLevel(logging.INFO)
        logging.getLogger("hola.distributed.server").setLevel(logging.INFO)
        logging.getLogger("hola.distributed.scheduler").setLevel(logging.INFO)
        logging.getLogger("hola.distributed.worker").setLevel(logging.INFO)

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
        },
        "f3": {
            "target": 0.0,
            "limit": 100.0,
            "direction": "minimize",
            "priority": 0.5,
            "comparison_group": 1
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

    # Set default ZMQ endpoints
    zmq_ipc_endpoint = "ipc:///tmp/hola-optimization.ipc"
    zmq_tcp_endpoint = "tcp://127.0.0.1:5555"

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

    # Create scheduler with max evaluations
    scheduler_config = SchedulerConfig(max_retries=3, retry_delay=5.0)
    scheduler = OptimizationScheduler(coordinator=coordinator, config=scheduler_config)
    scheduler.max_evaluations = max_evaluations  # Add for monitoring

    # Set verbosity level
    scheduler.set_verbose_logging(verbose)

    # Create server with both ZMQ IPC and TCP
    server_config = ServerConfig(
        zmq_ipc_endpoint=zmq_ipc_endpoint,  # For local workers
        zmq_tcp_endpoint=zmq_tcp_endpoint,  # For monitoring
        http_port=None,                     # Disable HTTP
        job_cleanup_interval=10.0,
        max_job_age=60.0,
        zmq_poll_timeout=100                # 100ms polling for faster response
    )
    server = OptimizationServer(scheduler=scheduler, config=server_config)

    # Set verbosity level
    server.set_verbose_logging(verbose)

    # Start server
    server.start()
    logger.info(f"Server started with ZMQ IPC endpoint: {zmq_ipc_endpoint}")
    logger.info(f"Server started with ZMQ TCP endpoint: {zmq_tcp_endpoint}")

    # Sleep briefly to let server initialize
    time.sleep(1.0)

    # Start monitor process
    monitor_proc = Process(
        target=monitor_process,
        args=(zmq_tcp_endpoint, open_browser)
    )
    monitor_proc.daemon = True
    monitor_proc.start()
    logger.info("Started monitor process")

    # Wait for dashboard to initialize
    wait_time = 5.0 if wait_for_dashboard else 2.0
    logger.info(f"Waiting {wait_time} seconds for monitor to initialize...")
    time.sleep(wait_time)

    # If we need to wait for dashboard connection before proceeding
    if wait_for_dashboard:
        # Create a ping socket to check if dashboard has connected to server
        ping_context = zmq.Context()
        ping_socket = ping_context.socket(zmq.REQ)
        ping_socket.setsockopt(zmq.LINGER, 0)
        ping_socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second timeout
        ping_socket.connect(zmq_tcp_endpoint)

        # Wait for dashboard to connect (up to 20 seconds)
        max_wait = 20
        dashboard_connected = False

        logger.info("Waiting for dashboard to connect to server...")
        for i in range(max_wait):
            try:
                # Check if there are any connections
                ping_socket.send_json({"action": "get_server_info"})
                response = ping_socket.recv_json()
                if response.get("status") == "ok":
                    logger.info("Dashboard connected to server")
                    dashboard_connected = True
                    break
            except zmq.error.Again:
                # Timeout, no response
                pass
            except Exception as e:
                logger.warning(f"Error checking dashboard connection: {str(e)}")

            logger.info(f"Waiting for dashboard... ({i+1}/{max_wait})")
            time.sleep(1.0)

        ping_socket.close()

        if not dashboard_connected:
            logger.warning("Dashboard did not connect in time. Proceeding anyway.")

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
                slow_demo,                # slow_demo flag
            )
        )
        worker_proc.daemon = True
        worker_proc.start()
        workers.append(worker_proc)
        logger.info(f"Started worker process {worker_id} (PID: {worker_proc.pid})")

    # Set up signal handler for graceful shutdown
    stop_event = threading.Event()

    def signal_handler(sig, frame):
        logger.info("Received signal to shut down")
        stop_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Monitor progress
    try:
        start_time = time.time()
        last_eval_count = 0

        while coordinator.get_total_evaluations() < max_evaluations:
            if stop_event.is_set():
                logger.info("Stopping optimization due to signal")
                break

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
            if time.time() - start_time > 600:  # 10 minute timeout
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

        # Terminate monitor process
        if monitor_proc.is_alive():
            logger.info("Terminating monitor process")
            monitor_proc.terminate()
            monitor_proc.join(timeout=2.0)

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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run HOLA distributed optimization with monitoring")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes")
    parser.add_argument("--evals", type=int, default=50, help="Maximum number of evaluations")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    parser.add_argument("--verbose", action="store_true", help="Use verbose logging")
    parser.add_argument("--slow-demo", action="store_true", help="Use slower evaluation times for better dashboard visualization")
    parser.add_argument("--wait", action="store_true", help="Wait for dashboard to connect before starting workers")

    args = parser.parse_args()

    # Set multiprocessing start method to 'spawn' for better cross-platform compatibility
    multiprocessing.set_start_method('spawn', force=True)

    # Run the optimization
    run_zmq_optimization_with_monitor(
        n_workers=args.workers,
        max_evaluations=args.evals,
        open_browser=not args.no_browser,
        verbose=args.verbose,
        slow_demo=args.slow_demo,
        wait_for_dashboard=args.wait
    )