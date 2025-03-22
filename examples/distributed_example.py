"""
Example demonstrating the HOLA distributed optimization system.

This example shows how to:
1. Set up and run a scheduler process
2. Create and run worker processes to evaluate functions
3. Connect a monitor to visualize optimization progress

The example optimizes a simple multi-objective problem with two parameters.
"""

import logging
import multiprocessing as mp
import sys
import threading
import time
import os
from pathlib import Path
from multiprocessing.sharedctypes import Synchronized
import argparse

from hola.core.coordinator import OptimizationCoordinator
from hola.core.samplers import ExploreExploitSampler, SobolSampler, ClippedGaussianMixtureSampler
from hola.core.objectives import ObjectiveName
from hola.core.parameters import ParameterName
import hola.core.monitor as monitor
import zmq
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Import scheduler components from the scheduler module
from hola.core.scheduler import (
    SchedulerProcess,
    LocalWorker,
    Server,
    setup_logging,
    spawn_local_worker,
    shutdown_system
)

# Define the test problem
def get_default_parameters():
    """Define the parameter space for the test problem."""
    return {
        "x": {"type": "continuous", "min": 0.0, "max": 10.0},
        "y": {"type": "continuous", "min": 0.0, "max": 10.0},
    }

def get_default_objectives():
    """Define the objectives for the test problem."""
    return {
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
        },
    }

# Define the evaluation function
def evaluate(x: float, y: float) -> dict[str, float]:
    """Evaluate the test function."""
    # Add a small sleep to simulate computation time
    time.sleep(0.2)
    f1 = x**2 + y**2
    f2 = (x-2)**2 + (y-2)**2
    f3 = (x-4)**2 + (y-4)**2
    return {"f1": f1, "f2": f2, "f3": f3}

def run_optimization(num_workers: int = 4, total_iterations: int = 100, dashboard: bool = False):
    """Run the distributed optimization example."""
    main_logger = setup_logging("Main")
    main_logger.info(f"Starting distributed optimization with {num_workers} workers")

    # Create shared counter for active workers
    active_workers = mp.Value("i", 0)

    # Create and configure OptimizationCoordinator
    explore_sampler = SobolSampler(dimension=2)
    exploit_sampler = ClippedGaussianMixtureSampler(dimension=2, n_components=2)

    sampler = ExploreExploitSampler(
        explore_sampler=explore_sampler,
        exploit_sampler=exploit_sampler,
    )

    coordinator = OptimizationCoordinator.from_dict(
        hypercube_sampler=sampler,
        objectives_dict=get_default_objectives(),
        parameters_dict=get_default_parameters()
    )

    # Initialize and start scheduler
    scheduler = SchedulerProcess(coordinator)
    scheduler.active_workers = active_workers
    scheduler_process = mp.Process(target=scheduler.run)
    scheduler_process.start()

    main_logger.info("Scheduler started")
    time.sleep(0.5)  # Give scheduler time to start

    # Initialize and start server (REST API & WebSocket)
    server = Server(active_workers=active_workers)
    server.start()
    main_logger.info("Server started")

    # Launch dashboard in separate process if requested
    dashboard_process = None
    if dashboard:
        main_logger.info("Starting Streamlit dashboard")

        # Launch dashboard directly using the run_dashboard.py script
        cmd = f"poetry run streamlit run {os.path.join('hola', 'monitor', 'run_dashboard.py')} --server.port 8501"
        dashboard_process = mp.Process(
            target=lambda: os.system(cmd),
            daemon=True
        )
        dashboard_process.start()
        main_logger.info("Dashboard process started")

    # Start worker processes
    worker_processes = []
    for i in range(num_workers):
        use_ipc = i < num_workers // 2  # Half workers use IPC, half use TCP
        p = mp.Process(
            target=spawn_local_worker,
            args=(i, active_workers, evaluate, use_ipc)
        )
        p.start()
        worker_processes.append(p)
        main_logger.info(f"Started worker {i}")

    try:
        # Main monitoring loop
        start_time = time.time()
        iterations_target = min(total_iterations, 1000)  # Cap at 1000 iterations

        main_logger.info(f"Target: {iterations_target} iterations")

        while True:
            with active_workers.get_lock():
                current_workers = active_workers.value

            current_iterations = coordinator.get_total_evaluations()
            elapsed_time = time.time() - start_time

            main_logger.info(
                f"Progress: {current_iterations}/{iterations_target} iterations "
                f"({current_iterations/iterations_target:.1%}), "
                f"Active workers: {current_workers}, "
                f"Time: {elapsed_time:.1f}s"
            )

            # Stop conditions
            if current_iterations >= iterations_target:
                main_logger.info(f"Reached target iterations: {current_iterations} >= {iterations_target}")
                break

            if current_workers <= 0 and current_iterations > 0:
                main_logger.info("All workers finished")
                break

            if elapsed_time > 300:  # 5 minute timeout
                main_logger.info("Timeout reached (5 minutes)")
                break

            time.sleep(2)  # Check every 2 seconds

        # Gather final results
        main_logger.info(f"Optimization completed")
        main_logger.info(f"Total iterations: {coordinator.get_total_evaluations()}")

        best_trial = coordinator.get_best_trial()
        if best_trial:
            main_logger.info(f"Best parameters: {best_trial.parameters}")
            main_logger.info(f"Best objectives: {best_trial.objectives}")

        # If dashboard is running, keep it alive for viewing results
        if dashboard:
            main_logger.info("Optimization completed. Dashboard remains active.")
            main_logger.info("Press Ctrl+C when done viewing results to exit")

            # Keep the scheduler running for the dashboard
            while dashboard_process.is_alive():
                time.sleep(1)

    except KeyboardInterrupt:
        main_logger.info("\nReceived interrupt signal")
    finally:
        # Clean up
        main_logger.info("Cleaning up resources...")
        shutdown_system(scheduler_process, server, active_workers)

        for p in worker_processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=1)

        if dashboard_process and dashboard_process.is_alive():
            dashboard_process.terminate()
            dashboard_process.join(timeout=1)

        main_logger.info("Cleanup complete")

if __name__ == "__main__":
    import os

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run distributed optimization example")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes")
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations")
    parser.add_argument("--dashboard", action="store_true", help="Launch Streamlit dashboard")

    args = parser.parse_args()

    # Run the example
    run_optimization(
        num_workers=args.workers,
        total_iterations=args.iterations,
        dashboard=args.dashboard
    )