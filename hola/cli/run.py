"""System control functions for running distributed optimization."""

import logging
import multiprocessing as mp
import sys
from datetime import datetime
from multiprocessing.sharedctypes import Synchronized
from typing import Callable, Optional

import msgspec
import zmq

from hola.core.coordinator import OptimizationCoordinator, OptimizationState
from hola.core.objectives import ObjectiveName
from hola.core.system import HOLA, SystemConfig
from hola.messages.protocol import Message, ShutdownRequest, SubmitResultResponse
from hola.server import Server
from hola.utils.logging import setup_logging


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
    file_handler = logging.FileHandler(f'{name.lower()}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def shutdown_system(
    scheduler_process: mp.Process,
    server: Server,
    active_workers: Synchronized,
    logger: logging.Logger = None
):
    """Gracefully shutdown all system components.

    Args:
        scheduler_process: Process running the scheduler
        server: REST API server instance
        active_workers: Shared counter for active workers
        logger: Logger to use (or create a new one if None)
    """
    if logger is None:
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

        logger.info("Shutdown complete")


def run_optimization_system(
    coordinator: OptimizationCoordinator,
    evaluation_fn: Callable[..., dict[ObjectiveName, float]],
    num_workers: int = 4,
    use_ipc_ratio: float = 0.5,
    server_host: str = "localhost",
    server_port: int = 8000,
    timeout: Optional[float] = None,
) -> OptimizationState:
    """Run a distributed optimization system and return the final state.

    This is a convenience wrapper around the HOLA class that simplifies running
    an optimization in a single function call.

    Args:
        coordinator: Configured optimization coordinator
        evaluation_fn: Function that evaluates parameters and returns objectives
        num_workers: Number of worker processes to start
        use_ipc_ratio: Fraction of workers that should use IPC (vs TCP)
        server_host: Host for the REST API server
        server_port: Port for the REST API server
        timeout: Maximum time to run optimization (None for no timeout)

    Returns:
        Final optimization state
    """
    logger = setup_logging("Optimizer")
    logger.info("Starting optimization system")

    # Create system configuration
    config = SystemConfig(
        local_workers=num_workers,
        use_ipc_ratio=use_ipc_ratio,
        server_host=server_host,
        server_port=server_port,
        timeout=timeout
    )

    # Use the HOLA class with context manager
    with HOLA(coordinator, evaluation_fn, config) as system:
        logger.info(f"Optimization system started with {num_workers} workers")

        # Wait for optimization to complete
        system.wait_until_complete()

        # Get final state
        final_state = system.get_final_state()

        logger.info(f"Optimization completed with {final_state.total_evaluations} evaluations")
        if final_state.best_result:
            logger.info(f"Best parameters: {final_state.best_result.parameters}")
            logger.info(f"Best objectives: {final_state.best_result.objectives}")

        return final_state


if __name__ == "__main__":
    # Example usage
    import numpy as np

    from hola.core.objectives import Direction, ObjectiveConfig
    from hola.core.parameters import ContinuousParameterConfig
    from hola.core.samplers import SobolSampler

    # Create and configure coordinator
    hypercube_sampler = SobolSampler(dimension=2)

    # Example objective configuration
    objectives_dict = {
        "objective1": {
            "direction": "minimize",
            "target": 0.0,
            "limit": 1.0,
            "priority": 1.0,
            "comparison_group": 0
        }
    }

    # Example parameter configuration
    parameters_dict = {
        "x": {"tag": "continuous", "min": 0.0, "max": 1.0},
        "y": {"tag": "continuous", "min": 0.0, "max": 1.0},
    }

    coordinator = OptimizationCoordinator.from_dict(
        hypercube_sampler=hypercube_sampler,
        objectives_dict=objectives_dict,
        parameters_dict=parameters_dict,
    )

    # Example evaluation function
    def example_evaluation_fn(x: float, y: float) -> dict[str, float]:
        objective1 = x**2 + y**2
        return {"objective1": objective1}

    # Run the optimization system
    final_state = run_optimization_system(
        coordinator=coordinator,
        evaluation_fn=example_evaluation_fn,
        num_workers=4,
        timeout=30,  # 30 second timeout
    )

    print(f"Final state: {final_state}")