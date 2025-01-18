import asyncio
import pytest
import numpy as np
import logging
import sys
import zmq
from datetime import datetime, timezone
from pathlib import Path

from hola.core.objectives import Direction, ObjectiveConfig
from hola.core.parameters import ContinuousParameterConfig, Scale
from hola.core.samplers import SobolSampler
from hola.core.coordinator import OptimizationCoordinator
from hola.server.config import ConnectionConfig, SocketType
from hola.server.server import OptimizationServer
from hola.worker.worker import OptimizationWorker
from hola.client.client import OptimizationClient

# Configure comprehensive logging
def setup_logging():
    # Clear any existing handlers
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG,  # Set to DEBUG to get all logs
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Configure specific loggers
    loggers = [
        'hola.server.server',
        'hola.server.worker',
        'hola.server.client',
        'hola.core.coordinator'
    ]

    for logger_name in loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        # Ensure logger propagates to root
        logger.propagate = True

logger = logging.getLogger(__name__)
setup_logging()

@pytest.fixture
def ipc_config():
    # Use a unique path for tests to avoid conflicts
    test_path = f"/tmp/hola-test-{datetime.now().timestamp()}"
    return ConnectionConfig(
        socket_type=SocketType.IPC,
        ipc_path=test_path
    )

@pytest.fixture
def objectives_config():
    return {
        "accuracy": ObjectiveConfig(
            target=1.0,
            limit=0.0,
            direction=Direction.MAXIMIZE,
            priority=1.0
        ),
        "latency": ObjectiveConfig(
            target=0.0,
            limit=1000.0,
            direction=Direction.MINIMIZE,
            priority=0.5
        )
    }

@pytest.fixture
def parameters_config():
    return {
        "learning_rate": ContinuousParameterConfig(
            min=1e-4,
            max=1e-1,
            scale=Scale.LOG
        ),
        "batch_size": ContinuousParameterConfig(
            min=16,
            max=256,
            scale=Scale.LOG
        )
    }

@pytest.fixture
def coordinator(objectives_config, parameters_config):
    sampler = SobolSampler(dimension=2)
    return OptimizationCoordinator.from_dict(
        sampler,
        objectives_config,
        parameters_config
    )

async def mock_evaluation(**params):
    # Simple mock evaluation function
    learning_rate = params["learning_rate"]
    batch_size = params["batch_size"]

    # Simulate some relationship between parameters and objectives
    accuracy = 1.0 - np.exp(-learning_rate * batch_size / 50)
    latency = batch_size * (1 + 1/learning_rate) / 10

    # Add some noise
    accuracy += np.random.normal(0, 0.05)
    latency *= (1 + np.random.normal(0, 0.1))

    return {
        "accuracy": float(np.clip(accuracy, 0, 1)),
        "latency": float(np.clip(latency, 0, np.inf))
    }

@pytest.mark.asyncio
async def test_server_end_to_end(ipc_config, coordinator):
    """Test full server workflow including initialization, worker registration, and optimization."""

    # Create tasks list to track all async tasks
    tasks = []

    # Start server
    server = OptimizationServer(coordinator, ipc_config)
    server_task = asyncio.create_task(server.start())
    tasks.append(server_task)
    logger.info("Server started")

    # Give server time to start up
    await asyncio.sleep(0.1)

    # Start worker
    worker = OptimizationWorker(ipc_config, mock_evaluation)
    worker_task = asyncio.create_task(worker.start())
    tasks.append(worker_task)
    logger.info("Worker started")

    # Start client
    client = OptimizationClient(ipc_config)
    await client.start()

    try:
        # Initialize optimization with explicit ObjectiveConfig and ParameterConfig objects
        await client.initialize(
            objectives_config={
                "accuracy": ObjectiveConfig(
                    target=1.0,
                    limit=0.0,
                    direction=Direction.MAXIMIZE,
                    priority=1.0
                ),
                "latency": ObjectiveConfig(
                    target=0.0,
                    limit=1000.0,
                    direction=Direction.MINIMIZE,
                    priority=0.5
                )
            },
            parameters_config={
                "learning_rate": ContinuousParameterConfig(
                    min=1e-4,
                    max=1e-1,
                    scale=Scale.LOG
                ),
                "batch_size": ContinuousParameterConfig(
                    min=16,
                    max=256,
                    scale=Scale.LOG
                )
            }
        )
        logger.info("Optimization initialized")

        # Let optimization run for a few iterations
        for _ in range(3):
            # Get status periodically
            status = await client.get_status()
            assert status is not None
            await asyncio.sleep(1)

        logger.info("Status gotten three times")

        # Pause optimization
        await client.pause()

        # Check status while paused
        status = await client.get_status()
        assert status is not None

        # Resume optimization
        await client.resume()

        # Let it run a bit more
        await asyncio.sleep(2)

        # Update objective configuration
        new_objectives = {
            "accuracy": ObjectiveConfig(
                target=0.95,  # Slightly relaxed target
                limit=0.0,
                direction=Direction.MAXIMIZE,
                priority=1.0,
                comparison_group=0  # Add the missing comparison_group
            ),
            "latency": ObjectiveConfig(
                target=0.0,
                limit=500.0,  # More stringent limit
                direction=Direction.MINIMIZE,
                priority=0.8,  # Increased priority
                comparison_group=0  # Add the missing comparison_group
            )
        }
        await client.update_objectives(new_objectives)


        # Let it run with new objectives
        await asyncio.sleep(2)

        # Get final status
        final_status = await client.get_status()
        assert final_status is not None

        best_result = await server.coordinator.get_best_result()
        logger.info(f"Best result: {best_result}")
        # assert final_status.status.total_evaluations > 0

    finally:
        # Set a timeout for cleanup
        cleanup_timeout = 5  # seconds

        # Clean up
        logger.info("Starting cleanup...")

        # Stop the worker first
        logger.info("Stopping worker...")
        await worker.stop()
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass

        # Stop the client
        logger.info("Stopping client...")
        await client.stop()

        # Stop the server last
        logger.info("Stopping server...")
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass

        # Close ZMQ context to ensure all sockets are closed
        logger.info("Closing ZMQ context...")
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=cleanup_timeout
            )
        except asyncio.TimeoutError:
            logger.warning("Cleanup timed out, forcing context destruction")
        finally:
            # Force close all ZMQ contexts
            server.context.destroy(linger=0)
            worker.context.destroy(linger=0)
            client.context.destroy()

        # Clean up IPC files
        logger.info("Cleaning up IPC files...")
        for suffix in ['-worker', '-client']:
            try:
                Path(ipc_config.ipc_path + suffix).unlink()
                logger.info(f"Removed IPC file: {ipc_config.ipc_path + suffix}")
            except FileNotFoundError:
                logger.warning(f"IPC file not found: {ipc_config.ipc_path + suffix}")

        logger.info("Cleanup completed")

if __name__ == "__main__":
    pytest.main([__file__])