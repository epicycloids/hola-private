"""
High-level system interface for hyperparameter optimization.

This module provides a user-friendly interface for setting up and running
optimization experiments with support for local and distributed workers
using various transport protocols (IPC, TCP, HTTP).
"""

import logging
import multiprocessing as mp
import signal
import time
from typing import Callable

from hola.core.coordinator import OptimizationCoordinator, OptimizationState
from hola.network.scheduler import SchedulerProcess
from hola.network.server import Server
from hola.network.worker import LocalWorker
from hola.utils.config import SystemConfig
from hola.utils.logging import setup_logging


class HOLA:
    """
    High-level interface for the hyperparameter optimization system.

    This class provides a user-friendly way to set up and run optimization
    experiments with support for:
    - Local workers using IPC (fastest for same-machine communication)
    - Local workers using TCP (useful for testing distributed setups)
    - Remote workers using TCP (for distributed computing)
    - Remote workers using HTTP(S) (for web-based or cloud deployments)

    Example usage:
    ```python
    def evaluation_fn(param1: float, param2: float) -> dict[str, float]:
        return {"objective1": param1 + param2}

    coordinator = OptimizationCoordinator.from_dict(...)
    config = SystemConfig(local_workers=4)

    with OptimizationSystem(coordinator, evaluation_fn, config) as system:
        # System is running with workers
        system.wait_until_complete()

        # Get results
        final_state = system.get_final_state()
        print(f"Best result: {final_state.best_result}")
    ```
    """

    def __init__(
        self,
        coordinator: OptimizationCoordinator,
        evaluation_fn: Callable[..., dict[str, float]],
        config: SystemConfig = SystemConfig(),
    ):
        """
        Initialize the optimization system.

        Args:
            coordinator: The optimization coordinator instance
            evaluation_fn: Function that evaluates parameters and returns objectives
            config: System configuration
        """
        self.coordinator = coordinator
        self.evaluation_fn = evaluation_fn
        self.config = config

        self.logger = setup_logging("OptimizationSystem")
        self.active_workers = mp.Value("i", 0)

        # Initialize components
        self._scheduler_process = None
        self._local_workers = []
        self._server = None
        self._running = False

        # Setup signal handlers
        self._setup_signal_handlers()

    def start(self):
        """Start the optimization system."""
        if self._running:
            return

        self.logger.info(
            f"Starting optimization system:\n"
            f"- Local workers: {self.config.local_workers} ({self.config.local_transport})\n"
            f"- TCP endpoint: {self.config.network.tcp_host}:{self.config.network.tcp_port}\n"
            f"- HTTP endpoint: {self.config.network.rest_host}:{self.config.network.rest_port}"
        )

        # Start scheduler
        self._scheduler_process = mp.Process(target=self._run_scheduler)
        self._scheduler_process.start()

        # Give scheduler time to initialize
        time.sleep(0.5)

        # Start REST API server
        self._server = Server(
            host=self.config.network.rest_host,
            port=self.config.network.rest_port,
            active_workers=self.active_workers,
        )
        self._server.start()

        # Start local workers if configured
        if self.config.local_workers > 0:
            self._start_local_workers()

        self._running = True
        self.logger.info("System started successfully")

    def stop(self):
        """Stop the optimization system."""
        if not self._running:
            return

        self.logger.info("Stopping optimization system...")

        # Stop scheduler
        if self._scheduler_process:
            self._scheduler_process.terminate()
            self._scheduler_process.join(timeout=2)

        # Stop REST server
        if self._server:
            self._server.stop()

        # Stop local workers
        for p in self._local_workers:
            if p.is_alive():
                p.terminate()
                p.join(timeout=1)

        self._running = False
        self.logger.info("System stopped successfully")

    def wait_until_complete(self, timeout: float = None) -> bool:
        """
        Wait until all workers have completed.

        Args:
            timeout: Maximum time to wait in seconds. None for no timeout.

        Returns:
            True if completed normally, False if timed out.
        """
        start_time = time.time()
        while self._running:
            with self.active_workers.get_lock():
                if self.active_workers.value <= 0:
                    return True

            if timeout and (time.time() - start_time) > timeout:
                return False

            time.sleep(0.1)
        return True

    def get_final_state(self) -> OptimizationState:
        """Get the final optimization state."""
        return self.coordinator.get_state()

    def get_connection_info(self) -> dict[str, str]:
        """
        Get connection information for remote workers.

        Returns:
            Dictionary containing connection endpoints for different protocols.
        """
        return {
            "tcp": f"tcp://{self.config.network.tcp_host}:{self.config.network.tcp_port}",
            "http": f"http{'s' if self.config.network.use_ssl else ''}://"
            f"{self.config.network.rest_host}:{self.config.network.rest_port}",
        }

    def _run_scheduler(self):
        """Internal method to run the scheduler."""
        scheduler = SchedulerProcess(self.coordinator)
        scheduler.active_workers = self.active_workers
        scheduler.run()

    def _start_local_workers(self):
        """Start local worker processes."""
        for i in range(self.config.local_workers):
            worker = LocalWorker(
                worker_id=i,
                active_workers=self.active_workers,
                evaluation_fn=self.evaluation_fn,
                transport=self.config.local_transport,
                network_config=self.config.network,
            )
            p = mp.Process(target=worker.run)
            p.start()
            self._local_workers.append(p)

    def _setup_signal_handlers(self):
        """Setup graceful shutdown on signals."""

        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}")
            self.stop()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


def run_optimization(
    coordinator: OptimizationCoordinator,
    evaluation_fn: Callable[..., dict[str, float]],
    config: SystemConfig = SystemConfig(),
    timeout: float = None,
) -> OptimizationState:
    """
    Convenience function to run an optimization experiment.

    Args:
        coordinator: The optimization coordinator instance
        evaluation_fn: Function that evaluates parameters and returns objectives
        config: System configuration
        timeout: Maximum time to run in seconds

    Returns:
        Final optimization state
    """
    with HOLA(coordinator, evaluation_fn, config) as system:
        completed = system.wait_until_complete(timeout)
        if not completed:
            logging.warning("Optimization timed out")
        return system.get_final_state()
