"""
Core system class for running optimizations with context manager support.
"""

import multiprocessing as mp
import signal
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from hola.core.coordinator import OptimizationCoordinator, OptimizationState
from hola.core.objectives import ObjectiveName
from hola.server import SchedulerProcess, Server
from hola.utils.logging import setup_logging
from hola.worker.util import start_workers


@dataclass
class SystemConfig:
    """Configuration for the optimization system."""

    local_workers: int = 4
    """Number of local worker processes."""

    use_ipc_ratio: float = 0.5
    """Fraction of workers that should use IPC (vs TCP)."""

    server_host: str = "localhost"
    """Host for the REST API server."""

    server_port: int = 8000
    """Port for the REST API server."""

    timeout: Optional[float] = None
    """Maximum time to run optimization (None for no timeout)."""


class HOLA:
    """
    High-level optimization system with context manager support.

    This class provides a user-friendly way to run optimizations with:
    - Support for local and network workers
    - Dynamic worker addition/removal
    - Graceful shutdown

    Example:
        ```python
        coordinator = OptimizationCoordinator.from_dict(...)

        with HOLA(coordinator, evaluate_fn, SystemConfig()) as system:
            # System runs automatically
            system.wait_until_complete()
            result = system.get_final_state()
        ```
    """

    def __init__(
        self,
        coordinator: OptimizationCoordinator,
        evaluation_fn: Callable[..., dict[ObjectiveName, float]],
        config: SystemConfig = SystemConfig(),
    ):
        """
        Initialize the optimization system.

        Args:
            coordinator: The optimization coordinator
            evaluation_fn: Function that evaluates parameters and returns objectives
            config: System configuration
        """
        self.coordinator = coordinator
        self.evaluation_fn = evaluation_fn
        self.config = config

        self.logger = setup_logging("HOLA")
        self.active_workers = mp.Value("i", 0)
        self.running = False
        self.completed = threading.Event()

        # System components
        self.scheduler = None
        self.scheduler_process = None
        self.server = None
        self.worker_processes = []

    def __enter__(self):
        """Start the optimization system when entering the context."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources when exiting the context."""
        self.stop()

    def start(self):
        """Start the optimization system."""
        if self.running:
            self.logger.warning("System is already running")
            return

        # Remember original signal handler
        self.original_sigint_handler = signal.getsignal(signal.SIGINT)

        # Start scheduler
        self.scheduler = SchedulerProcess(self.coordinator)
        self.scheduler.active_workers = self.active_workers
        self.scheduler_process = mp.Process(target=self.scheduler.run)
        self.scheduler_process.start()

        self.logger.info("Scheduler started")
        time.sleep(0.5)  # Give scheduler time to start

        # Start server
        self.server = Server(
            host=self.config.server_host,
            port=self.config.server_port,
            active_workers=self.active_workers
        )
        self.server.start()
        self.logger.info(f"Server started on http://{self.config.server_host}:{self.config.server_port}")

        # Start workers
        self.worker_processes = start_workers(
            num_workers=self.config.local_workers,
            active_workers=self.active_workers,
            evaluation_fn=self.evaluation_fn,
            use_ipc_ratio=self.config.use_ipc_ratio
        )
        self.logger.info(f"Started {self.config.local_workers} worker processes")

        self.running = True

        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_optimization)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop(self):
        """Stop the optimization system."""
        if not self.running:
            return

        self.logger.info("Stopping optimization system")
        self.running = False

        # Wait for monitor thread to exit
        if hasattr(self, 'monitor_thread') and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2)

        # Clean up components
        if self.scheduler_process and self.scheduler_process.is_alive():
            self.scheduler_process.terminate()
            self.scheduler_process.join(timeout=2)

        if self.server:
            self.server.stop()

        # Terminate any remaining worker processes
        for p in self.worker_processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=1)

        # Restore signal handler
        signal.signal(signal.SIGINT, self.original_sigint_handler)

        self.logger.info("Optimization system stopped")

    def wait_until_complete(self, timeout: Optional[float] = None):
        """
        Wait for the optimization to complete.

        Args:
            timeout: Maximum time to wait, in seconds (None for no timeout)

        Returns:
            True if completed, False if timed out
        """
        # Use either the method timeout or the system timeout
        effective_timeout = timeout or self.config.timeout
        return self.completed.wait(timeout=effective_timeout)

    def get_final_state(self) -> OptimizationState:
        """
        Get the current optimization state.

        Returns:
            The current optimization state
        """
        return self.coordinator.get_state()

    def add_workers(self, num_workers: int, use_ipc: bool = True):
        """
        Add more worker processes to the running optimization.

        Args:
            num_workers: Number of workers to add
            use_ipc: Whether to use IPC (True) or TCP (False)
        """
        if not self.running:
            self.logger.warning("System is not running, can't add workers")
            return

        new_processes = start_workers(
            num_workers=num_workers,
            active_workers=self.active_workers,
            evaluation_fn=self.evaluation_fn,
            use_ipc_ratio=1.0 if use_ipc else 0.0
        )

        self.worker_processes.extend(new_processes)
        self.logger.info(f"Added {num_workers} {'IPC' if use_ipc else 'TCP'} workers")

    def _monitor_optimization(self):
        """Monitor the optimization progress and detect completion."""
        start_time = time.time()

        while self.running:
            # Check if timed out
            if self.config.timeout and time.time() - start_time > self.config.timeout:
                self.logger.info(f"Optimization timed out after {self.config.timeout}s")
                self.completed.set()
                self.running = False
                break

            # Check if all workers are done
            with self.active_workers.get_lock():
                current_workers = self.active_workers.value

            if current_workers <= 0:
                self.logger.info("All workers finished")
                self.completed.set()
                break

            # Log current status periodically
            state = self.coordinator.get_state()
            self.logger.info(
                f"Active workers: {current_workers}, "
                f"Total evaluations: {state.total_evaluations}"
            )

            time.sleep(1)