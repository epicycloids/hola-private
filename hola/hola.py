import asyncio
import logging
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Callable

from hola.client.client import OptimizationClient
from hola.core.coordinator import OptimizationCoordinator
from hola.core.objectives import ObjectiveName, ObjectiveConfig
from hola.core.parameters import ParameterName, ParameterConfig
from hola.core.samplers import ClippedGaussianMixtureSampler, ExploreExploitSampler, SobolSampler
from hola.server.config import ConnectionConfig, SocketType
from hola.server.server import OptimizationServer
from hola.worker.worker import OptimizationWorker


def setup_logging(level=logging.INFO):
    # Clear any existing handlers
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # Configure specific loggers
    loggers = [
        'hola.server.server',
        'hola.worker.worker',
        'hola.client.client',
        'hola.core.coordinator',
        'hola.local'
    ]

    for logger_name in loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
        # Ensure logger propagates to root
        logger.propagate = True

logger = logging.getLogger(__name__)

class LocalOptimizer:

    def __init__(
        self,
        evaluation_fn: Callable[..., dict[str, float]],
        objectives_config: dict[str, dict[str, Any]],
        parameters_config: dict[str, dict[str, Any]],
        n_workers: int = -1,
        use_exploit: bool = True,
        temp_dir: str | None = None,
        log_level: int = logging.INFO
    ):
        setup_logging(log_level)
        self.evaluation_fn = evaluation_fn
        self.objectives_config = objectives_config
        self.parameters_config = parameters_config
        self.n_workers = multiprocessing.cpu_count() if n_workers <= 0 else n_workers
        self.use_exploit = use_exploit
        self.temp_dir = temp_dir or tempfile.gettempdir()

        # Components that will be initialized during optimization
        self.server = None
        self.workers: list[OptimizationWorker] = []
        self.client = None
        self.tasks: list[asyncio.Task] = []

    def _create_sampler(self) -> SobolSampler | ExploreExploitSampler:
        dimension = len(self.parameters_config)

        if not self.use_exploit:
            return SobolSampler(dimension=dimension)

        explore_sampler = SobolSampler(dimension=dimension)
        exploit_sampler = ClippedGaussianMixtureSampler(
            dimension=dimension,
            n_components=min(5, len(self.parameters_config))
        )

        return ExploreExploitSampler(
            explore_sampler=explore_sampler,
            exploit_sampler=exploit_sampler
        )

    async def _setup(self):
        # Create unique IPC path
        ipc_path = os.path.join(self.temp_dir, f"hola-local-{os.getpid()}")
        config = ConnectionConfig(
            socket_type=SocketType.IPC,
            ipc_path=ipc_path
        )

        # Create and start server
        sampler = self._create_sampler()
        coordinator = OptimizationCoordinator.from_dict(
            sampler,
            self.objectives_config,
            self.parameters_config
        )
        self.server = OptimizationServer(coordinator, config)
        server_task = asyncio.create_task(self.server.start())
        self.tasks.append(server_task)

        # Give server time to start
        await asyncio.sleep(0.1)

        # Start workers
        for _ in range(self.n_workers):
            worker = OptimizationWorker(config, self.evaluation_fn)
            worker_task = asyncio.create_task(worker.start())
            self.tasks.append(worker_task)
            self.workers.append(worker)

        # Start client
        self.client = OptimizationClient(config)
        await self.client.start()

        # Initialize optimization
        await self.client.initialize(
            objectives_config=self.objectives_config,
            parameters_config=self.parameters_config
        )

    async def _cleanup(self):
        cleanup_timeout = 5  # seconds

        # Stop workers
        for worker in self.workers:
            await worker.stop()

        # Stop client
        if self.client:
            await self.client.stop()

        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Clean up contexts
        if self.server:
            self.server.context.destroy(linger=0)
        for worker in self.workers:
            worker.context.destroy(linger=0)
        if self.client:
            self.client.context.destroy()

        # Clean up IPC files
        ipc_path = os.path.join(self.temp_dir, f"hola-local-{os.getpid()}")
        for suffix in ['-worker', '-client']:
            try:
                Path(ipc_path + suffix).unlink()
            except FileNotFoundError:
                pass

    async def optimize(self, max_evaluations: int = 100, status_interval: float = 1.0):
        try:
            await self._setup()

            while True:
                status = await self.client.get_status()
                if status.status.total_evaluations >= max_evaluations:
                    break

                await asyncio.sleep(status_interval)

            return status.status.best_result

        finally:
            await self._cleanup()

def run_optimization(
    evaluation_fn: Callable[..., dict[str, float]],
    objectives_config: dict[str, dict[str, Any]],
    parameters_config: dict[str, dict[str, Any]],
    max_evaluations: int = 100,
    n_workers: int = -1,
    use_exploit: bool = True,
    temp_dir: str | None = None,
    status_interval: float = 1.0,
    log_level: int = logging.INFO
) -> Any:
    optimizer = LocalOptimizer(
        evaluation_fn=evaluation_fn,
        objectives_config=objectives_config,
        parameters_config=parameters_config,
        n_workers=n_workers,
        use_exploit=use_exploit,
        temp_dir=temp_dir,
        log_level=log_level
    )

    return asyncio.run(optimizer.optimize(
        max_evaluations=max_evaluations,
        status_interval=status_interval
    ))