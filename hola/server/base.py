import logging
from dataclasses import dataclass

from hola.core.coordinator import OptimizationCoordinator
from hola.server.config import SocketConfig, ZMQSockets
from hola.worker.status import WorkerStatusManager


@dataclass
class ExecutionState:
    coordinator: OptimizationCoordinator
    worker_status: WorkerStatusManager
    sockets: ZMQSockets
    logger: logging.Logger

    @classmethod
    def create(
        cls,
        coordinator: OptimizationCoordinator,
        socket_config: SocketConfig | None = None,
        worker_status_config: WorkerStatusConfig | None = None,
    ) -> "ExecutionState":
        """Factory method to create a properly initialized ExecutionState."""
        # Create logger first
        logger = logging.getLogger("hola")
        logger.setLevel(logging.INFO)

        # Create sockets
        sockets = ZMQSockets(socket_config or SocketConfig())

        # Add ZMQ logging handler
        handler = PUBHandler(sockets.log_pub)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Create worker status manager
        worker_status = WorkerStatusManager(
            config=worker_status_config or WorkerStatusConfig()
        )

        return cls(
            coordinator=coordinator,
            worker_status=worker_status,
            sockets=sockets,
            logger=logger
        )
