import logging

from zmq.log.handlers import PUBHandler

from hola.server.base import ExecutionState, OptimizationCoordinator
from hola.server.config import SocketConfig, ZMQSockets
from hola.server.handlers.client import ClientMessageHandler
from hola.server.handlers.worker import WorkerMessageHandler
from hola.server.loops import ClientMessageLoop, WorkerMessageLoop
from hola.server.status import StatusBroadcaster, TaskSupervisor
from hola.worker.status import WorkerStatusConfig, WorkerStatusManager


class HOLA:
    def __init__(
        self,
        coordinator: OptimizationCoordinator,
        socket_config: SocketConfig | None = None,
        worker_status_config: WorkerStatusConfig | None = None,
        status_broadcast_interval: float = 1.0,
    ):
        # Create execution state which owns everything
        self.state = ExecutionState.create(
            coordinator=coordinator,
            socket_config=socket_config,
            worker_status_config=worker_status_config,
        )

        # Create message loops, passing in execution state
        self.worker_loop = WorkerMessageLoop(
            socket=self.state.sockets.worker_router,
            handler=WorkerMessageHandler(),
            state=self.state
        )

        self.client_loop = ClientMessageLoop(
            socket=self.state.sockets.client_router,
            handler=ClientMessageHandler(),
            state=self.state
        )

        self.status_broadcaster = StatusBroadcaster(
            socket=self.state.sockets.status_pub,
            state=self.state,
            interval=status_broadcast_interval
        )

        self.supervisor = TaskSupervisor(self.state.logger)

    async def serve(self) -> None:
        self.state.logger.info("Starting HOLA server")

        # Start worker status manager
        await self.state.worker_status.start()

        try:
            await self.supervisor.run(
                {
                    "worker_handler": self.worker_loop.run,
                    "client_handler": self.client_loop.run,
                    "status_broadcaster": self.status_broadcaster.run,
                }
            )
        except Exception as e:
            self.state.logger.error(f"Unexpected error in server: {e}")
            raise
        finally:
            await self.stop()

    async def stop(self) -> None:
        self.state.logger.info("Stopping HOLA server")

        # Stop all message handling loops
        await self.worker_loop.stop()
        await self.client_loop.stop()
        await self.status_broadcaster.stop()
        await self.state.worker_status.stop()
        await self.supervisor.stop()
        await self.state.sockets.cleanup()

        self.state.logger.info("Server shutdown complete")
