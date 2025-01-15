import asyncio
import logging
from datetime import datetime, timezone
from typing import Callable, Coroutine
from uuid import UUID

import msgspec
import zmq
from msgspec import Struct

from hola.core.coordinator import OptimizationState
from hola.server.base import ExecutionState


class OptimizationStatus(Struct, frozen=True):
    pass


class TaskSupervisor:
    """Manages and supervises async tasks."""

    def __init__(self, logger: logging.Logger):
        self.tasks: set[asyncio.Task] = set()
        self.logger = logger
        self._running = False

    async def run(self, tasks_dict: dict[str, Callable[[], Coroutine]]) -> None:
        self._running = True
        self.tasks = {asyncio.create_task(coro(), name=name) for name, coro in tasks_dict.items()}

        while self._running:
            done, pending = await asyncio.wait(
                self.tasks, return_when=asyncio.FIRST_COMPLETED, timeout=1.0
            )

            if not done and self._running:
                continue

            for task in done:
                self.tasks.remove(task)
                exc = task.exception()
                if exc:
                    self.logger.error(f"Task {task.get_name()} failed with error: {exc}")
                    if self._running:
                        new_task = asyncio.create_task(
                            tasks_dict[task.get_name()](), name=task.get_name()
                        )
                        self.tasks.add(new_task)
                else:
                    self.logger.warning(f"Task {task.get_name()} completed unexpectedly")
                    if self._running:
                        self.tasks.add(
                            asyncio.create_task(tasks_dict[task.get_name()](), name=task.get_name())
                        )

    async def stop(self) -> None:
        self._running = False
        for task in self.tasks:
            task.cancel()

        try:
            await asyncio.wait_for(asyncio.gather(*self.tasks, return_exceptions=True), timeout=5.0)
        except asyncio.TimeoutError:
            self.logger.warning("Some tasks did not shut down cleanly")


class StatusBroadcaster:
    """Handles periodic status broadcasts."""

    def __init__(
        self,
        socket: zmq.Socket,
        state: ExecutionState,
        interval: float = 1.0,
    ):
        self.socket = socket
        self.state = state
        self.interval = interval
        self._running = True

    async def stop(self) -> None:
        self._running = False

    async def run(self) -> None:
        while self._running:
            try:
                # Get current optimization state
                opt_state = await self.state.coordinator.get_state()
                worker_statuses = self.state.worker_status.get_all_statuses()

                # Create status update
                status = OptimizationStatus(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    optimization_state=opt_state,
                    worker_status=worker_statuses,
                    num_active_workers=len(self.state.worker_status.get_active_workers()),
                )

                if self._running:
                    await self.socket.send_multipart([b"status", msgspec.json.encode(status)])
                    self.state.logger.debug(f"Broadcast status update: {status}")

                await asyncio.sleep(self.interval)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.state.logger.error(f"Error in status broadcast loop: {e}")
                raise
