import asyncio
from dataclasses import dataclass
import logging
from datetime import datetime, timezone
from typing import Dict, Set
from uuid import UUID

from hola.messages.worker import (
    EvaluationContent,
    SampleRequestContent,
    WorkerErrorContent,
    WorkerMessage,
)


@dataclass
class WorkerStatusConfig:
    heartbeat_timeout: float = 60.0
    cleanup_interval: float = 30.0


class WorkerStatusManager:
    """Simple worker status tracking."""

    def __init__(self, config: WorkerStatusConfig | None = None):
        self.config = config or WorkerStatusConfig()
        self._workers: Dict[UUID, datetime] = {}  # worker_id -> last_heartbeat
        self._evaluating: Set[UUID] = set()  # workers currently evaluating
        self._lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task | None = None
        self._running = False

    async def handle_message(
        self, worker_id: UUID, message: WorkerMessage, logger: logging.Logger
    ) -> None:
        """Handle incoming worker message."""
        await self.heartbeat(worker_id)

        match message.content:
            case SampleRequestContent():
                await self.end_evaluation(worker_id)
                logger.debug(f"Worker {worker_id} ready for new evaluation")
            case EvaluationContent():
                await self.start_evaluation(worker_id)
                logger.debug(f"Worker {worker_id} started evaluation")
            case WorkerErrorContent():
                await self.end_evaluation(worker_id)
                logger.error(f"Worker {worker_id} reported error: {message.content.error}")

    async def heartbeat(self, worker_id: UUID) -> None:
        """Record worker heartbeat."""
        async with self._lock:
            self._workers[worker_id] = datetime.now(timezone.utc)

    async def start_evaluation(self, worker_id: UUID) -> None:
        """Record worker starting evaluation."""
        async with self._lock:
            self._evaluating.add(worker_id)

    async def end_evaluation(self, worker_id: UUID) -> None:
        """Record worker completing evaluation."""
        async with self._lock:
            self._evaluating.discard(worker_id)

    async def cleanup(self, logger: logging.Logger) -> None:
        """Remove stale workers."""
        now = datetime.now(timezone.utc)
        async with self._lock:
            stale = {
                wid
                for wid, last_seen in self._workers.items()
                if (now - last_seen).total_seconds() > self.heartbeat_timeout
            }
            for wid in stale:
                del self._workers[wid]
                self._evaluating.discard(wid)
                logger.warning(f"Removed stale worker {wid}")

    async def get_active_workers(self) -> Set[UUID]:
        """Get set of currently active workers."""
        now = datetime.now(timezone.utc)
        async with self._lock:
            return {
                wid
                for wid, last_seen in self._workers.items()
                if (now - last_seen).total_seconds() <= self.heartbeat_timeout
            }

    async def get_evaluating_workers(self) -> Set[UUID]:
        """Get set of workers currently performing evaluations."""
        async with self._lock:
            return self._evaluating.copy()
