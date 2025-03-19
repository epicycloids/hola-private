"""Worker components for distributed optimization."""

from hola.worker.local import LocalWorker
from hola.worker.util import spawn_local_worker

__all__ = ["LocalWorker", "spawn_local_worker"]