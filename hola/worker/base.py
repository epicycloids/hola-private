import asyncio
import logging
from typing import Any, Callable
from uuid import UUID

from hola.core.objectives import ObjectiveName
from hola.core.parameters import ParameterName


class WorkerState:

    def __init__(
        self,
        worker_id: UUID,
        objective_fn: Callable[..., dict[ObjectiveName, float]],
        logger: logging.Logger,
    ):
        self.worker_id = worker_id
        self.objective_fn = objective_fn
        self.logger = logger

    async def evaluate(self, params: dict[ParameterName, Any]) -> dict[ObjectiveName, float]:
        """Evaluate the objective function with the given parameters."""
        if asyncio.iscoroutinefunction(self.objective_fn):
            return await self.objective_fn(**params)
        else:
            return await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.objective_fn(**params)
            )
