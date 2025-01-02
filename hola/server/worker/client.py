"""HOLA Worker Client Implementation.

This module provides a client for running worker processes that:
1. Request tasks from the HOLA server
2. Execute objective function evaluations
3. Submit results back to the server
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Dict, Optional, Union

import aiohttp
from pydantic import BaseModel, Field

from hola.core.objective import ObjectiveName
from hola.core.params import ParamName


class WorkerConfig(BaseModel):
    """Configuration for a worker client."""
    server_url: str
    batch_size: int = Field(gt=0, default=1)
    poll_interval: float = 1.0
    max_retries: int = 3
    retry_delay: float = 1.0
    parallel_evaluations: int = Field(gt=0, default=1)


class HOLAWorker:
    """Worker client for running HOLA optimizations.

    Example:
        >>> async def evaluate(params: Dict[str, float]) -> Dict[str, float]:
        ...     # Train model with params
        ...     model = await train_model(**params)
        ...     return {
        ...         "accuracy": model.evaluate(),
        ...         "training_time": model.train_time
        ...     }
        >>>
        >>> worker = HOLAWorker(
        ...     server_url="http://localhost:8000",
        ...     evaluation_fn=evaluate
        ... )
        >>> await worker.run()
    """

    def __init__(
        self,
        evaluation_fn: Union[
            Callable[[Dict[ParamName, float]], Dict[ObjectiveName, float]],
            Callable[[Dict[ParamName, float]], Awaitable[Dict[ObjectiveName, float]]]
        ],
        config: Union[WorkerConfig, Dict[str, Any]],
    ) -> None:
        """Initialize worker client.

        :param evaluation_fn: Function that evaluates parameters and returns objectives.
            Can be synchronous or asynchronous.
        :param config: Worker configuration or dictionary of settings
        """
        self.evaluation_fn = evaluation_fn
        self.config = WorkerConfig.model_validate(config)
        self._running = False
        self._session: Optional[aiohttp.ClientSession] = None

    async def _ensure_session(self) -> None:
        """Ensure aiohttp session exists."""
        if self._session is None:
            self._session = aiohttp.ClientSession()

    async def _close_session(self) -> None:
        """Close aiohttp session if it exists."""
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def _request_tasks(self) -> Optional[list[Dict[str, Any]]]:
        """Request a batch of tasks from the server.

        :return: List of task data if successful, None if request failed
        """
        await self._ensure_session()
        assert self._session is not None

        for attempt in range(self.config.max_retries):
            try:
                async with self._session.post(
                    f"{self.config.server_url}/worker/request_task",
                    json={"batch_size": self.config.batch_size}
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 404:
                        return None
            except aiohttp.ClientError:
                if attempt + 1 < self.config.max_retries:
                    await asyncio.sleep(self.config.retry_delay)
                continue
        return None

    async def _submit_results(
        self,
        results: list[Dict[str, Any]]
    ) -> bool:
        """Submit batch results to the server.

        :param results: List of result objects
        :return: True if submission was successful
        """
        await self._ensure_session()
        assert self._session is not None

        batch_result = {"results": results}

        for attempt in range(self.config.max_retries):
            try:
                async with self._session.post(
                    f"{self.config.server_url}/worker/submit_result",
                    json=batch_result
                ) as response:
                    return response.status == 200
            except aiohttp.ClientError:
                if attempt + 1 < self.config.max_retries:
                    await asyncio.sleep(self.config.retry_delay)
                continue
        return False

    async def _process_batch(self, task_batch: list[Dict[str, Any]]) -> None:
        """Process a batch of tasks.

        :param task_batch: List of task data from server
        """
        # Create evaluation tasks
        async def evaluate_task(task_data: Dict[str, Any]) -> Dict[str, Any]:
            task_id = task_data["task_id"]
            parameters = task_data["parameters"]

            try:
                # Call evaluation function (handle both sync and async)
                if asyncio.iscoroutinefunction(self.evaluation_fn):
                    objectives = await self.evaluation_fn(parameters)
                else:
                    objectives = self.evaluation_fn(parameters)

                return {
                    "task_id": task_id,
                    "objectives": objectives
                }
            except Exception as e:
                return {
                    "task_id": task_id,
                    "error": str(e)
                }

        # Process tasks with bounded concurrency
        semaphore = asyncio.Semaphore(self.config.parallel_evaluations)
        async def bounded_evaluate(task_data: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                return await evaluate_task(task_data)

        # Evaluate all tasks
        results = await asyncio.gather(
            *[bounded_evaluate(task) for task in task_batch]
        )

        # Submit all results
        await self._submit_results(results)

    async def run(self) -> None:
        """Run the worker loop.

        Continuously requests and processes batches of tasks until stopped.
        """
        self._running = True
        try:
            while self._running:
                # Request batch of tasks
                task_batch = await self._request_tasks()
                if not task_batch:
                    # No tasks available, wait before retry
                    await asyncio.sleep(self.config.poll_interval)
                    continue

                # Process batch
                await self._process_batch(task_batch)

        finally:
            self._running = False
            await self._close_session()

    async def stop(self) -> None:
        """Stop the worker loop gracefully."""
        self._running = False