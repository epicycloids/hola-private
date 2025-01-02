"""Extended HOLA implementation with unified local and server modes.

This module extends the core HOLA implementation to provide a unified interface
for both local and server-based optimization.
"""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
import uvicorn

from fastapi import FastAPI
from pydantic import BaseModel, Field

from hola.core.algorithm import HOLA as CoreHOLA
from hola.core.objective import ObjectiveConfig, ObjectiveName
from hola.core.params import ParamConfig, ParamName
from hola.server.api.app import create_app
from hola.server.worker.client import HOLAWorker, WorkerConfig


class OptimizationMode(str, Enum):
    """Optimization execution modes."""
    LOCAL = "local"     # Direct local execution
    SERVER = "server"   # Run as server
    CLIENT = "client"   # Connect to existing server


class ServerConfig(BaseModel):
    """Configuration for server mode."""
    host: str = "localhost"
    port: int = Field(gt=0, default=8000)
    workers: int = Field(ge=0, default=0)  # Number of local workers (0 for external only)
    worker_batch_size: int = Field(gt=0, default=1)
    worker_parallel_evaluations: int = Field(gt=0, default=1)


class ClientConfig(BaseModel):
    """Configuration for client mode."""
    server_url: str
    batch_size: int = Field(gt=0, default=1)
    parallel_evaluations: int = Field(gt=0, default=1)


class HOLA(CoreHOLA):
    """Extended HOLA implementation with server support.

    This class extends the core HOLA implementation to provide:
    1. Unified interface for local and server-based optimization
    2. Built-in server capabilities
    3. Simplified client connection
    4. Automatic worker management

    Example:
        >>> # Local mode (same as before)
        >>> hola = HOLA(params_config=params, objectives_config=objectives)
        >>> result = await hola.tune(evaluate_fn, num_runs=100)
        >>>
        >>> # Server mode
        >>> server = HOLA(
        ...     params_config=params,
        ...     objectives_config=objectives,
        ...     mode="server",
        ...     server_config={"port": 8000, "workers": 2}
        ... )
        >>> await server.run()
        >>>
        >>> # Client mode
        >>> client = HOLA(
        ...     params_config=params,
        ...     objectives_config=objectives,
        ...     mode="client",
        ...     client_config={"server_url": "http://localhost:8000"}
        ... )
        >>> result = await client.tune(evaluate_fn, num_runs=100)
    """

    def __init__(
        self,
        params_config: Dict[ParamName, ParamConfig],
        objectives_config: Dict[ObjectiveName, ObjectiveConfig],
        mode: Union[str, OptimizationMode] = OptimizationMode.LOCAL,
        server_config: Optional[Union[Dict[str, Any], ServerConfig]] = None,
        client_config: Optional[Union[Dict[str, Any], ClientConfig]] = None,
        **kwargs: Any
    ) -> None:
        """Initialize HOLA instance.

        :param params_config: Parameter configuration
        :param objectives_config: Objective configuration
        :param mode: Optimization mode (local, server, or client)
        :param server_config: Server configuration if in server mode
        :param client_config: Client configuration if in client mode
        :param kwargs: Additional arguments passed to core HOLA
        """
        super().__init__(params_config, objectives_config, **kwargs)

        # Set mode
        self.mode = OptimizationMode(mode)

        # Initialize mode-specific configuration
        if self.mode == OptimizationMode.SERVER:
            if server_config is None:
                server_config = {}
            self.server_config = (
                server_config if isinstance(server_config, ServerConfig)
                else ServerConfig(**server_config)
            )
            self._app: Optional[FastAPI] = None
            self._server_task: Optional[asyncio.Task] = None
            self._worker_tasks: List[asyncio.Task] = []

        elif self.mode == OptimizationMode.CLIENT:
            if client_config is None:
                raise ValueError("client_config required for client mode")
            self.client_config = (
                client_config if isinstance(client_config, ClientConfig)
                else ClientConfig(**client_config)
            )

        self._exit_stack = AsyncExitStack()

    async def __aenter__(self) -> HOLA:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self._exit_stack.aclose()

    async def run_server(self) -> None:
        """Run the optimization server.

        Starts the FastAPI server and optional local workers.
        """
        if self.mode != OptimizationMode.SERVER:
            raise RuntimeError("run_server() only available in server mode")

        # Create and configure FastAPI app
        self._app = create_app()

        # Start server in background task
        config = uvicorn.Config(
            self._app,
            host=self.server_config.host,
            port=self.server_config.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        self._server_task = asyncio.create_task(server.serve())

        # Start local workers if configured
        for _ in range(self.server_config.workers):
            worker = HOLAWorker(
                evaluation_fn=self._evaluation_fn,
                config=WorkerConfig(
                    server_url=f"http://{self.server_config.host}:{self.server_config.port}",
                    batch_size=self.server_config.worker_batch_size,
                    parallel_evaluations=self.server_config.worker_parallel_evaluations
                )
            )
            task = asyncio.create_task(worker.run())
            self._worker_tasks.append(task)

        # Wait for server to complete
        try:
            await self._server_task
        finally:
            # Stop all workers
            for task in self._worker_tasks:
                task.cancel()
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)

    async def tune(
        self,
        func: Callable[..., Dict[str, float]],
        num_runs: int = 100,
        **kwargs: Any
    ) -> Any:
        """Run hyperparameter optimization.

        Supports all modes (local, server, and client) with a unified interface.

        :param func: Objective function that evaluates parameters
        :param num_runs: Number of optimization runs
        :param kwargs: Additional arguments based on mode
        :return: Optimization results
        """
        self._evaluation_fn = func

        if self.mode == OptimizationMode.LOCAL:
            # Use parent's tune() for local optimization
            return await super().tune(func, num_runs=num_runs, **kwargs)

        elif self.mode == OptimizationMode.SERVER:
            # Start server and wait
            return await self.run_server()

        else:  # CLIENT mode
            # Create and run worker
            worker = HOLAWorker(
                evaluation_fn=func,
                config=WorkerConfig(
                    server_url=self.client_config.server_url,
                    batch_size=self.client_config.batch_size,
                    parallel_evaluations=self.client_config.parallel_evaluations
                )
            )
            async with worker:
                await worker.run()

    @classmethod
    async def create_server(
        cls,
        params_config: Dict[ParamName, ParamConfig],
        objectives_config: Dict[ObjectiveName, ObjectiveConfig],
        server_config: Optional[Union[Dict[str, Any], ServerConfig]] = None,
        **kwargs: Any
    ) -> HOLA:
        """Create and start a HOLA server.

        Convenience method for server creation and startup.

        Example:
            >>> server = await HOLA.create_server(
            ...     params_config=params,
            ...     objectives_config=objectives,
            ...     server_config={"port": 8000, "workers": 2}
            ... )
            >>> # Server is now running

        :param params_config: Parameter configuration
        :param objectives_config: Objective configuration
        :param server_config: Server configuration
        :param kwargs: Additional arguments passed to HOLA
        :return: Running HOLA server instance
        """
        server = cls(
            params_config=params_config,
            objectives_config=objectives_config,
            mode=OptimizationMode.SERVER,
            server_config=server_config,
            **kwargs
        )
        await server.__aenter__()
        return server