"""HOLA Server Implementation using FastAPI.

This module implements a FastAPI-based server for the HOLA optimization algorithm,
providing:
- Centralized parameter sampling and result storage
- Asynchronous worker support
- Configuration management
- State persistence
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional
import uuid

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from hola.core.algorithm import HOLA
from hola.core.objective import ObjectiveConfig, ObjectiveName
from hola.core.params import ParamConfig, ParamName


class WorkerState(str, Enum):
    """States for worker tasks."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class WorkerTask(BaseModel):
    """Represents a worker task with parameters to evaluate."""
    task_id: str
    parameters: Dict[ParamName, float]
    state: WorkerState = WorkerState.PENDING


class WorkerResult(BaseModel):
    """Results from a worker's parameter evaluation."""
    task_id: str
    objectives: Dict[ObjectiveName, float]
    error: Optional[str] = None


class BatchTaskRequest(BaseModel):
    """Request for batch task creation."""
    batch_size: int = Field(gt=0, default=1)


class BatchResult(BaseModel):
    """Batch of results from multiple tasks."""
    results: List[WorkerResult]


class OptimizationState(BaseModel):
    """Current state of the optimization process."""
    num_completed: int
    num_pending: int
    num_failed: int
    best_objectives: Dict[ObjectiveName, float]
    best_parameters: Dict[ParamName, float]


@dataclass
class ServerState:
    """Server state management."""
    hola: Optional[HOLA]
    tasks: Dict[str, WorkerTask]
    lock: asyncio.Lock

    @property
    def num_completed(self) -> int:
        return sum(1 for task in self.tasks.values() if task.state == WorkerState.COMPLETED)

    @property
    def num_pending(self) -> int:
        return sum(1 for task in self.tasks.values()
                  if task.state in (WorkerState.PENDING, WorkerState.RUNNING))

    @property
    def num_failed(self) -> int:
        return sum(1 for task in self.tasks.values() if task.state == WorkerState.FAILED)


def create_app(hola_instance: Optional[HOLA] = None) -> FastAPI:
    """Create FastAPI application instance.

    :param hola_instance: Optional HOLA instance to use
    :return: Configured FastAPI application
    """
    app = FastAPI()

    # Server state for this app instance
    server_state: Optional[ServerState] = None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        """Initialize and cleanup server state."""
        nonlocal server_state
        server_state = ServerState(
            hola=hola_instance,
            tasks={},
            lock=asyncio.Lock()
        )
        yield
        server_state = None

    app.router.lifespan_context = lifespan

    @app.post("/optimizer/initialize")
    async def initialize_optimizer(
        params_config: Dict[ParamName, Dict[str, Any]],
        objectives_config: Dict[ObjectiveName, Dict[str, Any]],
        min_samples: Optional[int] = None,
    ) -> JSONResponse:
        """Initialize the HOLA optimizer with configuration."""
        if server_state is None:
            raise HTTPException(status_code=500, detail="Server state not initialized")

        try:
            async with server_state.lock:
                # Convert raw configs to proper objects
                param_configs = {
                    name: ParamConfig.parse(config)
                    for name, config in params_config.items()
                }
                objective_configs = {
                    name: ObjectiveConfig.parse(config)
                    for name, config in objectives_config.items()
                }

                # Create HOLA instance if none provided
                if server_state.hola is None:
                    server_state.hola = HOLA(
                        params_config=param_configs,
                        objectives_config=objective_configs,
                        min_samples=min_samples
                    )
                server_state.tasks = {}

            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={"message": "Optimizer initialized successfully"}
            )

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to initialize optimizer: {str(e)}"
            )

    @app.put("/optimizer/params")
    async def update_params_config(
        updates: Dict[ParamName, Dict[str, Any]]
    ) -> JSONResponse:
        """Update parameter configurations."""
        if server_state is None or server_state.hola is None:
            raise HTTPException(
                status_code=500,
                detail="Optimizer not initialized"
            )

        try:
            async with server_state.lock:
                param_updates = {
                    name: ParamConfig.parse(config)
                    for name, config in updates.items()
                }
                server_state.hola.update_params_config(param_updates)

            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={"message": "Parameter configurations updated successfully"}
            )

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to update parameter configurations: {str(e)}"
            )

    @app.put("/optimizer/objectives")
    async def update_objectives_config(
        updates: Dict[ObjectiveName, Dict[str, Any]]
    ) -> JSONResponse:
        """Update objective configurations."""
        if server_state is None or server_state.hola is None:
            raise HTTPException(
                status_code=500,
                detail="Optimizer not initialized"
            )

        try:
            async with server_state.lock:
                objective_updates = {
                    name: ObjectiveConfig.parse(config)
                    for name, config in updates.items()
                }
                server_state.hola.update_objectives_config(objective_updates)

            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={"message": "Objective configurations updated successfully"}
            )

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to update objective configurations: {str(e)}"
            )

    @app.post("/worker/request_task")
    async def request_task(request: BatchTaskRequest = BatchTaskRequest()) -> List[WorkerTask]:
        """Request one or more tasks with parameters to evaluate."""
        if server_state is None or server_state.hola is None:
            raise HTTPException(
                status_code=500,
                detail="Optimizer not initialized"
            )

        try:
            async with server_state.lock:
                # Sample parameters in batch
                tasks = []
                for _ in range(request.batch_size):
                    params = server_state.hola.sample()
                    task_id = str(uuid.uuid4())
                    task = WorkerTask(
                        task_id=task_id,
                        parameters=params,
                        state=WorkerState.PENDING
                    )
                    server_state.tasks[task_id] = task
                    tasks.append(task)

            return tasks

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create task: {str(e)}"
            )

    @app.post("/worker/submit_result")
    async def submit_result(result: WorkerResult | BatchResult) -> JSONResponse:
        """Submit results from completed tasks."""
        if server_state is None or server_state.hola is None:
            raise HTTPException(
                status_code=500,
                detail="Optimizer not initialized"
            )

        try:
            async with server_state.lock:
                # Convert single result to batch format
                results = (
                    result.results if isinstance(result, BatchResult)
                    else [result]
                )

                num_processed = 0
                num_failed = 0

                for single_result in results:
                    # Verify task exists
                    task = server_state.tasks.get(single_result.task_id)
                    if task is None:
                        raise HTTPException(
                            status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Task {single_result.task_id} not found"
                        )

                    # Handle failed evaluation
                    if single_result.error:
                        task.state = WorkerState.FAILED
                        num_failed += 1
                        continue

                    # Add successful result to optimizer
                    server_state.hola.add(single_result.objectives, task.parameters)
                    task.state = WorkerState.COMPLETED
                    num_processed += 1

                message = f"Processed {num_processed} results"
                if num_failed > 0:
                    message += f", {num_failed} failed"

            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={"message": message}
            )

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to record results: {str(e)}"
            )

    @app.get("/optimizer/state")
    async def get_state() -> OptimizationState:
        """Get current optimization state."""
        if server_state is None or server_state.hola is None:
            raise HTTPException(
                status_code=500,
                detail="Optimizer not initialized"
            )

        try:
            async with server_state.lock:
                result = server_state.hola.get_dataframe()

                return OptimizationState(
                    num_completed=server_state.num_completed,
                    num_pending=server_state.num_pending,
                    num_failed=server_state.num_failed,
                    best_objectives=result.best_objectives,
                    best_parameters=result.best_params
                )

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get optimization state: {str(e)}"
            )

    @app.post("/optimizer/save")
    async def save_state(filename: str) -> JSONResponse:
        """Save current optimization state to file."""
        if server_state is None or server_state.hola is None:
            raise HTTPException(
                status_code=500,
                detail="Optimizer not initialized"
            )

        try:
            async with server_state.lock:
                server_state.hola.save(Path(filename))

            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={"message": f"State saved to {filename}"}
            )

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save state: {str(e)}"
            )

    @app.post("/optimizer/load")
    async def load_state(filename: str) -> JSONResponse:
        """Load optimization state from file."""
        if server_state is None or server_state.hola is None:
            raise HTTPException(
                status_code=500,
                detail="Optimizer not initialized"
            )

        try:
            async with server_state.lock:
                server_state.hola.load(Path(filename))
                server_state.tasks = {}  # Reset tasks

            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={"message": f"State loaded from {filename}"}
            )

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load state: {str(e)}"
            )

    return app