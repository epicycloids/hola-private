"""Local worker implementation for parameter evaluation."""

from multiprocessing.sharedctypes import Synchronized
import atexit
import time
import uuid
import weakref
from typing import Any, Callable, Optional

import msgspec
import zmq

from hola.core.objectives import ObjectiveName
from hola.messages.protocol import (
    GetSuggestionRequest,
    GetSuggestionResponse,
    Message,
    Result,
    SubmitResultRequest,
    SubmitResultResponse,
)
from hola.utils.logging import setup_logging


class LocalWorker:
    """Worker process that executes objective function evaluations on the local machine.

    This implementation is specifically for Python-based workers running on the same
    machine or network as the scheduler. For non-Python or remote workers, use the HTTP
    API endpoints directly rather than this class.
    """

    # Class-level registry of active worker instances for cleanup
    _workers = weakref.WeakSet()

    def __init__(
        self,
        evaluation_fn: Callable[..., dict[ObjectiveName, float]],
        active_workers: Optional[Synchronized] = None,  # Deprecated, see note below
        worker_id: Optional[str] = None,
        address: str = "ipc:///tmp/scheduler.ipc",
    ):
        """Initialize a local worker.

        Args:
            evaluation_fn: Function to evaluate parameters and return objectives
            active_workers: Optional shared counter for active workers.
                DEPRECATED: This parameter is provided for backward compatibility
                but is not recommended for production use. Worker tracking should
                be implemented on the server side instead.
            worker_id: Optional unique worker ID (UUID string generated if None)
            address: ZeroMQ address for the scheduler
        """
        # Generate a unique worker ID if none provided
        self.worker_id = worker_id if worker_id is not None else str(uuid.uuid4())
        self.active_workers = active_workers  # May be None
        self.evaluation_fn = evaluation_fn
        self.address = address
        self.logger = setup_logging(f"Worker-{self.worker_id[:8]}")
        self.registered = False

        # Add to registry for cleanup
        LocalWorker._workers.add(self)

        # Register cleanup handler on first worker creation
        if not hasattr(LocalWorker, '_cleanup_registered'):
            atexit.register(LocalWorker._cleanup_all_workers)
            LocalWorker._cleanup_registered = True

    def run(self):
        """Run the worker evaluation loop."""
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(self.address)

        self.logger.info(f"Started worker {self.worker_id[:8]} using {self.address}")

        # Register worker if tracking is enabled
        if self.active_workers is not None:
            with self.active_workers.get_lock():
                self.active_workers.value += 1
                self.registered = True
                self.logger.info(
                    f"Worker {self.worker_id[:8]} registered. Active workers: {self.active_workers.value}"
                )

        try:
            while True:
                request = GetSuggestionRequest(worker_id=self.worker_id)
                socket.send(msgspec.json.encode(request))

                response = msgspec.json.decode(socket.recv(), type=Message)
                match response:
                    case GetSuggestionResponse(parameters=None):
                        self.logger.info(
                            f"Worker {self.worker_id[:8]}: No more parameter suggestions available"
                        )
                        return

                    case GetSuggestionResponse(parameters=params):
                        result = self.evaluate_parameters(params)
                        request = SubmitResultRequest(worker_id=self.worker_id, result=result)
                        socket.send(msgspec.json.encode(request))

                        response = msgspec.json.decode(socket.recv(), type=Message)
                        match response:
                            case SubmitResultResponse(success=True):
                                pass
                            case SubmitResultResponse(success=False, error=error):
                                self.logger.error(f"Error submitting result: {error}")

        except Exception as e:
            self.logger.error(f"Worker {self.worker_id[:8]} error: {e}")
        finally:
            self._deregister()
            socket.close()
            context.term()
            self.logger.info(f"Worker {self.worker_id[:8]} shutdown complete")

    def evaluate_parameters(self, params: dict[str, Any]) -> Result:
        """Evaluate parameters using the objective function.

        Args:
            params: Dictionary of parameter values to evaluate

        Returns:
            Result object containing parameters and objective values
        """
        self.logger.info(f"Worker {self.worker_id[:8]} processing parameters {params}")

        try:
            # Call the evaluation function with unpacked parameters
            objectives = self.evaluation_fn(**params)

            # Validate that the return value is a dictionary
            if not isinstance(objectives, dict):
                raise ValueError(f"Evaluation function must return a dict, got {type(objectives)}")

            return Result(parameters=params, objectives=objectives)

        except Exception as e:
            self.logger.error(f"Error evaluating function: {e}")
            raise

    def _deregister(self):
        """Safely deregister the worker from the active count."""
        if self.active_workers is not None and self.registered:
            with self.active_workers.get_lock():
                self.active_workers.value -= 1
                self.registered = False
                self.logger.info(
                    f"Worker {self.worker_id[:8]} deregistered. Active workers: {self.active_workers.value}"
                )

    @classmethod
    def _cleanup_all_workers(cls):
        """Class method to clean up all worker instances during interpreter shutdown."""
        for worker in list(cls._workers):
            try:
                if hasattr(worker, 'registered') and worker.registered:
                    worker._deregister()
            except Exception:
                pass  # Best effort cleanup during shutdown


# Note: For implementing non-Python workers or workers in other languages,
# use the HTTP API directly:
#
# - GET /suggestion - Get parameters to evaluate
# - POST /suggestion - Submit evaluation results
# - GET /status - Get optimization status
#
# See the REST API documentation for more details.
#
# ===== Worker Tracking in Distributed Settings =====
#
# For production distributed systems, worker tracking should be implemented on the server side:
#
# 1. The server should maintain a registry of active workers with:
#    - Unique worker IDs
#    - Last activity timestamp
#    - Worker metadata (type, capabilities, etc.)
#
# 2. Workers should:
#    - Register with the server on startup
#    - Send periodic heartbeats
#    - Gracefully deregister on shutdown
#
# 3. The server should:
#    - Detect stale workers via missed heartbeats
#    - Clean up resources for disconnected workers
#    - Provide accurate worker counts in status APIs
#
# This approach is scalable and works for any worker implementation regardless of
# programming language or network location.