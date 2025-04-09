"""
Contains the LocalWorker class for executing objective functions.
"""

import logging
import threading
import time
from typing import Any, Callable, Dict

import msgspec
import zmq

# Use relative imports for messages and utils within the same package
from .messages import (
    Message,
    ParameterName,
    ObjectiveName,
    Result,
    GetSuggestionRequest,
    SubmitResultRequest,
    HeartbeatRequest,
    GetSuggestionResponse,
    SubmitResultResponse,
    HeartbeatResponse,
)
from .utils import setup_logging


class LocalWorker:
    """Worker process that executes objective function evaluations."""

    def __init__(
        self,
        worker_id: int,
        evaluation_fn: Callable[..., dict[ObjectiveName, float]],
        use_ipc: bool = True,
        heartbeat_interval: float = 2.0,
    ):
        self.worker_id = worker_id
        self.evaluation_fn = evaluation_fn  # Store the evaluation function
        self.base_address = "ipc:///tmp/scheduler" if use_ipc else "tcp://localhost:555"
        self.main_address = (
            f"{self.base_address}.ipc" if use_ipc else f"{self.base_address}5"
        )
        self.heartbeat_address = (
            f"{self.base_address}_heartbeat.ipc" if use_ipc else f"{self.base_address}6"
        )
        # Use the setup_logging function from .utils
        self.logger = setup_logging(f"Worker-{worker_id}")
        self.running = True
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_thread = None
        # Lock for thread safety
        self.lock = threading.Lock()

    def run(self):
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(self.main_address)

        self.logger.info(f"Started worker {self.worker_id} using {self.main_address}")

        # Start heartbeat thread
        self.heartbeat_thread = threading.Thread(target=self.send_heartbeats)
        self.heartbeat_thread.daemon = True
        self.heartbeat_thread.start()

        consecutive_errors = 0
        MAX_CONSECUTIVE_ERRORS = 5

        try:
            while self.running:
                try:
                    with self.lock:  # Use lock to prevent race conditions
                        # Use GetSuggestionRequest from .messages
                        request = GetSuggestionRequest(worker_id=self.worker_id)
                        socket.send(msgspec.json.encode(request))

                        # Set a timeout for receiving the response
                        poller = zmq.Poller()
                        poller.register(socket, zmq.POLLIN)

                        if poller.poll(10000):  # 10 second timeout
                            response_bytes = socket.recv()
                            try:
                                # Use Message union type from .messages
                                response = msgspec.json.decode(response_bytes, type=Message)
                            except msgspec.ValidationError as ve:
                                # Try to decode as a more generic type
                                self.logger.warning(f"Worker {self.worker_id}: Validation error: {ve}, trying generic decode")
                                try:
                                    response_dict = msgspec.json.decode(response_bytes)
                                    if "parameters" in response_dict:
                                        # Use GetSuggestionResponse from .messages
                                        response = GetSuggestionResponse(parameters=response_dict.get("parameters"))
                                    else:
                                        self.logger.error(f"Worker {self.worker_id}: Cannot parse response: {response_dict}")
                                        raise ValueError(f"Cannot parse response: {response_dict}")
                                except Exception as e:
                                    self.logger.error(f"Worker {self.worker_id}: Failed to decode response: {e}")
                                    raise
                        else:
                            self.logger.warning(
                                f"Worker {self.worker_id}: Response timeout, retrying..."
                            )
                            # Recreate socket on timeout
                            socket.close()
                            socket = context.socket(zmq.REQ)
                            socket.setsockopt(zmq.LINGER, 0)
                            socket.connect(self.main_address)
                            consecutive_errors += 1
                            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                                self.logger.error(
                                    f"Worker {self.worker_id}: Too many consecutive errors, shutting down"
                                )
                                self.running = False
                                break
                            continue

                    # Validate response type
                    if not isinstance(response, GetSuggestionResponse):
                        self.logger.error(
                            f"Worker {self.worker_id}: Received unexpected response type: {type(response)}"
                        )
                        consecutive_errors += 1
                        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                            self.logger.error(
                                f"Worker {self.worker_id}: Too many consecutive errors, shutting down"
                            )
                            self.running = False
                            break
                        continue

                    # Reset error counter on successful response
                    consecutive_errors = 0

                    match response:
                        case GetSuggestionResponse(parameters=None):
                            self.logger.info(
                                f"Worker {self.worker_id}: No more parameter suggestions available"
                            )
                            self.running = False
                            break

                        case GetSuggestionResponse(parameters=params):
                            try:
                                result = self.evaluate_parameters(params)

                                with self.lock:  # Use lock again
                                    # Use SubmitResultRequest from .messages
                                    request = SubmitResultRequest(
                                        worker_id=self.worker_id, result=result
                                    )
                                    socket.send(msgspec.json.encode(request))

                                    # Set a timeout for receiving the response
                                    if poller.poll(10000):  # 10 second timeout
                                        response_bytes = socket.recv()
                                        try:
                                            # Use Message union type from .messages
                                            response = msgspec.json.decode(
                                                response_bytes, type=Message
                                            )
                                        except msgspec.ValidationError as ve:
                                            # Try to decode as a more generic type
                                            self.logger.warning(f"Worker {self.worker_id}: Result submission validation error: {ve}, trying generic decode")
                                            try:
                                                response_dict = msgspec.json.decode(response_bytes)
                                                if "success" in response_dict:
                                                    # Use SubmitResultResponse from .messages
                                                    response = SubmitResultResponse(
                                                        success=response_dict.get("success", False),
                                                        is_best=response_dict.get("is_best", False),
                                                        error=response_dict.get("error")
                                                    )
                                                else:
                                                    self.logger.error(f"Worker {self.worker_id}: Cannot parse result response: {response_dict}")
                                                    raise ValueError(f"Cannot parse result response: {response_dict}")
                                            except Exception as e:
                                                self.logger.error(f"Worker {self.worker_id}: Failed to decode result response: {e}")
                                                raise
                                    else:
                                        self.logger.warning(
                                            f"Worker {self.worker_id}: Result submission timeout"
                                        )
                                        # Recreate socket on timeout
                                        socket.close()
                                        socket = context.socket(zmq.REQ)
                                        socket.setsockopt(zmq.LINGER, 0)
                                        socket.connect(self.main_address)
                                        continue

                                # Validate response type
                                if not isinstance(response, SubmitResultResponse):
                                    self.logger.error(
                                        f"Worker {self.worker_id}: Received unexpected result response type: {type(response)}"
                                    )
                                    consecutive_errors += 1
                                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                                        self.logger.error(
                                            f"Worker {self.worker_id}: Too many result submission errors, shutting down"
                                        )
                                        self.running = False
                                        break
                                    continue

                                match response:
                                    case SubmitResultResponse(
                                        success=True, is_best=is_best
                                    ):
                                        if is_best:
                                            self.logger.info(
                                                f"Worker {self.worker_id}: Found new best result!"
                                            )
                                    case SubmitResultResponse(
                                        success=False, error=error
                                    ):
                                        self.logger.error(
                                            f"Error submitting result: {error}"
                                        )
                            except Exception as e:
                                self.logger.error(
                                    f"Error during evaluation: {e}", exc_info=True
                                )
                                consecutive_errors += 1
                                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                                    self.logger.error(
                                        f"Worker {self.worker_id}: Too many evaluation errors, shutting down"
                                    )
                                    self.running = False
                                    break
                                # Wait before retrying to avoid rapid failure loops
                                time.sleep(5)
                except zmq.ZMQError as e:
                    self.logger.error(f"Worker {self.worker_id} ZMQ error: {e}")
                    # Recreate socket on ZMQ errors
                    try:
                        socket.close()
                    except:
                        pass
                    socket = context.socket(zmq.REQ)
                    socket.setsockopt(zmq.LINGER, 0)
                    socket.connect(self.main_address)
                    consecutive_errors += 1
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        self.logger.error(
                            f"Worker {self.worker_id}: Too many ZMQ errors, shutting down"
                        )
                        self.running = False
                        break
                    time.sleep(2)  # Wait before reconnecting
                except Exception as e:
                    self.logger.error(
                        f"Worker {self.worker_id} unexpected error: {e}", exc_info=True
                    )
                    consecutive_errors += 1
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        self.logger.error(
                            f"Worker {self.worker_id}: Too many general errors, shutting down"
                        )
                        self.running = False
                        break
                    time.sleep(2)  # Wait before retrying
        except Exception as e:
            self.logger.error(f"Worker {self.worker_id} error: {e}")
        finally:
            self.running = False
            socket.close()
            context.term()

            # Wait for heartbeat thread to complete
            if self.heartbeat_thread and self.heartbeat_thread.is_alive():
                self.heartbeat_thread.join(timeout=1.0)

            self.logger.info(f"Worker {self.worker_id} shutdown complete")

    def send_heartbeats(self):
        """Send regular heartbeats to the scheduler."""
        # Create a separate socket for heartbeats
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(self.heartbeat_address)

        self.logger.info(
            f"Worker {self.worker_id}: Started heartbeat thread using {self.heartbeat_address}"
        )

        consecutive_errors = 0
        MAX_HEARTBEAT_ERRORS = 10  # More forgiving for heartbeats

        while self.running:
            try:
                time.sleep(self.heartbeat_interval)
                if not self.running:
                    break
                # Use HeartbeatRequest from .messages
                request = HeartbeatRequest(worker_id=self.worker_id)
                socket.send(msgspec.json.encode(request))

                # Set a timeout for receiving the response
                poller = zmq.Poller()
                poller.register(socket, zmq.POLLIN)

                if poller.poll(5000):  # 5 second timeout
                    # Use Message union type from .messages
                    response = msgspec.json.decode(socket.recv(), type=Message)
                    if (
                        not isinstance(response, HeartbeatResponse)
                        or not response.success
                    ):
                        self.logger.warning(
                            f"Worker {self.worker_id}: Received invalid heartbeat response: {type(response)}"
                        )
                        consecutive_errors += 1
                    else:
                        # Reset error counter on successful heartbeat
                        consecutive_errors = 0
                else:
                    self.logger.warning(
                        f"Worker {self.worker_id}: Heartbeat response timeout"
                    )
                    # Recreate socket on timeout
                    socket.close()
                    socket = context.socket(zmq.REQ)
                    socket.setsockopt(zmq.LINGER, 0)
                    socket.connect(self.heartbeat_address)
                    consecutive_errors += 1

                # If we've had too many errors, log a warning but keep trying
                # (we don't stop the heartbeat thread, just log the issue)
                if consecutive_errors >= MAX_HEARTBEAT_ERRORS:
                    self.logger.error(
                        f"Worker {self.worker_id}: Heartbeat experiencing persistent errors ({consecutive_errors}), but will continue trying"
                    )
                    # Don't reset counter, but slow down heartbeats when having issues
                    time.sleep(5)

            except zmq.ZMQError as e:
                self.logger.error(f"Worker {self.worker_id}: Heartbeat ZMQ error: {e}")
                # Always recreate socket on ZMQ errors
                try:
                    socket.close()
                except:
                    pass
                socket = context.socket(zmq.REQ)
                socket.setsockopt(zmq.LINGER, 0)
                socket.connect(self.heartbeat_address)
                consecutive_errors += 1
                time.sleep(2)  # Extra sleep on error
            except Exception as e:
                self.logger.error(
                    f"Worker {self.worker_id}: Heartbeat error: {e}", exc_info=True
                )
                consecutive_errors += 1
                time.sleep(5)  # Wait longer on unexpected errors

        socket.close()
        context.term()
        self.logger.info(f"Worker {self.worker_id}: Heartbeat thread stopped")

    def evaluate_parameters(self, params: dict[ParameterName, Any]) -> Result:
        self.logger.info(f"Worker {self.worker_id} processing parameters {params}")

        try:
            # Call the evaluation function with unpacked parameters
            objectives = self.evaluation_fn(**params)

            # Validate that the return value is a dictionary
            if not isinstance(objectives, dict):
                raise ValueError(
                    f"Evaluation function must return a dict, got {type(objectives)}"
                )
            # Use Result from .messages
            return Result(parameters=params, objectives=objectives)

        except Exception as e:
            self.logger.error(f"Error evaluating function: {e}", exc_info=True)
            # Create a fallback result with error information
            error_message = f"Evaluation failed: {str(e)}"
            objectives = {
                "error": 999.0
            }  # Using a high value for minimization objectives

            # Re-raise to allow the worker to handle this failure
            # The heartbeat mechanism will ensure the scheduler knows this worker is still alive
            raise