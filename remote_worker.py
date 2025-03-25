import argparse
import logging
import os
import random
import sys
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict

import msgspec
import zmq

# Message types needed for communication
class Result(msgspec.Struct):
    """Complete result of a trial evaluation"""
    parameters: dict[str, Any]
    objectives: dict[str, float]

class GetSuggestionRequest(msgspec.Struct, tag="get_suggestion", tag_field="tag"):
    worker_id: int

class SubmitResultRequest(msgspec.Struct, tag="submit_result", tag_field="tag"):
    worker_id: int
    result: Result

class HeartbeatRequest(msgspec.Struct, tag="heartbeat", tag_field="tag"):
    worker_id: int

class GetSuggestionResponse(msgspec.Struct, tag="suggestion_response", tag_field="tag"):
    parameters: dict[str, Any] | None

class SubmitResultResponse(msgspec.Struct, tag="result_response", tag_field="tag"):
    success: bool
    is_best: bool = False
    error: str | None = None

class HeartbeatResponse(msgspec.Struct, tag="heartbeat_response", tag_field="tag"):
    success: bool

# Union type for all messages
Message = (
    GetSuggestionRequest
    | SubmitResultRequest
    | HeartbeatRequest
    | GetSuggestionResponse
    | SubmitResultResponse
    | HeartbeatResponse
)

def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """Configure logging for a component with console and file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    # Create file handler in the logs directory
    log_file_path = os.path.join(logs_dir, f'remote_worker_{name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

class RemoteWorker:
    """Worker that connects to a remote scheduler via ZMQ TCP sockets."""

    def __init__(
        self,
        worker_id: int,
        evaluation_fn: Callable[..., dict[str, float]],
        host: str = "localhost",
        main_port: int = 5555,
        heartbeat_port: int = 5556,
        heartbeat_interval: float = 2.0,
    ):
        self.worker_id = worker_id
        self.evaluation_fn = evaluation_fn
        self.main_address = f"tcp://{host}:{main_port}"
        self.heartbeat_address = f"tcp://{host}:{heartbeat_port}"
        self.logger = setup_logging(f"RemoteWorker-{worker_id}")
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
                        request = GetSuggestionRequest(worker_id=self.worker_id)
                        socket.send(msgspec.json.encode(request))

                        # Set a timeout for receiving the response
                        poller = zmq.Poller()
                        poller.register(socket, zmq.POLLIN)

                        if poller.poll(10000):  # 10 second timeout
                            response_bytes = socket.recv()
                            try:
                                response = msgspec.json.decode(response_bytes, type=Message)
                            except msgspec.ValidationError as ve:
                                # Try to decode as a more generic type
                                self.logger.warning(f"Worker {self.worker_id}: Validation error: {ve}, trying generic decode")
                                try:
                                    response_dict = msgspec.json.decode(response_bytes)
                                    if "parameters" in response_dict:
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
                                    request = SubmitResultRequest(
                                        worker_id=self.worker_id, result=result
                                    )
                                    socket.send(msgspec.json.encode(request))

                                    # Set a timeout for receiving the response
                                    if poller.poll(10000):  # 10 second timeout
                                        response_bytes = socket.recv()
                                        try:
                                            response = msgspec.json.decode(
                                                response_bytes, type=Message
                                            )
                                        except msgspec.ValidationError as ve:
                                            # Try to decode as a more generic type
                                            self.logger.warning(f"Worker {self.worker_id}: Result submission validation error: {ve}, trying generic decode")
                                            try:
                                                response_dict = msgspec.json.decode(response_bytes)
                                                if "success" in response_dict:
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

                request = HeartbeatRequest(worker_id=self.worker_id)
                socket.send(msgspec.json.encode(request))

                # Set a timeout for receiving the response
                poller = zmq.Poller()
                poller.register(socket, zmq.POLLIN)

                if poller.poll(5000):  # 5 second timeout
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

    def evaluate_parameters(self, params: dict[str, Any]) -> Result:
        self.logger.info(f"Worker {self.worker_id} processing parameters {params}")

        try:
            # Call the evaluation function with unpacked parameters
            objectives = self.evaluation_fn(**params)

            # Validate that the return value is a dictionary
            if not isinstance(objectives, dict):
                raise ValueError(
                    f"Evaluation function must return a dict, got {type(objectives)}"
                )

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

# Demo evaluation function with 3 objectives (same as in test.py)
def example_evaluation_fn(x: float, y: float) -> dict[str, float]:
    # Add a small random delay to simulate computation time
    time.sleep(random.uniform(0.5, 1.5))

    # Calculate objectives
    # objective1: Higher is better (maximize), peak at x=0, y=0
    import math
    objective1 = math.exp(-(x**2 + y**2)/10)

    # objective2: Lower is better (minimize), valley along y=x
    objective2 = (x - y)**2

    # objective3: Lower is better (minimize), valley at origin
    objective3 = math.sqrt(x**2 + y**2)

    # Return objectives
    return {
        "objective1": objective1,
        "objective2": objective2,
        "objective3": objective3
    }

def main():
    parser = argparse.ArgumentParser(description='Remote worker for optimization server')
    parser.add_argument('--host', type=str, default='localhost',
                        help='Server hostname or IP address')
    parser.add_argument('--port', type=int, default=5555,
                        help='Server main port number (default: 5555)')
    parser.add_argument('--heartbeat-port', type=int, default=5556,
                        help='Server heartbeat port number (default: 5556)')
    parser.add_argument('--worker-id', type=int, default=None,
                        help='Worker ID (default: random integer)')
    args = parser.parse_args()

    # Generate a random worker ID if not provided
    worker_id = args.worker_id if args.worker_id is not None else random.randint(1000, 9999)

    print(f"Starting remote worker {worker_id} connecting to {args.host}:{args.port}")

    # Create and run the remote worker
    worker = RemoteWorker(
        worker_id=worker_id,
        evaluation_fn=example_evaluation_fn,
        host=args.host,
        main_port=args.port,
        heartbeat_port=args.heartbeat_port
    )
    worker.run()

if __name__ == "__main__":
    main()