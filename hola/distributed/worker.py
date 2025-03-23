"""
Worker for distributed optimization.

The worker connects to an optimization server, requests jobs, evaluates
objective functions, and submits results back to the server.
"""

import uuid
import json
import logging
import threading
import traceback
from typing import Any, Dict, Optional, Tuple, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod

import zmq
import requests

# Set up logging with higher verbosity
logging.basicConfig(
    level=logging.DEBUG,  # Changed from INFO to DEBUG for more detailed logs
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("hola.distributed.worker")


@dataclass
class WorkerConfig:
    """Configuration for optimization workers."""

    worker_id: Optional[str] = None
    """Unique identifier for the worker, auto-generated if None."""

    heartbeat_interval: float = 30.0
    """Interval in seconds between heartbeat messages."""

    max_consecutive_errors: int = 3
    """Maximum number of consecutive errors before backing off."""

    retry_delay: float = 5.0
    """Delay in seconds between retries after errors."""

    max_retry_delay: float = 300.0
    """Maximum delay between retries (for exponential backoff)."""

    zmq_timeout: int = 5000  # milliseconds
    """Timeout for ZMQ socket operations."""


class Worker(ABC):
    """
    Base class for distributed optimization workers.

    A worker:
    - Connects to a server
    - Receives parameter suggestions
    - Evaluates objective functions
    - Returns evaluation results
    """

    def __init__(
        self,
        objective_function: Callable[[Dict[str, Any]], Dict[str, float]],
        config: WorkerConfig = None
    ):
        """
        Initialize the worker.

        :param objective_function: Function that evaluates parameters and returns objectives
        :type objective_function: Callable[[Dict[str, Any]], Dict[str, float]]
        :param config: Worker configuration, or None for defaults
        :type config: Optional[WorkerConfig]
        """
        self.objective_function = objective_function
        self.config = config or WorkerConfig()

        # Generate worker ID if not provided
        if not self.config.worker_id:
            self.config.worker_id = str(uuid.uuid4())

        # Initialize worker state
        self._running = False
        self._consecutive_errors = 0
        self._heartbeat_thread = None
        self._worker_thread = None
        self._stop_event = threading.Event()

    def start(self):
        """Start the worker."""
        if self._running:
            logger.warning("Worker is already running")
            return

        # Register with server
        try:
            logger.debug(f"Worker {self.config.worker_id} attempting to register with server")
            self._register()
            logger.info(f"Successfully registered worker {self.config.worker_id}")
        except Exception as e:
            logger.error(f"Failed to register worker {self.config.worker_id}: {str(e)}")
            logger.debug(traceback.format_exc())
            raise

        # Start heartbeat thread
        self._heartbeat_thread = threading.Thread(
            target=self._run_heartbeat_loop,
            daemon=True
        )
        self._heartbeat_thread.start()
        logger.debug(f"Started heartbeat thread for worker {self.config.worker_id}")

        # Start worker thread
        self._worker_thread = threading.Thread(
            target=self._run_worker_loop,
            daemon=True
        )
        self._worker_thread.start()
        logger.debug(f"Started main worker thread for worker {self.config.worker_id}")

        self._running = True
        logger.info(f"Worker {self.config.worker_id} started")

    def stop(self):
        """Stop the worker."""
        if not self._running:
            logger.warning("Worker is not running")
            return

        # Signal threads to stop
        logger.debug(f"Stopping worker {self.config.worker_id}")
        self._stop_event.set()

        # Wait for threads to finish
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=5.0)

        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)

        self._running = False
        logger.info(f"Worker {self.config.worker_id} stopped")

    def wait(self):
        """Wait for the worker to complete."""
        if self._worker_thread:
            self._worker_thread.join()

    def _run_heartbeat_loop(self):
        """Run the heartbeat loop."""
        logger.info(f"Started heartbeat loop for worker {self.config.worker_id}")

        while not self._stop_event.is_set():
            try:
                # Send heartbeat
                logger.debug(f"Worker {self.config.worker_id} sending heartbeat")
                self._send_heartbeat()
                logger.debug(f"Worker {self.config.worker_id} heartbeat successful")

                # Reset consecutive errors if successful
                self._consecutive_errors = 0
            except Exception as e:
                logger.error(f"Error sending heartbeat for worker {self.config.worker_id}: {str(e)}")
                logger.debug(traceback.format_exc())
                self._consecutive_errors += 1

            # Sleep until next heartbeat or until stopped
            self._stop_event.wait(self.config.heartbeat_interval)

    def _run_worker_loop(self):
        """Run the worker's main loop."""
        logger.info(f"Started main loop for worker {self.config.worker_id}")

        while not self._stop_event.is_set():
            try:
                # Get job from server
                logger.debug(f"Worker {self.config.worker_id} requesting job from server")
                job_id, parameters = self._get_job()
                logger.info(f"Worker {self.config.worker_id} received job {job_id} with parameters: {parameters}")

                # Evaluate the objective function
                try:
                    logger.debug(f"Worker {self.config.worker_id} evaluating job {job_id}")
                    objectives = self.objective_function(parameters)
                    success = True
                    logger.info(f"Worker {self.config.worker_id} evaluated job {job_id}: {objectives}")
                except Exception as e:
                    logger.error(f"Error evaluating job {job_id} for worker {self.config.worker_id}: {str(e)}")
                    logger.debug(traceback.format_exc())
                    objectives = {}  # Will be populated by the server
                    success = False

                # Submit results
                logger.debug(f"Worker {self.config.worker_id} submitting results for job {job_id}")
                self._submit_result(job_id, objectives, success)
                logger.info(f"Worker {self.config.worker_id} submitted results for job {job_id}")

                # Reset consecutive errors if successful
                self._consecutive_errors = 0
            except Exception as e:
                logger.error(f"Error in worker loop for worker {self.config.worker_id}: {str(e)}")
                logger.debug(traceback.format_exc())
                self._consecutive_errors += 1

                # Implement exponential backoff
                delay = min(
                    self.config.retry_delay * (2 ** min(self._consecutive_errors - 1, 5)),
                    self.config.max_retry_delay
                )
                logger.info(f"Worker {self.config.worker_id} backing off for {delay:.1f} seconds")

                # Sleep until retry or until stopped
                self._stop_event.wait(delay)

    @abstractmethod
    def _register(self) -> None:
        """
        Register the worker with the server.

        :raises: Exception if registration fails
        """
        pass

    @abstractmethod
    def _send_heartbeat(self) -> None:
        """
        Send a heartbeat to the server.

        :raises: Exception if the heartbeat fails
        """
        pass

    @abstractmethod
    def _get_job(self) -> Tuple[str, Dict[str, Any]]:
        """
        Get a job from the server.

        :return: Tuple of (job_id, parameters)
        :rtype: Tuple[str, Dict[str, Any]]
        :raises: Exception if getting the job fails
        """
        pass

    @abstractmethod
    def _submit_result(self, job_id: str, objectives: Dict[str, float], success: bool) -> None:
        """
        Submit job results to the server.

        :param job_id: ID of the completed job
        :type job_id: str
        :param objectives: Objective values achieved
        :type objectives: Dict[str, float]
        :param success: Whether the evaluation was successful
        :type success: bool
        :raises: Exception if submitting results fails
        """
        pass


class LocalWorker(Worker):
    """
    Worker that communicates with the server via ZMQ IPC.

    This worker is designed for local execution on the same machine as the server.
    """

    def __init__(
        self,
        objective_function: Callable[[Dict[str, Any]], Dict[str, float]],
        zmq_ipc_endpoint: str,
        config: WorkerConfig = None
    ):
        """
        Initialize the local worker.

        :param objective_function: Function that evaluates parameters and returns objectives
        :type objective_function: Callable[[Dict[str, Any]], Dict[str, float]]
        :param zmq_ipc_endpoint: ZMQ IPC endpoint for server communication
        :type zmq_ipc_endpoint: str
        :param config: Worker configuration, or None for defaults
        :type config: Optional[WorkerConfig]
        """
        super().__init__(objective_function, config)

        self.zmq_ipc_endpoint = zmq_ipc_endpoint
        self._zmq_context = zmq.Context()
        self._zmq_socket = None
        self._zmq_lock = threading.RLock()
        self._zmq_poller = zmq.Poller()

        logger.debug(f"Initialized LocalWorker {self.config.worker_id} with IPC endpoint {zmq_ipc_endpoint}")

    def _send_zmq_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a message to the server via ZMQ.

        :param message: Message to send
        :type message: Dict[str, Any]
        :return: Server response
        :rtype: Dict[str, Any]
        :raises: Exception if the message fails
        """
        with self._zmq_lock:
            # Create socket if needed
            if self._zmq_socket is None:
                logger.debug(f"Worker {self.config.worker_id} creating new ZMQ socket to {self.zmq_ipc_endpoint}")
                self._zmq_socket = self._zmq_context.socket(zmq.REQ)
                self._zmq_socket.setsockopt(zmq.LINGER, 0)  # Don't wait when closing
                self._zmq_socket.connect(self.zmq_ipc_endpoint)
                self._zmq_poller.register(self._zmq_socket, zmq.POLLIN)
                logger.debug(f"Worker {self.config.worker_id} connected ZMQ socket to {self.zmq_ipc_endpoint}")

            # Send message
            logger.debug(f"Worker {self.config.worker_id} sending message: {message}")
            message_json = json.dumps(message)
            self._zmq_socket.send_string(message_json)
            logger.debug(f"Worker {self.config.worker_id} sent message, waiting for response")

            # Wait for response with timeout
            socks = dict(self._zmq_poller.poll(self.config.zmq_timeout))
            if self._zmq_socket in socks and socks[self._zmq_socket] == zmq.POLLIN:
                response_json = self._zmq_socket.recv_string()
                logger.debug(f"Worker {self.config.worker_id} received response: {response_json[:100]}...")
                response = json.loads(response_json)

                # Check for errors
                if response.get("status") == "error":
                    error = response.get("error", "Unknown error")
                    logger.error(f"Worker {self.config.worker_id} received error from server: {error}")
                    raise RuntimeError(f"Server error: {error}")

                return response
            else:
                # Timeout occurred, recreate socket
                logger.error(f"Worker {self.config.worker_id} timed out waiting for response")
                self._zmq_poller.unregister(self._zmq_socket)
                self._zmq_socket.close()
                self._zmq_socket = None
                raise TimeoutError("Timed out waiting for server response")

    def _register(self) -> None:
        """Register with the server via ZMQ IPC."""
        logger.debug(f"Worker {self.config.worker_id} registering with server")
        response = self._send_zmq_message({
            "action": "register",
            "worker_id": self.config.worker_id
        })

        # Update worker ID if needed
        self.config.worker_id = response["worker_id"]
        logger.debug(f"Worker registered with ID {self.config.worker_id}")

    def _send_heartbeat(self) -> None:
        """Send a heartbeat to the server via ZMQ IPC."""
        self._send_zmq_message({
            "action": "heartbeat",
            "worker_id": self.config.worker_id
        })

    def _get_job(self) -> Tuple[str, Dict[str, Any]]:
        """Get a job from the server via ZMQ IPC."""
        response = self._send_zmq_message({
            "action": "get_job",
            "worker_id": self.config.worker_id
        })

        return response["job_id"], response["parameters"]

    def _submit_result(self, job_id: str, objectives: Dict[str, float], success: bool) -> None:
        """Submit job results to the server via ZMQ IPC."""
        self._send_zmq_message({
            "action": "submit_result",
            "worker_id": self.config.worker_id,
            "job_id": job_id,
            "objectives": objectives,
            "success": success
        })

    def stop(self):
        """Stop the worker and release resources."""
        super().stop()

        with self._zmq_lock:
            if self._zmq_socket:
                logger.debug(f"Closing ZMQ socket for worker {self.config.worker_id}")
                self._zmq_poller.unregister(self._zmq_socket)
                self._zmq_socket.close()
                self._zmq_socket = None


class RemoteWorker(Worker):
    """
    Worker that communicates with the server via HTTP or ZMQ TCP.

    This worker can run on a different machine from the server and connect
    over a network connection.
    """

    def __init__(
        self,
        objective_function: Callable[[Dict[str, Any]], Dict[str, float]],
        server_url: Optional[str] = None,
        zmq_tcp_endpoint: Optional[str] = None,
        config: WorkerConfig = None
    ):
        """
        Initialize the remote worker.

        :param objective_function: Function that evaluates parameters and returns objectives
        :type objective_function: Callable[[Dict[str, Any]], Dict[str, float]]
        :param server_url: HTTP URL for server communication (e.g. "http://server:8080")
        :type server_url: Optional[str]
        :param zmq_tcp_endpoint: ZMQ TCP endpoint for server communication
        :type zmq_tcp_endpoint: Optional[str]
        :param config: Worker configuration, or None for defaults
        :type config: Optional[WorkerConfig]
        :raises ValueError: If neither server_url nor zmq_tcp_endpoint is provided
        """
        super().__init__(objective_function, config)

        # Validate connection options
        if not server_url and not zmq_tcp_endpoint:
            raise ValueError("Either server_url or zmq_tcp_endpoint must be provided")

        # Configure connection method
        self.server_url = server_url
        self.zmq_tcp_endpoint = zmq_tcp_endpoint
        self._use_zmq = bool(zmq_tcp_endpoint)

        # ZMQ resources (if used)
        self._zmq_context = None
        self._zmq_socket = None
        self._zmq_lock = None
        self._zmq_poller = None

        if self._use_zmq:
            self._zmq_context = zmq.Context()
            self._zmq_lock = threading.RLock()
            self._zmq_poller = zmq.Poller()

        logger.debug(f"Initialized RemoteWorker {self.config.worker_id} with " +
                   (f"TCP endpoint {zmq_tcp_endpoint}" if self._use_zmq else f"HTTP URL {server_url}"))

    def _send_zmq_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a message to the server via ZMQ TCP.

        :param message: Message to send
        :type message: Dict[str, Any]
        :return: Server response
        :rtype: Dict[str, Any]
        :raises: Exception if the message fails
        """
        with self._zmq_lock:
            # Create socket if needed
            if self._zmq_socket is None:
                logger.debug(f"Worker {self.config.worker_id} creating new ZMQ socket to {self.zmq_tcp_endpoint}")
                self._zmq_socket = self._zmq_context.socket(zmq.REQ)
                self._zmq_socket.setsockopt(zmq.LINGER, 0)  # Don't wait when closing
                self._zmq_socket.connect(self.zmq_tcp_endpoint)
                self._zmq_poller.register(self._zmq_socket, zmq.POLLIN)
                logger.debug(f"Worker {self.config.worker_id} connected ZMQ socket to {self.zmq_tcp_endpoint}")

            # Send message
            logger.debug(f"Worker {self.config.worker_id} sending message: {message}")
            message_json = json.dumps(message)
            self._zmq_socket.send_string(message_json)
            logger.debug(f"Worker {self.config.worker_id} sent message, waiting for response")

            # Wait for response with timeout
            socks = dict(self._zmq_poller.poll(self.config.zmq_timeout))
            if self._zmq_socket in socks and socks[self._zmq_socket] == zmq.POLLIN:
                response_json = self._zmq_socket.recv_string()
                logger.debug(f"Worker {self.config.worker_id} received response: {response_json[:100]}...")
                response = json.loads(response_json)

                # Check for errors
                if response.get("status") == "error":
                    error = response.get("error", "Unknown error")
                    logger.error(f"Worker {self.config.worker_id} received error from server: {error}")
                    raise RuntimeError(f"Server error: {error}")

                return response
            else:
                # Timeout occurred, recreate socket
                logger.error(f"Worker {self.config.worker_id} timed out waiting for response")
                self._zmq_poller.unregister(self._zmq_socket)
                self._zmq_socket.close()
                self._zmq_socket = None
                raise TimeoutError("Timed out waiting for server response")

    def _send_http_request(self, method: str, endpoint: str, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Send an HTTP request to the server.

        :param method: HTTP method (GET or POST)
        :type method: str
        :param endpoint: API endpoint
        :type endpoint: str
        :param data: Data to send (for POST)
        :type data: Optional[Dict[str, Any]]
        :return: Server response
        :rtype: Dict[str, Any]
        :raises: Exception if the request fails
        """
        url = f"{self.server_url}/api/{endpoint}"
        logger.debug(f"Worker {self.config.worker_id} sending {method} request to {url}")

        if method.upper() == "GET":
            response = requests.get(url, params=data, timeout=self.config.zmq_timeout/1000)
        else:  # POST
            response = requests.post(url, json=data, timeout=self.config.zmq_timeout/1000)

        # Check for HTTP errors
        response.raise_for_status()
        logger.debug(f"Worker {self.config.worker_id} received HTTP {response.status_code} response")

        # Parse JSON response
        result = response.json()

        # Check for API errors
        if result.get("status") == "error":
            error = result.get("error", "Unknown error")
            logger.error(f"Worker {self.config.worker_id} received error from server: {error}")
            raise RuntimeError(f"Server error: {error}")

        return result

    def _register(self) -> None:
        """Register with the server."""
        logger.debug(f"Worker {self.config.worker_id} registering with server")
        if self._use_zmq:
            response = self._send_zmq_message({
                "action": "register",
                "worker_id": self.config.worker_id
            })
        else:
            response = self._send_http_request("POST", "register", {
                "worker_id": self.config.worker_id
            })

        # Update worker ID if needed
        self.config.worker_id = response["worker_id"]
        logger.debug(f"Worker registered with ID {self.config.worker_id}")

    def _send_heartbeat(self) -> None:
        """Send a heartbeat to the server."""
        if self._use_zmq:
            self._send_zmq_message({
                "action": "heartbeat",
                "worker_id": self.config.worker_id
            })
        else:
            self._send_http_request("POST", "heartbeat", {
                "worker_id": self.config.worker_id
            })

    def _get_job(self) -> Tuple[str, Dict[str, Any]]:
        """Get a job from the server."""
        if self._use_zmq:
            response = self._send_zmq_message({
                "action": "get_job",
                "worker_id": self.config.worker_id
            })
        else:
            response = self._send_http_request("GET", "job", {
                "worker_id": self.config.worker_id
            })

        return response["job_id"], response["parameters"]

    def _submit_result(self, job_id: str, objectives: Dict[str, float], success: bool) -> None:
        """Submit job results to the server."""
        if self._use_zmq:
            self._send_zmq_message({
                "action": "submit_result",
                "worker_id": self.config.worker_id,
                "job_id": job_id,
                "objectives": objectives,
                "success": success
            })
        else:
            self._send_http_request("POST", "result", {
                "worker_id": self.config.worker_id,
                "job_id": job_id,
                "objectives": objectives,
                "success": success
            })

    def stop(self):
        """Stop the worker and release resources."""
        super().stop()

        if self._use_zmq and self._zmq_socket:
            with self._zmq_lock:
                logger.debug(f"Closing ZMQ socket for worker {self.config.worker_id}")
                self._zmq_poller.unregister(self._zmq_socket)
                self._zmq_socket.close()
                self._zmq_socket = None