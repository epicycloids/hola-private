"""
Server for distributed optimization.

The server coordinates workers and communicates with the scheduler to
manage optimization jobs across multiple processes or machines.
"""

import time
import uuid
import json
import logging
import threading
import traceback
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, asdict

import zmq
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from hola.distributed.scheduler import OptimizationScheduler

# Set up logging with less verbose default level
logging.basicConfig(
    level=logging.INFO,  # Changed from DEBUG to INFO for normal operation
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("hola.distributed.server")


# Pydantic models for API requests and responses
class WorkerRegisterRequest(BaseModel):
    worker_id: Optional[str] = None


class WorkerHeartbeatRequest(BaseModel):
    worker_id: str


class JobResultRequest(BaseModel):
    worker_id: str
    job_id: str
    objectives: Dict[str, float]
    success: bool = True


class ApiResponse(BaseModel):
    status: str
    error: Optional[str] = None
    worker_id: Optional[str] = None
    job_id: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    best_trial: Optional[Dict[str, Any]] = None


class StatusResponse(BaseModel):
    status: str
    active_jobs: int
    active_workers: int
    total_trials: int
    best_trial: Optional[Dict[str, Any]] = None


@dataclass
class ServerConfig:
    """Configuration for OptimizationServer."""

    zmq_ipc_endpoint: Optional[str] = None
    """ZMQ IPC endpoint for local workers."""

    zmq_tcp_endpoint: Optional[str] = None
    """ZMQ TCP endpoint for remote workers with ZMQ support."""

    http_host: str = "0.0.0.0"
    """HTTP server host."""

    http_port: int = 8080
    """HTTP server port."""

    job_cleanup_interval: float = 60.0
    """Time in seconds between job cleanup operations."""

    max_job_age: float = 3600.0
    """Maximum age in seconds before a job is considered stalled."""

    zmq_poll_timeout: int = 1000  # milliseconds
    """Timeout for ZMQ polling in milliseconds."""


class OptimizationServer:
    """
    Server for distributed optimization.

    The server handles:
    - Communication with workers over ZMQ (IPC/TCP) and HTTP
    - Relaying job requests and results to/from the scheduler
    - Tracking active workers and jobs
    - Cleaning up stalled jobs
    """

    def __init__(self, scheduler: OptimizationScheduler, config: ServerConfig = None):
        """
        Initialize the optimization server.

        :param scheduler: The scheduler for optimization jobs
        :type scheduler: OptimizationScheduler
        :param config: Server configuration, or None for defaults
        :type config: Optional[ServerConfig]
        """
        self.scheduler = scheduler
        self.config = config or ServerConfig()

        # Initialize server state
        self._running = False
        self._lock = threading.RLock()
        self._worker_heartbeats = {}  # worker_id -> last heartbeat time

        # Set up ZMQ context
        self._zmq_context = zmq.Context()
        self._zmq_ipc_socket = None
        self._zmq_tcp_socket = None
        self._zmq_poller = zmq.Poller()

        # Set up FastAPI app
        self._http_app = self._create_http_app()
        self._http_server = None

        # Set up job cleanup thread
        self._cleanup_thread = None
        self._cleanup_event = threading.Event()

        logger.debug(f"Initialized OptimizationServer with scheduler {scheduler} and config {config}")

    def start(self):
        """Start the optimization server."""
        with self._lock:
            if self._running:
                logger.warning("Server is already running")
                return

            # Record start time for uptime calculation
            self._start_time = time.time()

            # Set running flag first so threads don't exit immediately
            self._running = True

            # Start ZMQ IPC socket if configured
            if self.config.zmq_ipc_endpoint:
                logger.debug(f"Creating ZMQ IPC socket at {self.config.zmq_ipc_endpoint}")
                self._zmq_ipc_socket = self._zmq_context.socket(zmq.REP)
                self._zmq_ipc_socket.bind(self.config.zmq_ipc_endpoint)
                self._zmq_poller.register(self._zmq_ipc_socket, zmq.POLLIN)
                logger.info(f"Started ZMQ IPC server at {self.config.zmq_ipc_endpoint}")

            # Start ZMQ TCP socket if configured
            if self.config.zmq_tcp_endpoint:
                logger.debug(f"Creating ZMQ TCP socket at {self.config.zmq_tcp_endpoint}")
                self._zmq_tcp_socket = self._zmq_context.socket(zmq.REP)
                self._zmq_tcp_socket.bind(self.config.zmq_tcp_endpoint)
                self._zmq_poller.register(self._zmq_tcp_socket, zmq.POLLIN)
                logger.info(f"Started ZMQ TCP server at {self.config.zmq_tcp_endpoint}")

            # Start ZMQ worker thread if either socket is configured
            if self._zmq_ipc_socket or self._zmq_tcp_socket:
                threading.Thread(
                    target=self._run_zmq_loop,
                    daemon=True
                ).start()
                logger.info("Started ZMQ worker thread")

            # Start FastAPI HTTP server
            if self.config.http_port:
                logger.debug(f"Creating FastAPI HTTP server at {self.config.http_host}:{self.config.http_port}")
                http_thread = threading.Thread(
                    target=self._start_http_server,
                    daemon=True
                )
                http_thread.start()
                logger.info(f"Started HTTP server at http://{self.config.http_host}:{self.config.http_port}")

            # Start job cleanup thread
            logger.debug("Creating job cleanup thread")
            self._cleanup_thread = threading.Thread(
                target=self._run_job_cleanup,
                daemon=True
            )
            self._cleanup_thread.start()

            logger.info("Optimization server started")

    def stop(self):
        """Stop the optimization server."""
        with self._lock:
            if not self._running:
                logger.warning("Server is not running")
                return

            # Signal cleanup thread to stop
            if self._cleanup_thread:
                logger.debug("Signaling cleanup thread to stop")
                self._cleanup_event.set()

            # Stop the running flag to terminate threads
            self._running = False

            # Close ZMQ sockets
            if self._zmq_ipc_socket:
                logger.debug("Closing ZMQ IPC socket")
                self._zmq_poller.unregister(self._zmq_ipc_socket)
                self._zmq_ipc_socket.close()
                logger.info("Closed ZMQ IPC socket")

            if self._zmq_tcp_socket:
                logger.debug("Closing ZMQ TCP socket")
                self._zmq_poller.unregister(self._zmq_tcp_socket)
                self._zmq_tcp_socket.close()
                logger.info("Closed ZMQ TCP socket")

            # HTTP server will be killed when the process exits

            logger.info("Optimization server stopped")

    def _start_http_server(self):
        """Start the FastAPI HTTP server."""
        logger.debug(f"Starting Uvicorn server at {self.config.http_host}:{self.config.http_port}")
        uvicorn.run(
            self._http_app,
            host=self.config.http_host,
            port=self.config.http_port,
            log_level="info",
        )

    def _run_zmq_loop(self):
        """
        Run the ZMQ message loop using a poller to handle both IPC and TCP sockets.
        """
        logger.info("Started ZMQ worker loop")

        while self._running:
            try:
                # Poll for messages with timeout
                sockets = dict(self._zmq_poller.poll(self.config.zmq_poll_timeout))

                # Process IPC messages
                if self._zmq_ipc_socket in sockets and sockets[self._zmq_ipc_socket] == zmq.POLLIN:
                    self._process_zmq_message(self._zmq_ipc_socket, "IPC")

                # Process TCP messages
                if self._zmq_tcp_socket in sockets and sockets[self._zmq_tcp_socket] == zmq.POLLIN:
                    self._process_zmq_message(self._zmq_tcp_socket, "TCP")

            except Exception as e:
                logger.error(f"Error in ZMQ loop: {str(e)}")
                logger.debug(traceback.format_exc())
                # Continue the loop, don't exit on error

    def _process_zmq_message(self, socket: zmq.Socket, socket_type: str):
        """
        Process a message from a ZMQ socket.

        :param socket: ZMQ socket that has a message
        :type socket: zmq.Socket
        :param socket_type: Socket type (IPC or TCP) for logging
        :type socket_type: str
        """
        try:
            # Receive message
            logger.debug(f"ZMQ {socket_type} socket receiving message")
            message_json = socket.recv_string(flags=zmq.NOBLOCK)
            logger.debug(f"ZMQ {socket_type} socket received message: {message_json[:200]}...")

            try:
                message = json.loads(message_json)
                logger.debug(f"Parsed message: {message}")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON message: {str(e)}")
                logger.debug(f"Raw message: {message_json}")
                socket.send_string(json.dumps({
                    "status": "error",
                    "error": f"Invalid JSON: {str(e)}"
                }))
                return

            # Process message
            logger.debug(f"Processing message: {message}")
            response = self._handle_zmq_message(message, socket_type)
            logger.debug(f"Generated response: {response}")

            # Send response
            response_json = json.dumps(response)
            logger.debug(f"Sending response: {response_json[:200]}...")
            socket.send_string(response_json)
            logger.debug(f"Response sent successfully")

        except zmq.ZMQError as e:
            logger.error(f"ZMQ error in {socket_type} socket: {str(e)}")
            logger.debug(traceback.format_exc())
            # Don't try to send a response if there was a ZMQ error
        except Exception as e:
            logger.error(f"Error processing ZMQ {socket_type} message: {str(e)}")
            logger.debug(traceback.format_exc())

            # Try to send error response
            try:
                socket.send_string(json.dumps({
                    "status": "error",
                    "error": str(e)
                }))
            except Exception:
                logger.error("Failed to send error response")
                pass  # Socket might be closed

    def _handle_zmq_message(self, message: Dict[str, Any], socket_type: str) -> Dict[str, Any]:
        """
        Handle a ZMQ message from a worker.

        :param message: Message from worker
        :type message: Dict[str, Any]
        :param socket_type: Socket type (IPC or TCP) for logging
        :type socket_type: str
        :return: Response to send back to worker
        :rtype: Dict[str, Any]
        """
        action = message.get("action")
        worker_id = message.get("worker_id")

        logger.debug(f"Handling {action} message from worker {worker_id} via {socket_type}")

        # Handle monitoring-related actions
        if action == "ping":
            # Simple ping to check server availability
            return {
                "status": "ok",
                "server_time": time.time()
            }

        elif action == "get_server_info":
            # Return server information for monitoring
            return {
                "status": "ok",
                "data": self._get_server_info()
            }

        elif action == "get_workers_info":
            # Return worker information for monitoring
            return {
                "status": "ok",
                "data": self._get_workers_info()
            }

        elif action == "get_jobs_info":
            # Return active jobs information for monitoring
            return {
                "status": "ok",
                "data": self._get_jobs_info()
            }

        elif action == "get_coordinator_stats":
            # Return optimization coordinator stats for monitoring
            return {
                "status": "ok",
                "data": self._get_coordinator_stats()
            }

        elif action == "get_trials_data":
            # Return trials data for monitoring
            limit = message.get("limit", 100)
            return {
                "status": "ok",
                "data": self._get_trials_data(limit)
            }

        # Handle existing worker-related actions
        elif action == "register":
            # Register a new worker
            if not worker_id:
                worker_id = str(uuid.uuid4())
                logger.debug(f"Generated new worker ID: {worker_id}")

            self._worker_heartbeats[worker_id] = time.time()
            logger.info(f"Registered worker {worker_id} via {socket_type}")

            return {
                "status": "ok",
                "worker_id": worker_id
            }

        elif action == "heartbeat":
            # Update worker heartbeat
            if not worker_id:
                logger.warning("Received heartbeat with missing worker_id")
                return {
                    "status": "error",
                    "error": "Missing worker_id"
                }

            self._worker_heartbeats[worker_id] = time.time()
            logger.debug(f"Updated heartbeat for worker {worker_id}")

            return {
                "status": "ok"
            }

        elif action == "get_job":
            # Get a job for the worker
            if not worker_id:
                logger.warning("Received get_job request with missing worker_id")
                return {
                    "status": "error",
                    "error": "Missing worker_id"
                }

            # Update heartbeat
            self._worker_heartbeats[worker_id] = time.time()

            try:
                # Get job from scheduler
                logger.debug(f"Requesting job from scheduler for worker {worker_id}")
                job_id, parameters = self.scheduler.suggest_parameters(worker_id)
                logger.info(f"Assigned job {job_id} to worker {worker_id} with parameters: {parameters}")

                return {
                    "status": "ok",
                    "job_id": job_id,
                    "parameters": parameters
                }
            except Exception as e:
                logger.error(f"Error getting job for worker {worker_id}: {str(e)}")
                logger.debug(traceback.format_exc())
                return {
                    "status": "error",
                    "error": str(e)
                }

        elif action == "submit_result":
            # Record job results
            job_id = message.get("job_id")
            objectives = message.get("objectives")
            success = message.get("success", True)

            if not worker_id or not job_id or not objectives:
                logger.warning(f"Received submit_result with missing fields: worker_id={worker_id}, job_id={job_id}, objectives={objectives is not None}")
                return {
                    "status": "error",
                    "error": "Missing required fields: worker_id, job_id, or objectives"
                }

            # Update heartbeat
            self._worker_heartbeats[worker_id] = time.time()

            try:
                # Record results with scheduler
                logger.debug(f"Recording evaluation for job {job_id} from worker {worker_id}: {objectives}")
                best_trial = self.scheduler.record_evaluation(
                    job_id=job_id,
                    worker_id=worker_id,
                    objectives=objectives,
                    success=success
                )

                if best_trial:
                    logger.info(f"Recorded successful evaluation for job {job_id} from worker {worker_id}, best trial: {best_trial.trial_id}")
                else:
                    logger.info(f"Recorded successful evaluation for job {job_id} from worker {worker_id}, no best trial yet")

                return {
                    "status": "ok",
                    "best_trial": {
                        "parameters": best_trial.parameters if best_trial else None,
                        "objectives": best_trial.objectives if best_trial else None
                    }
                }
            except Exception as e:
                logger.error(f"Error recording results for job {job_id} from worker {worker_id}: {str(e)}")
                logger.debug(traceback.format_exc())
                return {
                    "status": "error",
                    "error": str(e)
                }

        else:
            # Unknown action
            logger.warning(f"Received unknown action '{action}' from worker {worker_id}")
            return {
                "status": "error",
                "error": f"Unknown action: {action}"
            }

    def _create_http_app(self) -> FastAPI:
        """
        Create the FastAPI app for HTTP API.

        :return: Configured FastAPI app
        :rtype: FastAPI
        """
        app = FastAPI(
            title="HOLA Optimization API",
            description="API for distributed hyperparameter optimization",
            version="1.0.0",
        )

        @app.post("/api/register", response_model=ApiResponse)
        async def register(request: WorkerRegisterRequest):
            worker_id = request.worker_id
            logger.debug(f"HTTP API: register request for worker {worker_id}")

            if not worker_id:
                worker_id = str(uuid.uuid4())
                logger.debug(f"Generated new worker ID: {worker_id}")

            self._worker_heartbeats[worker_id] = time.time()
            logger.info(f"Registered worker {worker_id} via HTTP")

            return ApiResponse(
                status="ok",
                worker_id=worker_id
            )

        @app.post("/api/heartbeat", response_model=ApiResponse)
        async def heartbeat(request: WorkerHeartbeatRequest):
            worker_id = request.worker_id
            logger.debug(f"HTTP API: heartbeat from worker {worker_id}")

            self._worker_heartbeats[worker_id] = time.time()
            logger.debug(f"Updated heartbeat for worker {worker_id}")

            return ApiResponse(
                status="ok"
            )

        @app.get("/api/job", response_model=ApiResponse)
        async def get_job(worker_id: str):
            logger.debug(f"HTTP API: get_job request from worker {worker_id}")

            if not worker_id:
                logger.warning("Received HTTP get_job request with missing worker_id")
                raise HTTPException(status_code=400, detail="Missing worker_id")

            # Update heartbeat
            self._worker_heartbeats[worker_id] = time.time()

            try:
                # Get job from scheduler
                logger.debug(f"Requesting job from scheduler for worker {worker_id}")
                job_id, parameters = self.scheduler.suggest_parameters(worker_id)
                logger.info(f"Assigned job {job_id} to worker {worker_id} with parameters: {parameters}")

                return ApiResponse(
                    status="ok",
                    job_id=job_id,
                    parameters=parameters
                )
            except Exception as e:
                logger.error(f"Error getting job for worker {worker_id}: {str(e)}")
                logger.debug(traceback.format_exc())
                raise HTTPException(status_code=500, detail=str(e))

        @app.post("/api/result", response_model=ApiResponse)
        async def submit_result(request: JobResultRequest):
            worker_id = request.worker_id
            job_id = request.job_id
            objectives = request.objectives
            success = request.success

            logger.debug(f"HTTP API: submit_result for job {job_id} from worker {worker_id}: {objectives}")

            # Update heartbeat
            self._worker_heartbeats[worker_id] = time.time()

            try:
                # Record results with scheduler
                logger.debug(f"Recording evaluation for job {job_id} from worker {worker_id}")
                best_trial = self.scheduler.record_evaluation(
                    job_id=job_id,
                    worker_id=worker_id,
                    objectives=objectives,
                    success=success
                )

                if best_trial:
                    logger.info(f"Recorded successful evaluation for job {job_id} from worker {worker_id}, best trial: {best_trial.trial_id}")
                else:
                    logger.info(f"Recorded successful evaluation for job {job_id} from worker {worker_id}, no best trial yet")

                return ApiResponse(
                    status="ok",
                    best_trial={
                        "parameters": best_trial.parameters if best_trial else None,
                        "objectives": best_trial.objectives if best_trial else None
                    }
                )
            except Exception as e:
                logger.error(f"Error recording results for job {job_id} from worker {worker_id}: {str(e)}")
                logger.debug(traceback.format_exc())
                raise HTTPException(status_code=500, detail=str(e))

        @app.get("/api/status", response_model=StatusResponse)
        async def get_status():
            logger.debug("HTTP API: status request")

            # Get basic status information
            active_jobs = self.scheduler.get_active_jobs()
            active_workers = self.scheduler.get_active_workers()
            total_trials = self.scheduler.coordinator.get_total_evaluations()
            best_trial = self.scheduler.coordinator.get_best_trial()

            logger.debug(f"Status: {len(active_jobs)} active jobs, {len(active_workers)} active workers, {total_trials} total trials")

            return StatusResponse(
                status="ok",
                active_jobs=len(active_jobs),
                active_workers=len(active_workers),
                total_trials=total_trials,
                best_trial={
                    "parameters": best_trial.parameters if best_trial else None,
                    "objectives": best_trial.objectives if best_trial else None
                } if best_trial else None
            )

        # Log all configured routes
        logger.debug(f"HTTP API routes configured: {[route.path for route in app.routes]}")

        return app

    def _run_job_cleanup(self):
        """Run the job cleanup loop."""
        logger.info("Started job cleanup thread")

        while not self._cleanup_event.is_set():
            try:
                # Reset stalled jobs
                logger.debug(f"Running job cleanup with max age {self.config.max_job_age}")
                reset_count = self.scheduler.reset_stalled_jobs(self.config.max_job_age)
                if reset_count > 0:
                    logger.info(f"Reset {reset_count} stalled jobs")
                else:
                    logger.debug("No stalled jobs found")

                # Wait for next cleanup interval or until stopped
                logger.debug(f"Waiting {self.config.job_cleanup_interval} seconds until next cleanup")
                self._cleanup_event.wait(self.config.job_cleanup_interval)
            except Exception as e:
                logger.error(f"Error in job cleanup: {str(e)}")
                logger.debug(traceback.format_exc())

                # Wait a bit before retrying
                time.sleep(10.0)

    def _get_server_info(self) -> Dict[str, Any]:
        """
        Get information about the server for monitoring.

        :return: Server information
        :rtype: Dict[str, Any]
        """
        with self._lock:
            # Calculate server uptime
            server_info = {
                "status": "running" if self._running else "stopped",
                "uptime": time.time() - getattr(self, "_start_time", time.time()),
                "active_workers": len(self._worker_heartbeats),
                "active_jobs": self.scheduler.get_active_job_count(),
                "max_evaluations": getattr(self.scheduler, "max_evaluations", 0),
                "zmq_ipc_endpoint": self.config.zmq_ipc_endpoint,
                "zmq_tcp_endpoint": self.config.zmq_tcp_endpoint,
                "http_enabled": self.config.http_port is not None,
            }
            return server_info

    def _get_workers_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all active workers.

        :return: Dictionary of worker information
        :rtype: Dict[str, Dict[str, Any]]
        """
        with self._lock:
            current_time = time.time()
            workers_info = {}

            for worker_id, last_heartbeat in self._worker_heartbeats.items():
                # Get worker statistics
                worker_stats = self.scheduler.get_worker_stats(worker_id)

                workers_info[worker_id] = {
                    "status": "active" if (current_time - last_heartbeat) < 30 else "inactive",
                    "last_heartbeat": last_heartbeat,
                    "heartbeat_age": current_time - last_heartbeat,
                    "total_jobs": worker_stats.get("total_jobs", 0),
                    "successful_jobs": worker_stats.get("successful_jobs", 0),
                    "failed_jobs": worker_stats.get("failed_jobs", 0),
                    "current_job": self.scheduler.get_active_job_for_worker(worker_id)
                }

            return workers_info

    def _get_jobs_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all active jobs.

        :return: Dictionary of job information
        :rtype: Dict[str, Dict[str, Any]]
        """
        with self._lock:
            return self.scheduler.get_all_active_jobs()

    def _get_coordinator_stats(self) -> Dict[str, Any]:
        """
        Get statistics from the optimization coordinator.

        :return: Coordinator statistics
        :rtype: Dict[str, Any]
        """
        coordinator = self.scheduler.coordinator
        stats = {
            "total_trials": coordinator.get_total_evaluations(),
            "feasible_trials": coordinator.get_feasible_count(),
            "ranked_trials": coordinator.get_ranked_count(),
            "infeasible_trials": coordinator.get_infeasible_count(),
        }

        # Get best trial if available
        best_trial = coordinator.get_best_trial()
        if best_trial:
            stats["best_trial"] = {
                "trial_id": best_trial.trial_id,
                "parameters": best_trial.parameters,
                "objectives": best_trial.objectives
            }

        return stats

    def _get_trials_data(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get trial data for visualization and monitoring.

        :param limit: Maximum number of trials to return
        :type limit: int
        :return: List of trial data
        :rtype: List[Dict[str, Any]]
        """
        coordinator = self.scheduler.coordinator

        # Get dataframe of all trials
        df = coordinator.get_all_trials_dataframe()
        if df.empty:
            return []

        # Limit the number of rows
        if len(df) > limit:
            df = df.tail(limit)

        # Convert dataframe to list of dictionaries
        return df.to_dict('records')

    def set_verbose_logging(self, verbose: bool = False):
        """
        Set the verbosity level of the server logging.

        :param verbose: Whether to use verbose (DEBUG) logging
        :type verbose: bool
        """
        if verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)