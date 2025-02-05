import asyncio
import logging
from datetime import datetime, timezone
from uuid import UUID

import msgspec
import zmq
import zmq.asyncio

from hola.core.coordinator import OptimizationCoordinator
from hola.messages import worker
from hola.messages.client import (
    ClientError,
    ClientMessage,
    InitializeRequest,
    LoadStateRequest,
    PauseRequest,
    ResumeRequest,
    SaveStateRequest,
    StatusRequest,
    UpdateObjectiveConfig,
    UpdateParameterConfig,
)
from hola.messages.errors import ErrorDomain, ErrorSeverity, StructuredError
from hola.messages.server import (
    OK,
    ConfigUpdated,
    SampleResponse,
    ServerError,
    ServerMessage,
    StatusUpdate,
)
from hola.messages.worker import (
    Evaluation,
    HeartbeatPing,
    SampleRequest,
    WorkerDeregistration,
    WorkerError,
    WorkerMessage,
    WorkerRegistration,
)
from hola.server.config import ConnectionConfig

logger = logging.getLogger(__name__)


class OptimizationServer:
    def __init__(self, coordinator: OptimizationCoordinator, config: ConnectionConfig):
        self.coordinator = coordinator
        self.config = config
        self.context = zmq.asyncio.Context()

        # Set up sockets
        self.worker_socket = self.context.socket(zmq.ROUTER)
        self.client_socket = self.context.socket(zmq.ROUTER)

        # Track connected workers and their last activity
        self.workers: dict[UUID, datetime] = {}
        self.clients: set[bytes] = set()

        # Lock for shared data
        self._lock = asyncio.Lock()

    async def start(self):
        """Start the server and begin processing messages."""
        self.worker_socket.bind(self.config.worker_uri)
        self.client_socket.bind(self.config.client_uri)

        logger.info(f"Server started - Worker URI: {self.config.worker_uri}")
        logger.info(f"Server started - Client URI: {self.config.client_uri}")

        await asyncio.gather(
            self._process_worker_messages(),
            self._process_client_messages(),
            self._monitor_workers(),
        )

    async def _process_worker_messages(self):
        """Process incoming worker messages."""
        while True:
            try:
                identity, message = await self.worker_socket.recv_multipart()
                worker_msg = msgspec.json.decode(message, type=WorkerMessage)
                logger.info(f"Received message from worker {identity!r}: {worker_msg}")

                # Get handler based on content type
                match worker_msg:
                    case WorkerRegistration():
                        await self._handle_worker_register(identity, worker_msg)
                    case WorkerDeregistration():
                        await self._handle_worker_deregister(identity, worker_msg)
                    case HeartbeatPing():
                        await self._handle_worker_heartbeat(identity, worker_msg)
                    case SampleRequest():
                        await self._handle_worker_sample(identity, worker_msg)
                    case Evaluation():
                        await self._handle_worker_evaluation(identity, worker_msg)
                    case WorkerError():
                        await self._handle_worker_error(identity, worker_msg)
                    case worker.StatusUpdate():
                        await self._handle_worker_status(identity, worker_msg)
                    case _:
                        logger.warning(f"Unknown worker message type: {worker_msg}")

            except Exception as e:
                logger.error(f"Error processing worker message: {e}")

    async def _process_client_messages(self):
        """Process incoming client messages."""
        while True:
            try:
                identity, message = await self.client_socket.recv_multipart()
                self.clients.add(identity)
                client_msg = msgspec.json.decode(message, type=ClientMessage)
                logger.info(f"Received message from client {identity!r}: {client_msg}")

                match client_msg:
                    case InitializeRequest():
                        await self._handle_client_initialize(identity, client_msg)
                    case StatusRequest():
                        await self._handle_client_status(identity, client_msg)
                    case PauseRequest():
                        await self._handle_client_pause(identity, client_msg)
                    case ResumeRequest():
                        await self._handle_client_resume(identity, client_msg)
                    case UpdateObjectiveConfig():
                        await self._handle_client_update_objective_config(identity, client_msg)
                    case UpdateParameterConfig():
                        await self._handle_client_update_parameter_config(identity, client_msg)
                    case SaveStateRequest():
                        await self._handle_client_save_state(identity, client_msg)
                    case LoadStateRequest():
                        await self._handle_client_load_state(identity, client_msg)
                    case ClientError():
                        await self._handle_client_error(identity, client_msg)
                    case _:
                        logger.warning(f"Unknown client message type: {client_msg}")

            except Exception as e:
                logger.error(f"Error processing client message: {e}")

    async def _monitor_workers(self):
        """Monitor worker health and remove inactive workers."""
        while True:
            await asyncio.sleep(60)  # Check every minute
            current_time = datetime.now(timezone.utc)
            inactive_workers = []

            for identity, last_heartbeat in self.workers.items():
                if (current_time - last_heartbeat).total_seconds() > 180:  # 3 minutes timeout
                    inactive_workers.append(identity)

            for identity in inactive_workers:
                await self._remove_worker(identity)
                logger.warning(f"Removed inactive worker with identity: {identity!r}")

    async def _remove_worker(self, identity: bytes) -> None:
        async with self._lock:
            try:
                # Convert identity to UUID and remove from workers dict if present
                worker_uuid = self._bytes_to_uuid(identity)
                if worker_uuid in self.workers:
                    self.workers.pop(worker_uuid)
                    logger.info(f"Worker {worker_uuid} removed from active workers")
            except ValueError:
                # Handle case where identity bytes cannot be converted to UUID
                logger.warning(
                    f"Could not convert worker identity {identity!r} to UUID during removal"
                )
            except Exception as e:
                # Log any other errors but don't re-raise to avoid disrupting server
                logger.error(f"Error removing worker {identity!r}: {e}")

    async def _send_to_worker(self, identity: bytes, message: ServerMessage):
        """Send a message to a specific worker."""
        try:
            await self.worker_socket.send_multipart([identity, msgspec.json.encode(message)])
        except Exception as e:
            logger.error(f"Error sending message to worker: {e}")

    async def _send_to_client(self, identity: bytes, message: ServerMessage):
        try:
            await self.client_socket.send_multipart([identity, msgspec.json.encode(message)])
        except zmq.ZMQError as e:
            if e.errno == zmq.ETERM:
                logger.debug("Context terminated, ignoring send error")
            else:
                logger.error(f"Client {identity!r} disconnected: {e}")
                self.clients.discard(identity)

    async def _broadcast_to_clients(self, message: ServerMessage):
        """Broadcast a message to all connected clients."""
        encoded_message = msgspec.json.encode(message)
        for client_id in self.clients:
            try:
                await self.client_socket.send_multipart([client_id, encoded_message])
            except Exception as e:
                logger.error(f"Error broadcasting to client {client_id!r}: {e}")

    def _bytes_to_uuid(self, identity: bytes) -> UUID:
        """Convert ZMQ identity bytes to UUID."""
        return UUID(int=int.from_bytes(identity, byteorder="big"))

    def _uuid_to_bytes(self, uuid: UUID) -> bytes:
        """Convert UUID back to bytes for ZMQ identity."""
        return uuid.bytes

    async def _get_worker_uuid(self, identity: bytes) -> UUID | None:
        """Get UUID for a worker identity if it exists."""
        async with self._lock:
            try:
                worker_uuid = self._bytes_to_uuid(identity)
                if worker_uuid in self.workers:
                    return worker_uuid
            except ValueError:
                pass
            return None

    # Worker message handlers
    async def _handle_worker_register(self, identity: bytes, message: WorkerRegistration):
        async with self._lock:
            self.workers[self._bytes_to_uuid(identity)] = message.timestamp
        logger.info(f"Worker registered with identity: {identity!r}")

    async def _handle_worker_deregister(self, identity: bytes, message: WorkerDeregistration):
        await self._remove_worker(identity)
        logger.info(f"Worker deregistered with identity: {identity!r}")

    async def _handle_worker_heartbeat(self, identity: bytes, message: HeartbeatPing):
        async with self._lock:
            self.workers[self._bytes_to_uuid(identity)] = message.timestamp

    async def _handle_worker_sample(self, identity: bytes, message: SampleRequest):
        samples = await self.coordinator.suggest_parameters(message.n_samples)
        await self._send_to_worker(
            identity, SampleResponse(timestamp=datetime.now(timezone.utc), samples=samples or [])
        )

    async def _handle_worker_evaluation(self, identity: bytes, message: Evaluation):
        await self.coordinator.record_evaluation(
            message, str(message.timestamp), self._bytes_to_uuid(identity)
        )
        # Broadcast updated status to clients
        state = await self.coordinator.get_state()
        await self._broadcast_to_clients(
            StatusUpdate(timestamp=datetime.now(timezone.utc), status=state)
        )

    async def _handle_worker_error(self, identity: bytes, message: WorkerError):
        """Handle error messages from workers."""
        logger.error(f"Worker {identity!r} reported error: {message.error}")
        await self.coordinator.record_failed_evaluation(
            str(message.error),
            timestamp=str(message.timestamp),
            worker_id=self._bytes_to_uuid(identity),
        )

    async def _handle_worker_status(self, identity: bytes, message: worker.StatusUpdate):
        """Handle status updates from workers."""
        logger.info(f"Worker {identity!r} status update: {message.status}")
        # Could be used for load balancing or monitoring in the future

    # Client message handlers
    async def _handle_client_initialize(self, identity: bytes, message: InitializeRequest):
        """Handle initialization requests from clients."""
        try:
            async with self._lock:
                self.coordinator = OptimizationCoordinator.from_dict(
                    self.coordinator.hypercube_sampler,
                    message.objectives_config,
                    message.parameters_config,
                )

            await self._send_to_client(identity, OK(timestamp=datetime.now(timezone.utc)))

            # Broadcast initial status to all clients
            state = await self.coordinator.get_state()
            await self._broadcast_to_clients(
                StatusUpdate(timestamp=datetime.now(timezone.utc), status=state)
            )
        except Exception as e:
            logger.error(f"Error initializing optimization: {e}")
            await self._send_to_client(
                identity,
                ServerError(
                    timestamp=datetime.now(timezone.utc),
                    error=StructuredError(
                        code="INITIALIZATION_ERROR",
                        domain=ErrorDomain.CONFIG,
                        severity=ErrorSeverity.CRITICAL,
                        message=str(e),
                    ),
                ),
            )

    async def _handle_client_status(self, identity: bytes, message: StatusRequest):
        state = await self.coordinator.get_state()
        await self._send_to_client(
            identity, StatusUpdate(timestamp=datetime.now(timezone.utc), status=state)
        )

    async def _handle_client_pause(self, identity: bytes, message: PauseRequest):
        await self.coordinator.pause()
        await self._send_to_client(
            identity,
            OK(
                timestamp=datetime.now(timezone.utc),
            ),
        )

    async def _handle_client_resume(self, identity: bytes, message: ResumeRequest):
        await self.coordinator.resume()
        await self._send_to_client(
            identity,
            OK(
                timestamp=datetime.now(timezone.utc),
            ),
        )

    async def _handle_client_update_objective_config(
        self, identity: bytes, message: UpdateObjectiveConfig
    ):
        """Handle updates to objective configuration."""
        try:
            await self.coordinator.update_objective_config(message.new_config)
            await self._send_to_client(
                identity,
                ConfigUpdated(
                    timestamp=datetime.now(timezone.utc),
                    message="Objective configuration updated successfully",
                ),
            )

            # Broadcast updated status to all clients
            state = await self.coordinator.get_state()
            await self._broadcast_to_clients(
                StatusUpdate(timestamp=datetime.now(timezone.utc), status=state)
            )
        except Exception as e:
            logger.error(f"Error updating objective config: {e}")
            await self._send_to_client(
                identity,
                ServerError(
                    timestamp=datetime.now(timezone.utc),
                    error=StructuredError(
                        code="CONFIG_UPDATE_ERROR",
                        domain=ErrorDomain.CONFIG,
                        severity=ErrorSeverity.ERROR,
                        message=str(e),
                    ),
                ),
            )

    async def _handle_client_update_parameter_config(
        self, identity: bytes, message: UpdateParameterConfig
    ):
        """Handle updates to parameter configuration."""
        try:
            await self.coordinator.update_parameter_config(message.new_config)
            await self._send_to_client(
                identity,
                ConfigUpdated(
                    timestamp=datetime.now(timezone.utc),
                    message="Parameter configuration updated successfully",
                ),
            )

            # Broadcast updated status to all clients
            state = await self.coordinator.get_state()
            await self._broadcast_to_clients(
                StatusUpdate(timestamp=datetime.now(timezone.utc), status=state)
            )
        except Exception as e:
            logger.error(f"Error updating parameter config: {e}")
            await self._send_to_client(
                identity,
                ServerError(
                    timestamp=datetime.now(timezone.utc),
                    error=StructuredError(
                        code="CONFIG_UPDATE_ERROR",
                        domain=ErrorDomain.CONFIG,
                        severity=ErrorSeverity.ERROR,
                        message=str(e),
                    ),
                ),
            )

    async def _handle_client_save_state(self, identity: bytes, message: SaveStateRequest):
        """Handle requests to save optimization state."""
        try:
            await self.coordinator.save_state(message.filepath)
            await self._send_to_client(
                identity,
                OK(
                    timestamp=datetime.now(timezone.utc),
                ),
            )
        except Exception as e:
            logger.error(f"Error saving state: {e}")
            await self._send_to_client(
                identity,
                ServerError(
                    timestamp=datetime.now(timezone.utc),
                    error=StructuredError(
                        code="SAVE_STATE_ERROR",
                        domain=ErrorDomain.SYSTEM,
                        severity=ErrorSeverity.ERROR,
                        message=str(e),
                    ),
                ),
            )

    async def _handle_client_load_state(self, identity: bytes, message: LoadStateRequest):
        """Handle requests to load optimization state."""
        try:
            await self.coordinator.load_state(message.filepath)
            await self._send_to_client(
                identity,
                OK(
                    timestamp=datetime.now(timezone.utc),
                ),
            )

            # Broadcast updated status to all clients after loading state
            state = await self.coordinator.get_state()
            await self._broadcast_to_clients(
                StatusUpdate(timestamp=datetime.now(timezone.utc), status=state)
            )
        except Exception as e:
            logger.error(f"Error loading state: {e}")
            await self._send_to_client(
                identity,
                ServerError(
                    timestamp=datetime.now(timezone.utc),
                    error=StructuredError(
                        code="LOAD_STATE_ERROR",
                        domain=ErrorDomain.SYSTEM,
                        severity=ErrorSeverity.ERROR,
                        message=str(e),
                    ),
                ),
            )
