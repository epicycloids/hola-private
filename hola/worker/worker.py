import asyncio
import logging
from datetime import datetime, timezone
from typing import Callable

import msgspec
import zmq
import zmq.asyncio

from hola.messages.errors import ErrorDomain, ErrorSeverity, StructuredError
from hola.messages.server import SampleResponse, ServerMessage
from hola.messages.worker import (
    Evaluation,
    HeartbeatPing,
    SampleRequest,
    WorkerDeregistration,
    WorkerError,
    WorkerRegistration,
)
from hola.server.config import ConnectionConfig

logger = logging.getLogger(__name__)


class OptimizationWorker:
    def __init__(self, config: ConnectionConfig, evaluation_fn: Callable[..., dict[str, float]]):
        self.config = config
        self.evaluation_fn = evaluation_fn
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.DEALER)
        self.is_running = False

    async def start(self):
        """Start the worker and begin processing."""
        self.socket.connect(self.config.worker_uri)
        self.is_running = True

        # Register with server
        await self._send_registration()

        # Start main tasks
        await asyncio.gather(self._process_messages(), self._send_heartbeats())

    async def stop(self):
        """Stop the worker gracefully."""
        self.is_running = False
        await self._send_deregistration()
        self.socket.close()
        self.context.term()

    async def _send_registration(self):
        """Send registration message to server."""
        message = WorkerRegistration(timestamp=datetime.now(timezone.utc), capabilities={})
        await self.socket.send(msgspec.json.encode(message))

        # Immediately request first sample after registration
        sample_request = SampleRequest(timestamp=datetime.now(timezone.utc), n_samples=1)
        await self.socket.send(msgspec.json.encode(sample_request))

    async def _send_deregistration(self):
        """Send deregistration message to server."""
        message = WorkerDeregistration(timestamp=datetime.now(timezone.utc))
        await self.socket.send(msgspec.json.encode(message))

    async def _send_heartbeats(self):
        """Periodically send heartbeat messages."""
        while self.is_running:
            now = datetime.now(timezone.utc)
            message = HeartbeatPing(timestamp=now, last_active=now, current_load=0)
            await self.socket.send(msgspec.json.encode(message))
            await asyncio.sleep(30)  # Heartbeat every 30 seconds

    async def _process_messages(self):
        """Process incoming messages from the server."""
        while self.is_running:
            try:
                message = await self.socket.recv()
                server_msg = msgspec.json.decode(message, type=ServerMessage)
                logger.debug(f"Received server message: {server_msg}")

                match server_msg:
                    case SampleResponse(samples=samples):
                        logger.info(f"Received {len(samples)} samples for evaluation")
                        # Process parameter samples and evaluate
                        for params in samples:
                            try:
                                logger.debug(f"Evaluating parameters: {params}")
                                objectives = self.evaluation_fn(**params)
                                logger.debug(f"Evaluation result: {objectives}")

                                eval_msg = Evaluation(
                                    timestamp=datetime.now(timezone.utc),
                                    parameters=params,
                                    objectives=objectives,
                                )
                                await self.socket.send(msgspec.json.encode(eval_msg))
                                logger.debug("Sent evaluation result")

                                # Request next sample after evaluation
                                sample_req = SampleRequest(
                                    timestamp=datetime.now(timezone.utc), n_samples=1
                                )
                                await self.socket.send(msgspec.json.encode(sample_req))
                                logger.debug("Requested next sample")

                            except Exception as e:
                                logger.error(f"Error evaluating parameters: {e}")
                                error_msg = WorkerError(
                                    timestamp=datetime.now(timezone.utc),
                                    error=StructuredError(
                                        code="EVALUATION_ERROR",
                                        domain=ErrorDomain.SYSTEM,
                                        severity=ErrorSeverity.ERROR,
                                        message=str(e),
                                    ),
                                )
                                await self.socket.send(msgspec.json.encode(error_msg))
                    case _:
                        logger.debug(f"Received unhandled message type: {type(server_msg)}")

            except Exception as e:
                logger.error(f"Error processing server message: {e}", exc_info=True)
                if self.is_running:
                    await asyncio.sleep(1)  # Brief pause before retry if still running
