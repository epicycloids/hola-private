import asyncio
import logging
from typing import Any, Callable
from uuid import UUID, uuid4

import msgspec
import zmq
import zmq.asyncio
from zmq.log.handlers import PUBHandler

from hola.core.objectives import ObjectiveName
from hola.server.messages.server import Error, ServerMessage
from hola.server.messages.worker import SampleRequest
from hola.worker.base import WorkerState
from hola.worker.config import WorkerConfig
from hola.worker.handler import ServerMessageHandler


class Worker:
    """A worker that executes objective function evaluations."""

    def __init__(
        self,
        objective_fn: Callable[..., dict[ObjectiveName, float]],
        config: WorkerConfig | None = None,
    ):
        self.config = config or WorkerConfig()
        self.worker_id = UUID(bytes=uuid4().bytes)
        self._running = False

        # Set up logging
        self.logger = logging.getLogger(f"hola.worker.{self.worker_id}")
        self.logger.setLevel(logging.INFO)

        # Initialize state and message handler
        self.state = WorkerState(
            worker_id=self.worker_id, objective_fn=objective_fn, logger=self.logger
        )
        self.message_handler = ServerMessageHandler()

        # Initialize ZMQ context and sockets
        self.context = zmq.asyncio.Context()
        self.dealer = self.context.socket(zmq.DEALER)
        self.dealer.setsockopt(zmq.IDENTITY, str(self.worker_id).encode())
        self.dealer.setsockopt(zmq.LINGER, 0)

        # Connect to log publisher
        self.log_sub = self.context.socket(zmq.SUB)
        self.log_sub.setsockopt(zmq.SUBSCRIBE, b"")

        self._connect_sockets()

        # Add ZMQ logging handler
        handler = PUBHandler(self.log_sub)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            f"%(asctime)s - Worker {self.worker_id} - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def _connect_sockets(self) -> None:
        """Connect to server sockets."""
        if self.config.transport == "tcp":
            self.dealer.connect(f"tcp://{self.config.host}:{self.config.port}")
            self.log_sub.connect(f"tcp://{self.config.host}:{self.config.log_port}")
        else:
            self.dealer.connect(f"ipc://{self.config.socket_path}")
            self.log_sub.connect(f"ipc://{self.config.log_socket_path}")

    async def _reconnect(self) -> None:
        """Attempt to reconnect to the server with retries."""
        for attempt in range(self.config.max_retries):
            try:
                self.dealer.close()
                self.log_sub.close()

                self.dealer = self.context.socket(zmq.DEALER)
                self.dealer.setsockopt(zmq.IDENTITY, str(self.worker_id).encode())
                self.dealer.setsockopt(zmq.LINGER, 0)

                self.log_sub = self.context.socket(zmq.SUB)
                self.log_sub.setsockopt(zmq.SUBSCRIBE, b"")

                self._connect_sockets()
                return

            except Exception as e:
                self.logger.error(f"Reconnection attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_interval)
                else:
                    raise

    async def _send_message(self, message: Any) -> None:
        """Send a message to the server with retries."""
        for attempt in range(self.config.max_retries):
            try:
                await self.dealer.send_multipart([b"", msgspec.json.encode(message)])
                return
            except Exception as e:
                self.logger.error(f"Failed to send message (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    await self._reconnect()
                else:
                    raise

    async def _receive_message(self) -> ServerMessage:
        """Receive and parse a message from the server with retries."""
        for attempt in range(self.config.max_retries):
            try:
                _, response_bytes = await self.dealer.recv_multipart()
                return msgspec.convert(msgspec.json.decode(response_bytes), ServerMessage)
            except Exception as e:
                self.logger.error(f"Failed to receive message (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    await self._reconnect()
                else:
                    raise

    async def run(self) -> None:
        """Run the worker's main loop."""
        self._running = True
        self.logger.info("Worker starting")

        try:
            while self._running:
                try:
                    # Request work
                    request = SampleRequest(n_samples=1)
                    await self._send_message(request)

                    # Handle response
                    response = await self._receive_message()
                    result = await self.message_handler.handle_message(response, self.state)

                    # Send result if we got one
                    if result is not None:
                        await self._send_message(result)
                        ack = await self._receive_message()
                        if isinstance(ack, Error):
                            self.logger.error(f"Server error: {ack.message}")
                            await asyncio.sleep(self.config.retry_interval)

                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    self.logger.error(f"Error in main loop: {e}")
                    await asyncio.sleep(self.config.retry_interval)

        finally:
            self._running = False
            self.dealer.close()
            self.log_sub.close()
            self.context.term()
            self.logger.info("Worker stopped")
