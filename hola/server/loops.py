import asyncio
from abc import ABC, abstractmethod
from uuid import UUID

import msgspec
import zmq

from hola.messages.client import ClientMessage
from hola.messages.worker import WorkerMessage
from hola.server.base import ExecutionState
from hola.server.handlers import MessageHandler


class MessageLoop(ABC):
    """Base class for message processing loops."""

    def __init__(
        self,
        socket: zmq.Socket,
        handler: MessageHandler,
        state: ExecutionState,
    ):
        self.socket = socket
        self.handler = handler
        self.state = state
        self._running = True

    async def stop(self) -> None:
        self._running = False

    @abstractmethod
    async def run(self) -> None:
        pass


class WorkerMessageLoop(MessageLoop):
    """Handles worker message processing."""

    async def run(self) -> None:
        while self._running:
            try:
                # Receive message parts
                identity, _, message_bytes = self.socket.recv_multipart()
                worker_id = UUID(bytes=identity)
                message = msgspec.convert(msgspec.json.decode(message_bytes), WorkerMessage)

                self.state.logger.debug(f"Received message from worker {worker_id}: {message}")

                # Update worker status
                await self.state.worker_status.handle_message(worker_id, message, self.state.logger)

                # Handle message
                response = await self.handler.handle_message(message, self.state)

                if self._running:
                    await self.socket.send_multipart([identity, b"", msgspec.json.encode(response)])
                    self.state.logger.debug(f"Sent response to worker {worker_id}: {response}")

            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.state.logger.error(f"Error in worker message loop: {e}")
                raise


class ClientMessageLoop(MessageLoop):
    """Handles client message processing."""

    async def run(self) -> None:
        while self._running:
            try:
                identity, _, message_bytes = self.socket.recv_multipart()
                client_id = UUID(bytes=identity)
                message = msgspec.convert(msgspec.json.decode(message_bytes), ClientMessage)

                self.state.logger.debug(f"Received message from client {client_id}: {message}")

                response = await self.handler.handle_message(message, self.state)

                if self._running:
                    await self.socket.send_multipart([identity, b"", msgspec.json.encode(response)])
                    self.state.logger.debug(f"Sent response to client {client_id}: {response}")

            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.state.logger.error(f"Error in client message loop: {e}")
                raise
