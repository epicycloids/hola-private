import asyncio
import logging
from typing import Any, AsyncIterator, Callable

import msgspec
import zmq
import zmq.asyncio
from zmq.log.handlers import PUBHandler

from hola.client.config import ClientConfig
from hola.client.handler import ServerMessageHandler
from hola.core.objectives import ObjectiveConfig, ObjectiveName
from hola.core.parameters import PredefinedParameterConfig, ParameterName
from hola.server.messages.base import OptimizationStatus
from hola.server.messages.client import (
    ClientMessage,
    LoadStateRequest,
    PauseRequest,
    ResumeRequest,
    SaveStateRequest,
    StatusRequest,
    UpdateObjectiveConfigRequest,
    UpdateParameterConfigRequest,
)
from hola.server.messages.server import ServerMessage, StatusUpdate


class Client:

    def __init__(
        self,
        config: ClientConfig | None = None,
        status_callback: Callable[[OptimizationStatus], None] | None = None,
    ):
        self.config = config or ClientConfig()
        self.status_callback = status_callback
        self._running = False

        # Set up logging
        self.logger = logging.getLogger("hola.client")
        self.logger.setLevel(logging.INFO)

        # Initialize ZMQ context and sockets
        self.context = zmq.asyncio.Context()

        # Command socket (REQ-REP pattern)
        self.command_socket = self.context.socket(zmq.REQ)
        self.command_socket.setsockopt(zmq.LINGER, 0)

        # Status subscription socket (PUB-SUB pattern)
        self.sub_socket = self.context.socket(zmq.SUB)
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b"status")

        # Log subscription socket
        self.log_sub = self.context.socket(zmq.SUB)
        self.log_sub.setsockopt(zmq.SUBSCRIBE, b"")

        self._connect_sockets()

        # Initialize message handler
        self.message_handler = ServerMessageHandler()

        # Add ZMQ logging handler
        handler = PUBHandler(self.log_sub)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - Client - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def _connect_sockets(self) -> None:
        if self.config.transport == "tcp":
            self.command_socket.connect(f"tcp://{self.config.host}:{self.config.command_port}")
            self.sub_socket.connect(f"tcp://{self.config.host}:{self.config.sub_port}")
            self.log_sub.connect(f"tcp://{self.config.host}:{self.config.log_port}")
        else:
            self.command_socket.connect(f"ipc://{self.config.command_socket}")
            self.sub_socket.connect(f"ipc://{self.config.sub_socket}")
            self.log_sub.connect(f"ipc://{self.config.log_socket}")

    async def _reconnect(self) -> None:
        # Close existing sockets
        self.command_socket.close()
        self.sub_socket.close()
        self.log_sub.close()

        # Create new sockets
        self.command_socket = self.context.socket(zmq.REQ)
        self.command_socket.setsockopt(zmq.LINGER, 0)

        self.sub_socket = self.context.socket(zmq.SUB)
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b"status")

        self.log_sub = self.context.socket(zmq.SUB)
        self.log_sub.setsockopt(zmq.SUBSCRIBE, b"")

        self._connect_sockets()

    async def _send_command(
        self,
        message: ClientMessage,
    ) -> Any:
        for attempt in range(self.config.max_retries):
            try:
                # Send command
                await self.command_socket.send(msgspec.json.encode(message))

                # Get response
                response_bytes = await self.command_socket.recv()
                response = msgspec.convert(msgspec.json.decode(response_bytes), ServerMessage)

                # Handle response
                return await self.message_handler.handle_message(response, self.logger)

            except Exception as e:
                self.logger.error(f"Command failed (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    await self._reconnect()
                    await asyncio.sleep(self.config.retry_interval)
                else:
                    raise

    async def start(self) -> None:
        """Start the client's status subscription handling."""
        self._running = True
        self.logger.info("Client starting")

        # Start status subscription handler if callback is provided
        if self.status_callback is not None:
            asyncio.create_task(self._handle_status_updates())

    async def stop(self) -> None:
        """Stop the client and clean up resources."""
        self._running = False
        self.logger.info("Client stopping")

        # Close all sockets
        self.command_socket.close()
        self.sub_socket.close()
        self.log_sub.close()

        # Terminate context
        self.context.term()

        self.logger.info("Client stopped")

    async def _handle_status_updates(self) -> None:
        """Handle incoming status updates from subscription."""
        while self._running:
            try:
                [topic, message_bytes] = await self.sub_socket.recv_multipart()
                if topic == b"status":
                    message = msgspec.convert(msgspec.json.decode(message_bytes), StatusUpdate)
                    if self.status_callback is not None:
                        self.status_callback(message.status)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error handling status update: {e}")
                await asyncio.sleep(self.config.retry_interval)

    async def get_status(self) -> OptimizationStatus:
        """Get current optimization status."""
        return await self._send_command(StatusRequest())

    async def pause(self) -> bool:
        """Pause the optimization."""
        return await self._send_command(PauseRequest())

    async def resume(self) -> bool:
        """Resume the optimization."""
        return await self._send_command(ResumeRequest())

    async def update_objective_config(self, config: dict[ObjectiveName, ObjectiveConfig]) -> bool:
        """Update objective configuration."""
        return await self._send_command(UpdateObjectiveConfigRequest(new_config=config))

    async def update_parameter_config(self, config: dict[ParameterName, PredefinedParameterConfig]) -> bool:
        """Update parameter configuration."""
        return await self._send_command(UpdateParameterConfigRequest(new_config=config))

    async def save_state(self, filepath: str) -> bool:
        return await self._send_command(SaveStateRequest(filepath=filepath))

    async def load_state(self, filepath: str) -> bool:
        return await self._send_command(LoadStateRequest(filepath=filepath))

    async def stream_status(self) -> AsyncIterator[OptimizationStatus]:
        """Stream status updates as an async iterator."""
        while self._running:
            try:
                [topic, message_bytes] = await self.sub_socket.recv_multipart()
                if topic == b"status":
                    message = msgspec.convert(msgspec.json.decode(message_bytes), StatusUpdate)
                    yield message.status

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in status stream: {e}")
                await asyncio.sleep(self.config.retry_interval)
