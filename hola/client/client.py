import logging
from datetime import datetime, timezone

import msgspec
import zmq
import zmq.asyncio

from hola.messages.client import (
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
from hola.messages.server import ServerMessage
from hola.server.config import ConnectionConfig

logger = logging.getLogger(__name__)


class OptimizationClient:
    def __init__(self, config: ConnectionConfig):
        self.config = config
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.DEALER)

    async def start(self):
        """Start the client and connect to the server."""
        self.socket.connect(self.config.client_uri)

    async def stop(self):
        """Stop the client gracefully."""
        self.socket.close()
        self.context.term()

    async def initialize(self, objectives_config: dict, parameters_config: dict) -> ServerMessage:
        """Initialize the optimization process."""
        message = InitializeRequest(
            timestamp=datetime.now(timezone.utc),
            objectives_config=objectives_config,
            parameters_config=parameters_config,
        )
        await self._send_message(message)
        response = await self._receive_message()
        logger.info(f"Received initialization response: {response}")
        return response

    async def get_status(self):
        """Request current optimization status."""
        message = StatusRequest(
            timestamp=datetime.now(timezone.utc),
        )
        await self._send_message(message)
        return await self._receive_message()

    async def pause(self):
        """Pause the optimization process."""
        message = PauseRequest(
            timestamp=datetime.now(timezone.utc),
        )
        await self._send_message(message)

    async def resume(self):
        """Resume the optimization process."""
        message = ResumeRequest(
            timestamp=datetime.now(timezone.utc),
        )
        await self._send_message(message)

    async def update_objectives(self, new_config: dict):
        """Update objective configuration."""
        message = UpdateObjectiveConfig(timestamp=datetime.now(timezone.utc), new_config=new_config)
        await self._send_message(message)

    async def update_parameters(self, new_config: dict):
        """Update parameter configuration."""
        message = UpdateParameterConfig(timestamp=datetime.now(timezone.utc), new_config=new_config)
        await self._send_message(message)

    async def save_state(self, filepath: str):
        """Request to save the current optimization state."""
        message = SaveStateRequest(timestamp=datetime.now(timezone.utc), filepath=filepath)
        await self._send_message(message)

    async def load_state(self, filepath: str):
        """Request to load a saved optimization state."""
        message = LoadStateRequest(timestamp=datetime.now(timezone.utc), filepath=filepath)
        await self._send_message(message)

    async def _send_message(self, message: ClientMessage):
        """Send a message to the server."""
        await self.socket.send(msgspec.json.encode(message))

    async def _receive_message(self) -> ServerMessage:
        """Receive a message from the server."""
        message = await self.socket.recv()
        return msgspec.json.decode(message, type=ServerMessage)
