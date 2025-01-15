from dataclasses import dataclass
from typing import Literal

import zmq
import zmq.asyncio


@dataclass
class SocketConfig:
    transport: Literal["tcp", "ipc"] = "tcp"
    host: str = "localhost"

    # Ports for TCP transport
    worker_port: int = 5555  # ROUTER for  worker DEALER
    client_port: int = 5556  # REP for client REQ
    pub_port: int = 5557  # PUB for client SUB
    log_port: int = 5558  # PUB for logging

    # Socket paths for IPC transport
    worker_socket: str = "/tmp/hola-workers"
    client_socket: str = "/tmp/hola-commands"
    pub_socket: str = "/tmp/hola-status"
    log_socket: str = "/tmp/hola-logs"


class ZMQSockets:
    """Manages ZMQ socket creation and cleanup."""

    def __init__(self, config: SocketConfig):
        self.context = zmq.asyncio.Context()
        self._sockets: list[zmq.Socket] = []

        # Worker socket (ROUTER for multiple workers)
        self.worker_router = self._bind_socket(
            zmq.ROUTER, config.transport, config.host, config.worker_port, config.worker_socket
        )

        # Client command socket (REP for client commands)
        self.client_router = self._bind_socket(
            zmq.REP,
            config.transport,
            config.host,
            config.client_port,
            config.client_socket,
        )

        # Status publication socket
        self.status_pub = self._bind_socket(
            zmq.PUB, config.transport, config.host, config.pub_port, config.pub_socket
        )

        # Log publication socket
        self.log_pub = self._bind_socket(
            zmq.PUB, config.transport, config.host, config.log_port, config.log_socket
        )

    def _bind_socket(
        self, sock_type: int, transport: str, host: str, port: int, socket_path: str
    ) -> zmq.Socket:
        socket = self.context.socket(sock_type)
        socket.setsockopt(zmq.LINGER, 0)
        if transport == "tcp":
            socket.bind(f"tcp://{host}:{port}")
        else:
            socket.bind(f"ipc://{socket_path}")
        self._sockets.append(socket)
        return socket

    async def cleanup(self) -> None:
        """Clean up all sockets and terminate context."""
        for socket in self._sockets:
            socket.close()
        self.context.term()
