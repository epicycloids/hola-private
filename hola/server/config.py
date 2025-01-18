from dataclasses import dataclass
from enum import Enum


class SocketType(str, Enum):
    TCP = "tcp"
    IPC = "ipc"


@dataclass
class ConnectionConfig:
    socket_type: SocketType
    host: str = "127.0.0.1"
    port: int = 5555
    ipc_path: str = "/tmp/hola-optimizer"

    @property
    def worker_uri(self) -> str:
        match self.socket_type:
            case SocketType.TCP:
                return f"tcp://{self.host}:{self.port}"
            case SocketType.IPC:
                return f"ipc://{self.ipc_path}-worker"

    @property
    def client_uri(self) -> str:
        match self.socket_type:
            case SocketType.TCP:
                return f"tcp://{self.host}:{self.port + 1}"
            case SocketType.IPC:
                return f"ipc://{self.ipc_path}-client"
