from dataclasses import dataclass
from typing import Literal


@dataclass
class WorkerConfig:
    """Configuration for Worker connection and behavior."""

    transport: Literal["tcp", "ipc"] = "tcp"
    host: str = "localhost"
    port: int = 5555
    log_port: int = 5558
    socket_path: str = "/tmp/hola-workers"
    log_socket_path: str = "/tmp/hola-logs"
    retry_interval: float = 1.0
    max_retries: int = 5
    heartbeat_interval: float = 30.0
