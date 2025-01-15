from dataclasses import dataclass
from typing import Literal


@dataclass
class ClientConfig:

    transport: Literal["tcp", "ipc"] = "tcp"
    host: str = "localhost"
    command_port: int = 5556
    sub_port: int = 5557
    log_port: int = 5558
    command_socket: str = "/tmp/hola-commands"
    sub_socket: str = "/tmp/hola-status"
    log_socket: str = "/tmp/hola-logs"
    retry_interval: float = 1.0
    max_retries: int = 5
