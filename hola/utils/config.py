from dataclasses import dataclass
from typing import Literal


@dataclass
class NetworkConfig:
    """Network configuration for the optimization system."""

    tcp_host: str = "localhost"
    """Host address for TCP connections."""

    tcp_port: int = 5555
    """Port for TCP connections."""

    rest_host: str = "localhost"
    """Host address for REST API."""

    rest_port: int = 8000
    """Port for REST API."""

    use_ssl: bool = False
    """Whether to use HTTPS for REST API."""


@dataclass
class SystemConfig:
    """Configuration for the optimization system."""

    network: NetworkConfig = NetworkConfig()
    """Network configuration for various connection types."""

    local_workers: int = 0
    """Number of local worker processes to spawn."""

    local_transport: Literal["ipc", "tcp"] = "ipc"
    """Transport protocol for local workers: 'ipc' (recommended) or 'tcp'."""
