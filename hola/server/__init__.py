"""Server components for distributed optimization."""

from hola.server.rest import Server
from hola.server.scheduler import SchedulerProcess

__all__ = ["SchedulerProcess", "Server"]
