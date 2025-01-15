from typing import TypeAlias, Union

from hola.server.handlers.client import ClientMessageHandler
from hola.server.handlers.worker import WorkerMessageHandler

MessageHandler: TypeAlias = Union[ClientMessageHandler, WorkerMessageHandler]
