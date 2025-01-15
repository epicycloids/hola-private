from datetime import datetime

from msgspec import Struct


class BaseMessage(Struct, frozen=True):
    timestamp: datetime
