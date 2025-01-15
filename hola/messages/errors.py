from enum import Enum, auto
from typing import Any

from msgspec import Struct


class ErrorSeverity(Enum):
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


class ErrorDomain(Enum):
    PARAMETER = auto()
    OBJECTIVE = auto()
    SYSTEM = auto()
    CONFIG = auto()


class StructuredError(Struct, frozen=True):
    code: str
    domain: ErrorDomain
    severity: ErrorSeverity
    message: str
    details: dict[str, Any] | None = None
    retry_suggested: bool = False
