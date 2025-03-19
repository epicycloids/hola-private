"""
HOLA: Hyperparameter Optimization, Lightweight and Asynchronous.

A flexible framework for distributed hyperparameter optimization with support for
multi-objective optimization, adaptive sampling, and distributed evaluation.
"""

from hola.cli.run import run_optimization_system
from hola.core.coordinator import OptimizationCoordinator
from hola.core.system import HOLA, SystemConfig
from hola.server import SchedulerProcess, Server
from hola.utils.logging import setup_logging
from hola.worker.local import LocalWorker
from hola.worker.util import start_workers, stop_workers

__version__ = "0.1.0"

__all__ = [
    "OptimizationCoordinator",
    "SchedulerProcess",
    "Server",
    "LocalWorker",
    "start_workers",
    "stop_workers",
    "run_optimization_system",
    "HOLA",
    "SystemConfig",
    "setup_logging",
]
