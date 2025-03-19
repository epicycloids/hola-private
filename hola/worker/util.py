"""
Utility functions for worker management in the HOLA optimization system.
"""

import multiprocessing as mp
import time
from typing import Callable, List, Tuple

from hola.utils.logging import setup_logging
from hola.worker.local import LocalWorker

# Setup module-level logger
logger = setup_logging("WorkerUtil")

def start_workers(
    eval_function: Callable,
    num_workers: int,
    server_url: str = "http://localhost:8000",
    use_ipc: bool = True,
) -> List[Tuple[LocalWorker, mp.Process]]:
    """
    Start multiple worker processes to evaluate parameters in parallel.

    Args:
        eval_function: Function that takes parameters and returns objective values
        num_workers: Number of worker processes to start
        server_url: URL of the HOLA server
        use_ipc: Whether to use IPC (faster) or HTTP for communication

    Returns:
        List of tuples containing (worker, process) pairs
    """
    logger.info(f"Starting {num_workers} workers connecting to {server_url}")

    workers = []
    for i in range(num_workers):
        worker = LocalWorker(
            eval_function=eval_function,
            worker_id=f"worker-{i}",
            server_url=server_url,
            use_ipc=use_ipc,
        )
        process = mp.Process(target=worker.run)
        process.daemon = True
        process.start()
        workers.append((worker, process))

        # Small delay to prevent server overload during registration
        time.sleep(0.1)

    logger.info(f"Successfully started {len(workers)} workers")
    return workers


def stop_workers(workers: List[Tuple[LocalWorker, mp.Process]]) -> None:
    """
    Stop a list of worker processes.

    Args:
        workers: List of (worker, process) tuples to stop
    """
    logger.info(f"Stopping {len(workers)} workers")

    for _, process in workers:
        if process.is_alive():
            process.terminate()

    # Wait for processes to terminate
    for _, process in workers:
        process.join(timeout=2)

    logger.info("All workers stopped")