"""
Scheduler for distributed optimization.

The scheduler wraps an OptimizationCoordinator and handles asynchronous parameter
suggestions and trial recording for distributed workers.
"""

import time
import uuid
import threading
import logging
import traceback
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from hola.core.coordinator import OptimizationCoordinator
from hola.core.leaderboard import Trial

# Set up logging with less verbose default level
logging.basicConfig(
    level=logging.INFO,  # Default to INFO for normal operation
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("hola.distributed.scheduler")


@dataclass
class SchedulerConfig:
    """Configuration for OptimizationScheduler."""

    max_retries: int = 3
    """Maximum number of retries for failed evaluations."""

    retry_delay: float = 5.0
    """Delay in seconds between retries."""


@dataclass
class SchedulerJob:
    """Represents a scheduled optimization job."""

    job_id: str
    """Unique identifier for the job."""

    parameters: Dict[str, Any]
    """Parameters to evaluate."""

    metadata: Dict[str, Any]
    """Metadata about the job."""

    worker_id: Optional[str] = None
    """ID of the worker assigned to this job."""

    attempts: int = 0
    """Number of attempt to evaluate this job."""

    timestamp: float = field(default_factory=time.time)
    """Time when the job was created."""


@dataclass
class OptimizationScheduler:
    """
    Scheduler for distributed optimization.

    The scheduler manages:
    - Suggesting parameters for workers
    - Tracking active jobs and workers
    - Handling job failures and retries
    - Recording evaluation results
    """

    coordinator: OptimizationCoordinator
    """The underlying optimization coordinator."""

    config: SchedulerConfig = field(default_factory=SchedulerConfig)
    """Configuration for the scheduler."""

    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)
    """Lock for thread-safe operations."""

    _active_jobs: Dict[str, SchedulerJob] = field(default_factory=dict, init=False)
    """Map of job_id to active jobs."""

    _active_workers: Dict[str, str] = field(default_factory=dict, init=False)
    """Map of worker_id to job_id."""

    def __post_init__(self):
        """Initialize the scheduler after construction."""
        logger.debug(f"Initialized OptimizationScheduler with coordinator {self.coordinator}")

    def suggest_parameters(self, worker_id: str, n_samples: int = 1) -> Tuple[str, Dict[str, Any]]:
        """
        Suggest parameters for a worker to evaluate.

        :param worker_id: ID of the worker requesting parameters
        :type worker_id: str
        :param n_samples: Number of parameter sets to generate
        :type n_samples: int
        :return: Tuple of (job_id, parameters)
        :rtype: Tuple[str, Dict[str, Any]]
        """
        logger.debug(f"Worker {worker_id} requesting parameters (n_samples={n_samples})")

        with self._lock:
            # Check if worker already has an active job
            if worker_id in self._active_workers:
                job_id = self._active_workers[worker_id]
                job = self._active_jobs.get(job_id)
                if job:
                    logger.info(f"Worker {worker_id} already has active job {job_id}")
                    return job_id, job.parameters
                else:
                    logger.warning(f"Worker {worker_id} has reference to job {job_id} which no longer exists")

            # Get new parameters from coordinator
            logger.debug(f"Requesting parameters from coordinator for worker {worker_id}")
            try:
                param_dicts, metadata = self.coordinator.suggest_parameters(n_samples)
                logger.debug(f"Received {len(param_dicts)} parameter sets from coordinator")
            except Exception as e:
                logger.error(f"Error getting parameters from coordinator: {str(e)}")
                logger.debug(traceback.format_exc())
                raise

            # Create new job (use the first set of parameters)
            if param_dicts:
                job_id = str(uuid.uuid4())
                job = SchedulerJob(
                    job_id=job_id,
                    parameters=param_dicts[0],
                    metadata=metadata,
                    worker_id=worker_id,
                    attempts=1
                )

                # Track the job
                self._active_jobs[job_id] = job
                self._active_workers[worker_id] = job_id

                logger.info(f"Created job {job_id} for worker {worker_id} with parameters: {param_dicts[0]}")
                return job_id, job.parameters
            else:
                logger.error("Coordinator returned empty parameter list")
                raise RuntimeError("Failed to generate parameters")

    def record_evaluation(
        self,
        job_id: str,
        worker_id: str,
        objectives: Dict[str, float],
        success: bool = True
    ) -> Trial:
        """
        Record the evaluation results for a job.

        :param job_id: ID of the completed job
        :type job_id: str
        :param worker_id: ID of the worker that completed the job
        :type worker_id: str
        :param objectives: Objective values achieved
        :type objectives: Dict[str, float]
        :param success: Whether the evaluation was successful
        :type success: bool
        :return: The best trial after recording the evaluation
        :rtype: Trial
        """
        logger.debug(f"Worker {worker_id} submitting results for job {job_id}: {objectives}, success={success}")

        with self._lock:
            # Check if job exists
            if job_id not in self._active_jobs:
                logger.warning(f"Job {job_id} not found in active jobs")
                raise ValueError(f"Job {job_id} not found")

            job = self._active_jobs[job_id]

            # Check if worker matches
            if job.worker_id != worker_id:
                logger.warning(f"Worker ID mismatch for job {job_id}: {worker_id} ≠ {job.worker_id}")
                # Continue anyway, as the worker might have reconnected

            if success:
                # Record successful evaluation
                logger.debug(f"Recording successful evaluation for job {job_id}")
                try:
                    best_trial = self.coordinator.record_evaluation(
                        parameters=job.parameters,
                        objectives=objectives,
                        metadata={
                            **job.metadata,
                            "job_id": job_id,
                            "worker_id": worker_id,
                            "attempts": job.attempts,
                            "duration": time.time() - job.timestamp
                        }
                    )

                    # Log result
                    if best_trial:
                        logger.info(f"Recorded job {job_id} from worker {worker_id}, best trial: {best_trial.trial_id}")
                    else:
                        logger.info(f"Recorded job {job_id} from worker {worker_id}, no best trial yet")

                except Exception as e:
                    logger.error(f"Error recording evaluation in coordinator: {str(e)}")
                    logger.debug(traceback.format_exc())
                    raise

                # Clean up job tracking
                logger.debug(f"Removing job {job_id} from active jobs and worker {worker_id} from active workers")
                self._active_jobs.pop(job_id, None)
                self._active_workers.pop(worker_id, None)

                return best_trial
            else:
                # Handle failed evaluation
                if job.attempts < self.config.max_retries:
                    # Increment attempts and reset worker
                    job.attempts += 1
                    job.worker_id = None
                    job.timestamp = time.time()
                    self._active_workers.pop(worker_id, None)

                    logger.warning(f"Job {job_id} failed, will retry (attempt {job.attempts}/{self.config.max_retries})")
                else:
                    # Too many failures, record as failed
                    logger.error(f"Job {job_id} failed after {job.attempts} attempts, abandoning")

                    # Record with infinity values for all objectives
                    try:
                        self.coordinator.record_evaluation(
                            parameters=job.parameters,
                            objectives={obj: float('inf') for obj in objectives.keys()},
                            metadata={
                                **job.metadata,
                                "job_id": job_id,
                                "worker_id": worker_id,
                                "attempts": job.attempts,
                                "failed": True
                            }
                        )
                    except Exception as e:
                        logger.error(f"Error recording failed evaluation: {str(e)}")
                        logger.debug(traceback.format_exc())

                    # Clean up job tracking
                    self._active_jobs.pop(job_id, None)
                    self._active_workers.pop(worker_id, None)

    def get_active_jobs(self) -> List[SchedulerJob]:
        """
        Get all active jobs.

        :return: List of active jobs
        :rtype: List[SchedulerJob]
        """
        with self._lock:
            jobs = list(self._active_jobs.values())
            logger.debug(f"Retrieved {len(jobs)} active jobs")
            return jobs

    def get_active_workers(self) -> Dict[str, str]:
        """
        Get all active workers and their jobs.

        :return: Map of worker_id to job_id
        :rtype: Dict[str, str]
        """
        with self._lock:
            workers = self._active_workers.copy()
            logger.debug(f"Retrieved {len(workers)} active workers")
            return workers

    def reset_stalled_jobs(self, max_age: float = 3600.0) -> int:
        """
        Reset jobs that have been active for too long.

        :param max_age: Maximum age in seconds before a job is considered stalled
        :type max_age: float
        :return: Number of jobs reset
        :rtype: int
        """
        logger.debug(f"Checking for stalled jobs (max_age={max_age})")

        with self._lock:
            current_time = time.time()
            count = 0

            for job_id, job in list(self._active_jobs.items()):
                age = current_time - job.timestamp

                if age > max_age:
                    # Job is stalled, reset it if there are attempts left
                    if job.attempts < self.config.max_retries:
                        # Remove worker assignment
                        if job.worker_id:
                            logger.debug(f"Removing stalled worker {job.worker_id} from active workers")
                            self._active_workers.pop(job.worker_id, None)

                        # Reset job
                        job.worker_id = None
                        job.attempts += 1
                        job.timestamp = current_time
                        count += 1

                        logger.warning(f"Reset stalled job {job_id} (attempt {job.attempts}/{self.config.max_retries})")
                    else:
                        # Too many failures, record as failed
                        logger.error(f"Job {job_id} stalled after {job.attempts} attempts, abandoning")

                        # Clean up job tracking
                        self._active_jobs.pop(job_id, None)
                        if job.worker_id:
                            self._active_workers.pop(job.worker_id, None)

            if count > 0:
                logger.info(f"Reset {count} stalled jobs")
            else:
                logger.debug("No stalled jobs found")

            return count

    def get_active_job_count(self) -> int:
        """
        Get the count of active jobs.

        :return: Number of active jobs
        :rtype: int
        """
        with self._lock:
            return len(self._active_jobs)

    def get_worker_stats(self, worker_id: str) -> Dict[str, Any]:
        """
        Get statistics for a specific worker.

        :param worker_id: ID of the worker
        :type worker_id: str
        :return: Worker statistics
        :rtype: Dict[str, Any]
        """
        with self._lock:
            # Initialize stats with defaults
            stats = {
                "total_jobs": 0,
                "successful_jobs": 0,
                "failed_jobs": 0,
                "current_job": None
            }

            # If we have an active job for this worker, include it
            if worker_id in self._active_workers:
                job_id = self._active_workers[worker_id]
                job = self._active_jobs.get(job_id)
                if job:
                    stats["current_job"] = job_id

            # Count jobs in coordinator metadata
            try:
                trials_metadata = self.coordinator.get_all_trials_metadata()
                if trials_metadata:
                    for trial_id, metadata in trials_metadata.items():
                        if metadata.get("worker_id") == worker_id:
                            stats["total_jobs"] += 1
                            if metadata.get("failed", False):
                                stats["failed_jobs"] += 1
                            else:
                                stats["successful_jobs"] += 1
            except Exception as e:
                logger.error(f"Error getting worker stats from trials: {str(e)}")

            return stats

    def get_active_job_for_worker(self, worker_id: str) -> Optional[str]:
        """
        Get the active job ID for a worker, if any.

        :param worker_id: ID of the worker
        :type worker_id: str
        :return: Job ID or None if worker has no active job
        :rtype: Optional[str]
        """
        with self._lock:
            return self._active_workers.get(worker_id)

    def get_all_active_jobs(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all active jobs for monitoring.

        :return: Dictionary of job information, keyed by job ID
        :rtype: Dict[str, Dict[str, Any]]
        """
        with self._lock:
            current_time = time.time()
            jobs_info = {}

            for job_id, job in self._active_jobs.items():
                jobs_info[job_id] = {
                    "worker_id": job.worker_id,
                    "parameters": job.parameters,
                    "status": "assigned" if job.worker_id else "unassigned",
                    "attempts": job.attempts,
                    "age": current_time - job.timestamp,
                    "start_time": job.timestamp,
                    "metadata": job.metadata
                }

            return jobs_info

    def set_verbose_logging(self, verbose: bool = False):
        """
        Set the verbosity level of the scheduler logging.

        :param verbose: Whether to use verbose (DEBUG) logging
        :type verbose: bool
        """
        if verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)