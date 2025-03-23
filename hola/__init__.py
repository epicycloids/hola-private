"""
HOLA: Hyperparameter Optimization, Lightweight and Asynchronous.

A flexible framework for distributed hyperparameter optimization with support for
multi-objective optimization, adaptive sampling, and distributed evaluation.
"""

# Core functionality
from hola.core.coordinator import OptimizationCoordinator
from hola.core.leaderboard import Leaderboard, Trial
from hola.core.objectives import ObjectiveScorer
from hola.core.parameters import ParameterTransformer
import hola.core.samplers

# Distributed functionality
from hola.distributed.scheduler import OptimizationScheduler, SchedulerConfig
from hola.distributed.server import OptimizationServer, ServerConfig
from hola.distributed.worker import Worker, LocalWorker, RemoteWorker, WorkerConfig

# Convenience function for simple optimizations
def run_optimization(
    objective_function,
    parameters_dict,
    objectives_dict,
    n_iterations=100,
    use_distributed=False,
    n_workers=None,
    **kwargs
):
    """
    Run an optimization process with the specified parameters and objectives.

    This is a convenience function that sets up the optimization system and runs
    it for the specified number of iterations. For more complex setups, use the
    individual components directly.

    :param objective_function: Function that evaluates parameters and returns objectives
    :type objective_function: Callable[[Dict[str, Any]], Dict[str, float]]
    :param parameters_dict: Parameter configuration dictionary
    :type parameters_dict: Dict[str, Dict[str, Any]]
    :param objectives_dict: Objective configuration dictionary
    :type objectives_dict: Dict[str, Dict[str, Any]]
    :param n_iterations: Number of iterations to run
    :type n_iterations: int
    :param use_distributed: Whether to use distributed optimization
    :type use_distributed: bool
    :param n_workers: Number of workers to use (only for distributed)
    :type n_workers: Optional[int]
    :param kwargs: Additional keyword arguments
    :type kwargs: Any
    :return: Best trial found
    :rtype: Trial

    Example:
    ```python
    parameters = {
        "x": {"type": "continuous", "min": 0.0, "max": 10.0},
        "y": {"type": "continuous", "min": 0.0, "max": 10.0}
    }

    objectives = {
        "f1": {
            "target": 0.0,
            "limit": 100.0,
            "direction": "minimize",
            "priority": 1.0,
            "comparison_group": 0
        }
    }

    def evaluate(params):
        x, y = params["x"], params["y"]
        return {"f1": x**2 + y**2}

    best_trial = run_optimization(
        objective_function=evaluate,
        parameters_dict=parameters,
        objectives_dict=objectives,
        n_iterations=100
    )
    ```
    """
    from hola.core.samplers import ExploreExploitSampler, SobolSampler, ClippedGaussianMixtureSampler

    # Determine parameter dimension (number of parameters)
    dimension = len(parameters_dict)

    # Create samplers for exploration and exploitation
    explore_sampler = SobolSampler(dimension=dimension)
    exploit_sampler = ClippedGaussianMixtureSampler(dimension=dimension, n_components=2)

    # Create an explore-exploit sampler
    sampler = ExploreExploitSampler(
        explore_sampler=explore_sampler,
        exploit_sampler=exploit_sampler
    )

    # Extra parameters that should go to the coordinator
    minimum_fit_samples = kwargs.get('minimum_fit_samples', 5)
    top_frac = kwargs.get('top_frac', 0.2)

    # Create coordinator
    coordinator = OptimizationCoordinator.from_dict(
        hypercube_sampler=sampler,
        objectives_dict=objectives_dict,
        parameters_dict=parameters_dict,
        minimum_fit_samples=minimum_fit_samples,
        top_frac=top_frac
    )

    if use_distributed:
        # Use distributed optimization
        import time
        from hola.distributed.scheduler import OptimizationScheduler
        from hola.distributed.server import OptimizationServer
        from hola.distributed.worker import LocalWorker

        # Configure number of workers
        n_workers = n_workers or max(1, min(4, __import__('os').cpu_count() or 1))

        # Create scheduler and server
        scheduler = OptimizationScheduler(coordinator=coordinator)
        server = OptimizationServer(
            scheduler=scheduler,
            config=ServerConfig(
                zmq_ipc_endpoint="ipc:///tmp/hola-optimization.ipc",
                job_cleanup_interval=10.0
            )
        )

        # Start server
        server.start()

        # Create and start workers
        workers = []
        for i in range(n_workers):
            worker = LocalWorker(
                objective_function=objective_function,
                zmq_ipc_endpoint="ipc:///tmp/hola-optimization.ipc",
                config=WorkerConfig(worker_id=f"worker-{i+1}")
            )
            worker.start()
            workers.append(worker)

        # Wait for iterations to complete
        try:
            while coordinator.get_total_evaluations() < n_iterations:
                time.sleep(0.1)
        finally:
            # Clean up
            for worker in workers:
                worker.stop()
            server.stop()
    else:
        # Use simple single-threaded optimization
        for i in range(n_iterations):
            # Get parameter suggestions
            params_list, metadata = coordinator.suggest_parameters()

            # Evaluate objectives
            if params_list:
                objectives = objective_function(params_list[0])

                # Record the evaluation
                coordinator.record_evaluation(params_list[0], objectives, metadata)

    # Return best trial
    return coordinator.get_best_trial()