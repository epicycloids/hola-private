# HOLA: Hyperparameter Optimization, Lightweight and Asynchronous

HOLA is a flexible and powerful hyperparameter optimization framework designed for both single-machine and distributed settings. It provides an intuitive API for defining optimization problems and allows for efficient parallel evaluation of hyperparameter configurations.

## Core Features

- **Multi-objective optimization** with user-defined targets and limits
- **Distributed evaluation** support through IPC, TCP, or HTTP
- **Adaptive sampling** strategies that focus on promising regions
- **Flexible parameter types**: continuous, integer, categorical, and lattice-valued
- **Comparison groups** for handling incomparable objectives
- **Priority-weighted scoring** for balancing multiple objectives
- **Modern FastAPI-based HTTP server** for remote workers

## How It Works

### Parameter Configuration

Define your parameter space using various parameter types:

- **Continuous parameters**: float values with linear or log scaling
- **Integer parameters**: discrete integer values in a range
- **Categorical parameters**: selection from a fixed set of options
- **Lattice parameters**: evenly-spaced float values

Each parameter is defined with its constraints (min/max or categories) that determine its valid domain.

### Objective Configuration

Define your objectives with:

- **Target value**: the value at which you consider the objective satisfied
- **Limit value**: the value beyond which solutions are rejected
- **Direction**: whether to minimize or maximize the objective
- **Priority weight**: the relative importance within its comparison group
- **Comparison group**: how objectives are grouped for comparison

### Scoring System

The scoring system works as follows:

1. **Objective scaling**: Each objective is scaled to represent a percentage between its target and limit values:
   - Score of 0 means the objective meets or exceeds its target
   - Score of 1 means the objective is at the limit
   - Score of infinity means the objective exceeds its limit

2. **Comparison groups**: Objectives are combined within comparison groups using priority-weighted sums. This allows handling incomparable objectives separately.

3. **Optimization**: The system minimizes each comparison group's score, with adaptive sampling focusing on the most promising regions based on elite samples.

## Usage Example

### Simple Single-Machine Optimization

```python
from hola import run_optimization

# Define your parameters
parameters = {
    "x": {"type": "continuous", "min": 0.0, "max": 10.0},
    "y": {"type": "continuous", "min": 0.0, "max": 10.0},
}

# Define your objectives
objectives = {
    "f1": {
        "target": 0.0,
        "limit": 100.0,
        "direction": "minimize",
        "priority": 1.0,
        "comparison_group": 0
    },
    "f2": {
        "target": 0.0,
        "limit": 100.0,
        "direction": "minimize",
        "priority": 0.8,
        "comparison_group": 0
    },
}

# Define your evaluation function
def evaluate(params):
    x = params["x"]
    y = params["y"]
    f1 = x**2 + y**2
    f2 = (x-2)**2 + (y-2)**2
    return {"f1": f1, "f2": f2}

# Run the optimization
best_trial = run_optimization(
    objective_function=evaluate,
    parameters_dict=parameters,
    objectives_dict=objectives,
    n_iterations=100,
    minimum_fit_samples=5
)

print(f"Best parameters: {best_trial.parameters}")
print(f"Best objectives: {best_trial.objectives}")
```

### Distributed Optimization

HOLA supports distributed optimization with a central server and multiple workers, similar to Folding@Home. Workers can connect via ZMQ (IPC or TCP) or HTTP with FastAPI.

```python
from hola.core.coordinator import OptimizationCoordinator
from hola.core.samplers import ExploreExploitSampler, SobolSampler, ClippedGaussianMixtureSampler
from hola.distributed.scheduler import OptimizationScheduler, SchedulerConfig
from hola.distributed.server import OptimizationServer, ServerConfig
from hola.distributed.worker import LocalWorker, RemoteWorker, WorkerConfig

# Define parameters and objectives (same as above)
parameters = {
    "x": {"type": "continuous", "min": 0.0, "max": 10.0},
    "y": {"type": "continuous", "min": 0.0, "max": 10.0},
}

# Define objectives
objectives = {
    "f1": {
        "target": 0.0,
        "limit": 100.0,
        "direction": "minimize",
        "priority": 1.0,
        "comparison_group": 0
    },
    "f2": {
        "target": 0.0,
        "limit": 100.0,
        "direction": "minimize",
        "priority": 0.8,
        "comparison_group": 0
    },
}

# Create samplers for exploration and exploitation
explore_sampler = SobolSampler(dimension=2)
exploit_sampler = ClippedGaussianMixtureSampler(dimension=2, n_components=2)

# Create an explore-exploit sampler
sampler = ExploreExploitSampler(
    explore_sampler=explore_sampler,
    exploit_sampler=exploit_sampler
)

# Create coordinator
coordinator = OptimizationCoordinator.from_dict(
    hypercube_sampler=sampler,
    objectives_dict=objectives,
    parameters_dict=parameters,
    minimum_fit_samples=5,
    top_frac=0.2
)

# Create scheduler
scheduler = OptimizationScheduler(coordinator=coordinator)

# Create server
server = OptimizationServer(
    scheduler=scheduler,
    config=ServerConfig(
        zmq_ipc_endpoint="ipc:///tmp/hola-optimization.ipc",  # For local workers
        zmq_tcp_endpoint="tcp://127.0.0.1:5555",             # For remote workers with ZMQ
        http_port=8080                                        # For HTTP workers
    )
)

# Start server
server.start()

# Create and start workers
workers = []

# Local worker using ZMQ IPC
local_worker = LocalWorker(
    objective_function=evaluate,
    zmq_ipc_endpoint="ipc:///tmp/hola-optimization.ipc"
)
local_worker.start()
workers.append(local_worker)

# Remote worker using ZMQ TCP
remote_worker_zmq = RemoteWorker(
    objective_function=evaluate,
    zmq_tcp_endpoint="tcp://127.0.0.1:5555"
)
remote_worker_zmq.start()
workers.append(remote_worker_zmq)

# Remote worker using HTTP
remote_worker_http = RemoteWorker(
    objective_function=evaluate,
    server_url="http://localhost:8080"
)
remote_worker_http.start()
workers.append(remote_worker_http)

# Monitor progress
import time
try:
    while coordinator.get_total_evaluations() < 100:
        time.sleep(1.0)
        print(f"Progress: {coordinator.get_total_evaluations()}/100 evaluations")
finally:
    # Stop workers and server
    for worker in workers:
        worker.stop()
    server.stop()

# Get final results
best_trial = coordinator.get_best_trial()
print(f"Best parameters: {best_trial.parameters}")
print(f"Best objectives: {best_trial.objectives}")
```

### Using the Simplified Distributed API

For convenience, you can also use the `run_optimization` function with `use_distributed=True`:

```python
from hola import run_optimization

# Define parameters, objectives, and evaluation function (same as above)
parameters = {
    "x": {"type": "continuous", "min": 0.0, "max": 10.0},
    "y": {"type": "continuous", "min": 0.0, "max": 10.0},
}
# ...

# Run distributed optimization
best_trial = run_optimization(
    objective_function=evaluate,
    parameters_dict=parameters,
    objectives_dict=objectives,
    n_iterations=100,
    use_distributed=True,
    n_workers=4,  # Number of local workers to create
    minimum_fit_samples=5  # Additional parameter for the coordinator
)

print(f"Best parameters: {best_trial.parameters}")
print(f"Best objectives: {best_trial.objectives}")
```

## External Worker Implementation

For non-Python workers or other languages, you can implement the HTTP API endpoints provided by FastAPI:

1. **Register**: `POST /api/register` with optional `{"worker_id": "..."}` to get a worker ID
2. **Get Job**: `GET /api/job?worker_id=...` to get parameters to evaluate
3. **Submit Result**: `POST /api/result` with `{"worker_id": "...", "job_id": "...", "objectives": {...}}`
4. **Heartbeat**: `POST /api/heartbeat` with `{"worker_id": "..."}` to keep connection alive

The API also includes automatic OpenAPI documentation at `/docs` which provides detailed information about request and response schemas. This allows workers implemented in any language to participate in the optimization.
