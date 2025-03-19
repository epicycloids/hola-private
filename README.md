# HOLA: Hyperparameter Optimization, Lightweight and Asynchronous

HOLA is a flexible and powerful hyperparameter optimization framework designed for both single-machine and distributed settings. It provides an intuitive API for defining optimization problems and allows for efficient parallel evaluation of hyperparameter configurations.

## Core Features

- **Multi-objective optimization** with user-defined targets and limits
- **Distributed evaluation** support through IPC, TCP, or HTTP
- **Adaptive sampling** strategies that focus on promising regions
- **Flexible parameter types**: continuous, integer, categorical, and lattice-valued
- **Comparison groups** for handling incomparable objectives
- **Priority-weighted scoring** for balancing multiple objectives

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

```python
from hola import HOLA, SystemConfig
from hola.core.coordinator import OptimizationCoordinator
from hola.core.samplers import ExploreExploitSampler, SobolSampler, ClippedGaussianMixtureSampler

# Define your parameters
parameters = {
    "x": {"tag": "continuous", "min": 0.0, "max": 10.0},
    "y": {"tag": "continuous", "min": 0.0, "max": 10.0},
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

# Create samplers for exploration and exploitation
explore_sampler = SobolSampler(dimension=2)
exploit_sampler = ClippedGaussianMixtureSampler(dimension=2, n_components=2)

# Create an explore-exploit sampler (combines exploration and exploitation)
sampler = ExploreExploitSampler(
    explore_sampler=explore_sampler,
    exploit_sampler=exploit_sampler,
    min_explore_samples=10,
    min_fit_samples=5
)

# Create coordinator
coordinator = OptimizationCoordinator.from_dict(
    hypercube_sampler=sampler,
    objectives_dict=objectives,
    parameters_dict=parameters
)

# Define your evaluation function
def evaluate(x: float, y: float) -> dict[str, float]:
    f1 = x**2 + y**2
    f2 = (x-2)**2 + (y-2)**2
    return {"f1": f1, "f2": f2}

# Configure the system
config = SystemConfig(
    local_workers=4,           # Number of local workers
    use_ipc_ratio=0.5,         # Half use IPC, half use TCP
    server_host="localhost",   # Server host address
    server_port=8000,          # Server port
    timeout=60                 # Optional timeout in seconds
)

# Run the optimization using context manager
with HOLA(coordinator, evaluate, config) as system:
    # Add more workers if needed during optimization
    system.add_workers(2, use_ipc=True)    # Add 2 more IPC workers
    system.add_workers(1, use_ipc=False)   # Add 1 more TCP worker

    # Wait for optimization to complete
    system.wait_until_complete()

    # Get final results
    result = system.get_final_state()

print(f"Best parameters: {result.best_result.parameters}")
print(f"Best objectives: {result.best_result.objectives}")
```

## Distributed Mode

HOLA provides a robust distributed optimization system with the following components:

### Architecture

The distributed system consists of three main components:

1. **Scheduler**: Manages parameter suggestion and result collection
2. **Workers**: Evaluate parameters and report results
3. **REST API Server**: Provides HTTP access for remote clients
4. **Monitoring Dashboard**: Visualizes optimization progress

### Running a Distributed Optimization

#### Using Command-Line Tools

1. Start the optimization server:
   ```bash
   bin/hola-server --host 0.0.0.0 --port 8000
   ```

2. Launch the monitoring dashboard:
   ```bash
   bin/hola-monitor --server-url http://localhost:8000 --port 8050
   ```

3. Connect workers to the server:
   ```python
   from hola import HOLA, SystemConfig

   # Define your evaluation function
   def evaluate(x, y):
       return {"objective": x**2 + y**2}

   # Configure the system
   config = SystemConfig(
       local_workers=8,        # Number of local workers
       use_ipc_ratio=0.5,      # Half use IPC, half use TCP
       server_host="localhost",
       server_port=8000,
       timeout=60
   )

   # Run the optimization using context manager
   with HOLA(coordinator, evaluate, config) as system:
       system.wait_until_complete()
       result = system.get_final_state()
   ```

#### Using the API Directly

```python
from hola import SchedulerProcess, Server, LocalWorker, start_workers, setup_logging

# Setup logging
logger = setup_logging("OptimizationSystem")

# Start scheduler
scheduler = SchedulerProcess(coordinator)
scheduler_process = mp.Process(target=scheduler.run)
scheduler_process.start()

# Start server
server = Server(host="0.0.0.0", port=8000)
server.start()

# Start workers
workers = start_workers(
    eval_function=evaluate_function,
    num_workers=4,
    server_url="http://localhost:8000",
    use_ipc=True
)

# When done, clean up
stop_workers(workers)
server.stop()
scheduler_process.terminate()
scheduler_process.join()
```

### Communication Protocols

HOLA supports multiple communication protocols:

- **IPC**: For fastest communication between processes on the same machine
- **TCP**: For network communication between machines
- **HTTP**: For web-based clients and language-agnostic workers

### Worker Implementation

Workers can be implemented in any language that supports HTTP requests:

- **Python Workers**: Use the built-in `LocalWorker` class
- **Custom Workers**: Implement the protocol in any language

The worker protocol is simple:
1. Request parameter suggestions from the server
2. Evaluate the parameters using your objective function
3. Submit the results back to the server

### Monitoring

The dashboard provides real-time monitoring of:

- Number of active workers
- Total evaluations performed
- Best objectives found so far
- Optimization progress over time
- Parameter space exploration visualization

## System Utilities

HOLA provides several utilities to help manage the optimization system:

### Centralized Logging

```python
from hola import setup_logging

# Create a logger with a specific name and level
logger = setup_logging(
    name="MyComponent",
    level="INFO",
    log_dir="/path/to/logs"  # Optional
)

# Use the logger
logger.info("Starting optimization")
logger.error("Something went wrong")
```

### Worker Management

```python
from hola import start_workers, stop_workers

# Start workers
workers = start_workers(
    eval_function=my_evaluation_function,
    num_workers=4,
    server_url="http://localhost:8000",
    use_ipc=True
)

# Stop workers when done
stop_workers(workers)
```

## Advanced Features

- **Dynamic reconfiguration**: Update parameter bounds or objective targets during optimization
- **Leaderboard maintenance**: Track and rank all trials for post-analysis
- **Custom samplers**: Implement your own sampling strategies
- **Monitoring and visualization**: Track optimization progress (experimental)

## Status

The core optimization functionality is stable and well-tested. The distributed server mode is functional but still undergoing refinement. The web-based dashboard visualization is currently in experimental status.
