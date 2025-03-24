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
from hola import run_optimization_system
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

# TODO: Run the optimization system
# Run the optimization for a specified number of iterations
n_iterations = 100
for i in range(n_iterations):
    # Get parameter suggestions
    params_list, metadata = coordinator.suggest_parameters()

    # Evaluate objectives for each parameter set
    for params in params_list:
        objectives = evaluate(**params)

        # Record the evaluation results with metadata
        coordinator.record_evaluation(params, objectives, metadata)

    # Optional: Print progress update
    if (i + 1) % 10 == 0:
        print(f"Completed {i + 1}/{n_iterations} iterations")

    # Get current best trial
    best_trial = coordinator.get_best_trial()
    if best_trial:
        print(f"Current best parameters: {best_trial.parameters}")
        print(f"Current best objectives: {best_trial.objectives}")

# Get final result
result = coordinator.get_best_trial()

print(f"Best parameters: {result.parameters}")
print(f"Best objectives: {result.objectives}")
```
