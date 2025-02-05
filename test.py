import logging
import numpy as np
from hola.hola import run_optimization

# Define evaluation function
def evaluate_model(**params):
    """Simple test function that simulates ML model evaluation."""
    learning_rate = params["learning_rate"]
    batch_size = params["batch_size"]
    hidden_size = params["hidden_size"]

    # Simulate some relationship between parameters and objectives
    accuracy = 1.0 - np.exp(-learning_rate * batch_size / 50)
    latency = batch_size * hidden_size * (1 + 1/learning_rate) / 1000
    memory = hidden_size * batch_size / 1024  # MB

    # Add some noise
    accuracy += np.random.normal(0, 0.05)
    latency *= (1 + np.random.normal(0, 0.1))
    memory *= (1 + np.random.normal(0, 0.05))

    return {
        "accuracy": float(np.clip(accuracy, 0, 1)),
        "latency": float(np.clip(latency, 0, np.inf)),
        "memory": float(np.clip(memory, 0, np.inf))
    }

# Define objectives configuration
objectives_config = {
    "accuracy": {
        "target": 1.0,
        "limit": 0.0,
        "direction": "maximize",
        "priority": 1.0
    },
    "latency": {
        "target": 0.0,
        "limit": 1000.0,  # 1000ms
        "direction": "minimize",
        "priority": 0.5
    },
    "memory": {
        "target": 0.0,
        "limit": 2048.0,  # 2GB
        "direction": "minimize",
        "priority": 0.3
    }
}

# Define parameters configuration
parameters_config = {
    "learning_rate": {
        "type": "continuous",
        "min": 1e-4,
        "max": 1e-1,
        "scale": "log"
    },
    "batch_size": {
        "type": "integer",
        "min": 16,
        "max": 256
    },
    "hidden_size": {
        "type": "lattice",
        "min": 64,
        "max": 1024,
        "num_values": 10
    }
}

# Run optimization
best_result = run_optimization(
    evaluation_fn=evaluate_model,
    objectives_config=objectives_config,
    parameters_config=parameters_config,
    max_evaluations=100,
    n_workers=4,  # Use 4 parallel workers
    use_exploit=True,  # Enable exploration-exploitation
    log_level=logging.DEBUG  # Enable detailed logging
)

print("Best result found:")
print(f"Parameters: {best_result.parameters}")
print(f"Objectives: {best_result.objectives}")