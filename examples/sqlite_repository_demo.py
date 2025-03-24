#!/usr/bin/env python
"""
Example of using the SQLite repository with HOLA optimization.

This example demonstrates:
1. Creating an optimization with SQLite persistence
2. Running multiple optimization trials with automatic persistence
3. Visualizing results from the SQLite database
4. Reloading an optimization from the database
"""

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sqlite3

from hola.core.samplers import ExploreExploitSampler, SobolSampler, ClippedGaussianMixtureSampler
from hola.core.coordinator import OptimizationCoordinator
from hola.core.repository import SQLiteTrialRepository


def main():
    # Create output directory
    os.makedirs("output", exist_ok=True)

    # Define database path
    db_path = "output/optimization.db"

    # Remove existing database if it exists
    if os.path.exists(db_path):
        os.remove(db_path)

    # Define parameters
    parameters = {
        "x": {"type": "continuous", "min": -10.0, "max": 10.0},
        "y": {"type": "continuous", "min": -10.0, "max": 10.0},
    }

    # Define objectives (multi-group example)
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
        "f3": {
            "target": 0.0,
            "limit": 100.0,
            "direction": "minimize",
            "priority": 0.5,
            "comparison_group": 1
        },
    }

    # Create samplers for exploration and exploitation
    explore_sampler = SobolSampler(dimension=2)
    exploit_sampler = ClippedGaussianMixtureSampler(dimension=2, n_components=2)

    # Create an explore-exploit sampler (combines exploration and exploitation)
    sampler = ExploreExploitSampler(
        explore_sampler=explore_sampler,
        exploit_sampler=exploit_sampler,
    )

    # Create coordinator with SQLite repository
    print("Creating optimization coordinator with SQLite persistence...")
    coordinator = OptimizationCoordinator.from_dict(
        hypercube_sampler=sampler,
        objectives_dict=objectives,
        parameters_dict=parameters,
        db_path=db_path  # This will create a SQLiteTrialRepository automatically
    )

    # Define evaluation function (multi-objective test function)
    def evaluate(x: float, y: float) -> dict[str, float]:
        """Simple multi-objective test function"""
        f1 = x**2 + y**2  # Center at origin
        f2 = (x-2)**2 + (y-2)**2  # Center at (2,2)
        f3 = (x-4)**2 + (y-4)**2  # Center at (4,4)
        return {"f1": f1, "f2": f2, "f3": f3}

    # Run optimization (100 iterations)
    print("\nRunning optimization with 100 trials...")
    for i in range(100):
        params_list, metadata = coordinator.suggest_parameters()
        for params in params_list:
            objectives = evaluate(**params)
            coordinator.record_evaluation(params, objectives, metadata)

        # Print progress every 10 iterations
        if (i+1) % 10 == 0:
            print(f"Progress: {i+1}/100 iterations")

    # Print statistics about the optimization
    print("\nOptimization statistics:")
    print(f"Total trials: {coordinator.get_total_evaluations()}")
    print(f"Feasible trials: {coordinator.get_feasible_count()}")
    print(f"Ranked trials: {coordinator.get_ranked_count()}")

    # Get the best trial
    best_trial = coordinator.get_best_trial()
    print(f"\nBest trial: {best_trial.trial_id}")
    print(f"Parameters: {best_trial.parameters}")
    print(f"Objectives: {best_trial.objectives}")

    # Demonstrate direct access to SQLite database
    print("\nDemonstrating direct access to SQLite database...")
    with sqlite3.connect(db_path) as conn:
        # Count trials
        cursor = conn.execute("SELECT COUNT(*) FROM trials")
        count = cursor.fetchone()[0]
        print(f"Trials in database: {count}")

        # Get trial parameters for a specific trial
        cursor = conn.execute("""
            SELECT name, value FROM parameters
            WHERE trial_id = ?
        """, (best_trial.trial_id,))
        params = cursor.fetchall()
        print(f"Parameters for best trial from database:")
        for name, value in params:
            print(f"  {name}: {value}")

    # Show how to create a visualization from the database
    print("\nCreating visualization from stored trials...")
    df = coordinator.get_all_trials_dataframe()

    # Create scatter plot of parameter space with points colored by objective value
    plt.figure(figsize=(10, 8))
    plt.scatter(df['x'], df['y'], c=df['f1'], cmap='viridis', alpha=0.7)
    plt.colorbar(label='f1 value')
    plt.title('Parameter Space Exploration')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('output/parameter_space.png')
    print("Saved visualization to output/parameter_space.png")

    # Save coordinator state
    print("\nSaving coordinator state...")
    coordinator.save_to_file("output/coordinator_state.json")

    # Demonstrate reloading from database
    print("\nReloading optimization from database...")

    # Create a new repository connected to the same database
    repository = SQLiteTrialRepository(db_path)

    # Load coordinator with the repository
    loaded_coordinator = OptimizationCoordinator.load_from_file(
        "output/coordinator_state.json",
        repository=repository
    )

    # Verify trials were loaded correctly
    print(f"Loaded coordinator has {loaded_coordinator.get_total_evaluations()} trials")
    print(f"Loaded ranked trials: {loaded_coordinator.get_ranked_count()}")

    # Run a few more iterations with the loaded coordinator
    print("\nRunning 10 more iterations with loaded coordinator...")
    for i in range(10):
        params_list, metadata = loaded_coordinator.suggest_parameters()
        for params in params_list:
            objectives = evaluate(**params)
            loaded_coordinator.record_evaluation(params, objectives, metadata)

    print(f"Total trials after additional iterations: {loaded_coordinator.get_total_evaluations()}")

    print("\nDemo complete! The optimization database is stored at:")
    print(f"  {os.path.abspath(db_path)}")
    print("You can connect to this database directly using any SQLite client.")


if __name__ == "__main__":
    main()