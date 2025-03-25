#!/usr/bin/env python

"""
Demonstration of visualization methods for HOLA optimization leaderboards.

This script shows how to use the visualization methods added to the Leaderboard class:
1. plot_parameters - Visualize sampled points in parameter space
2. plot_objectives - Visualize tradeoff curves in objective space
3. plot_comparison_groups - Visualize tradeoff curves in comparison group space (2D)
4. plot_comparison_groups_3d - Visualize tradeoff surfaces in comparison group space (3D)
5. plot_objective_vs_trial - Visualize objective values vs trial ID

The script sets up a simple multi-objective optimization problem with mixed parameter types,
runs a few iterations, and then creates various visualizations of the results.
"""

import os
import numpy as np
from pathlib import Path

from hola.core.objectives import ObjectiveScorer
from hola.core.leaderboard import Leaderboard, Trial


def main():
    # Create output directory
    output_dir = Path("visualization_output")
    output_dir.mkdir(exist_ok=True)

    # 1. Define objectives
    objectives_dict = {
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
            "target": 10.0,
            "limit": 0.0,
            "direction": "maximize",
            "priority": 1.0,
            "comparison_group": 1
        }
    }
    objective_scorer = ObjectiveScorer.from_dict(objectives_dict)

    # 2. Create leaderboard
    leaderboard = Leaderboard(objective_scorer)

    # 3. Generate some sample trials
    np.random.seed(42)  # For reproducibility
    num_trials = 50

    # Define categorical options
    categories = ["A", "B", "C", "D"]

    for i in range(num_trials):
        # Sample parameters randomly
        x = np.random.uniform(0, 10)
        y = np.random.uniform(0, 10)
        category = np.random.choice(categories)
        integer = np.random.randint(1, 6)

        # Calculate objective values
        f1 = (x - 2) ** 2 + (y - 2) ** 2  # Minimization objective
        f2 = (x - 8) ** 2 + (y - 8) ** 2  # Minimization objective
        f3 = x + y  # Maximization objective

        # Some parameter combinations are infeasible
        is_feasible = not (x > 8 and y > 8 and category == "D")

        # Create trial and add to leaderboard
        trial = Trial(
            trial_id=i,
            parameters={"x": x, "y": y, "category": category, "integer": integer},
            objectives={"f1": f1, "f2": f2, "f3": f3},
            is_feasible=is_feasible,
            metadata={"iteration": i // 10}
        )
        leaderboard.add(trial)

    # 4. Create and save visualizations

    # Parameter space visualization
    fig_params_xy = leaderboard.plot_parameters("x", "y")
    fig_params_xy.write_html(output_dir / "parameter_space_xy.html")

    # Parameter space with categorical parameter
    fig_params_cat = leaderboard.plot_parameters("category", "x")
    fig_params_cat.write_html(output_dir / "parameter_space_category_x.html")

    # Single parameter visualization
    fig_params_single = leaderboard.plot_parameters("x")
    fig_params_single.write_html(output_dir / "parameter_space_x.html")

    # Another visualization with integer parameter
    fig_params_int = leaderboard.plot_parameters("integer", "y")
    fig_params_int.write_html(output_dir / "parameter_space_integer_y.html")

    # Objective space visualization
    fig_obj_f1f2 = leaderboard.plot_objectives("f1", "f2")
    fig_obj_f1f2.write_html(output_dir / "objective_space_f1_f2.html")

    fig_obj_f1f3 = leaderboard.plot_objectives("f1", "f3")
    fig_obj_f1f3.write_html(output_dir / "objective_space_f1_f3.html")

    # Comparison group space visualization
    try:
        fig_groups = leaderboard.plot_comparison_groups(0, 1)
        fig_groups.write_html(output_dir / "comparison_groups.html")
    except ValueError as e:
        print(f"Note: {e}")

    # Try to create 3D comparison group visualization (requires 3+ groups)
    try:
        fig_groups_3d = leaderboard.plot_comparison_groups_3d(0, 1, 2)
        fig_groups_3d.write_html(output_dir / "comparison_groups_3d.html")
    except ValueError as e:
        print(f"Note: {e}")

    # Objective vs trial visualization
    fig_obj_trial_f1 = leaderboard.plot_objective_vs_trial("f1")
    fig_obj_trial_f1.write_html(output_dir / "objective_vs_trial_f1.html")

    fig_obj_trial_f3 = leaderboard.plot_objective_vs_trial("f3")
    fig_obj_trial_f3.write_html(output_dir / "objective_vs_trial_f3.html")

    print(f"Visualizations saved to {output_dir.absolute()}")

    # Print some statistics
    print(f"Total trials: {leaderboard.get_total_count()}")
    print(f"Feasible trials: {leaderboard.get_feasible_count()}")
    print(f"Ranked trials: {leaderboard.get_ranked_count()}")

    # Show best trial
    best_trial = leaderboard.get_best_trial()
    if best_trial:
        print("\nBest trial:")
        print(f"  Trial ID: {best_trial.trial_id}")
        print(f"  Parameters: {best_trial.parameters}")
        print(f"  Objectives: {best_trial.objectives}")


if __name__ == "__main__":
    main()