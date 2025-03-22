"""
Test the save and load functionality for the OptimizationCoordinator.

This script verifies that the serialize/deserialize functionality works correctly
by:
1. Creating an optimization coordinator
2. Running some trials
3. Saving the coordinator to disk
4. Loading it back
5. Verifying that the loaded coordinator behaves the same way
"""

import os
import tempfile

from hola.core.coordinator import OptimizationCoordinator
from hola.core.samplers import ExploreExploitSampler, SobolSampler, ClippedGaussianMixtureSampler


def test_save_load():
    """Test the save and load functionality for the OptimizationCoordinator."""

    print("Creating test optimization coordinator...")

    # Define parameters
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
        "f3": {
            "target": 0.0,
            "limit": 100.0,
            "direction": "minimize",
            "priority": 0.5,
            "comparison_group": 1
        },
    }

    # Create samplers
    explore_sampler = SobolSampler(dimension=2)
    exploit_sampler = ClippedGaussianMixtureSampler(dimension=2, n_components=2)

    # Create sampler
    sampler = ExploreExploitSampler(
        explore_sampler=explore_sampler,
        exploit_sampler=exploit_sampler,
    )

    # Create coordinator
    coordinator = OptimizationCoordinator.from_dict(
        hypercube_sampler=sampler,
        objectives_dict=objectives,
        parameters_dict=parameters
    )

    # Define evaluation function
    def evaluate(x: float, y: float) -> dict[str, float]:
        f1 = x**2 + y**2
        f2 = (x-2)**2 + (y-2)**2
        f3 = (x-4)**2 + (y-4)**2
        return {"f1": f1, "f2": f2, "f3": f3}

    print("Running 50 trials...")
    # Run 50 trials
    for i in range(50):
        params_list, metadata = coordinator.suggest_parameters()
        for params in params_list:
            objectives = evaluate(**params)
            coordinator.record_evaluation(params, objectives, metadata)

    # Get statistics before saving
    pre_save_stats = {
        "total_trials": coordinator.get_total_evaluations(),
        "feasible_trials": coordinator.get_feasible_count(),
        "ranked_trials": coordinator.get_ranked_count(),
        "top_trial_id": coordinator.get_best_trial().trial_id if coordinator.get_best_trial() else None,
    }

    print(f"Pre-save statistics: {pre_save_stats}")

    # Get sampler state before saving
    pre_save_sampler_state = coordinator.hypercube_sampler.get_state()
    print("Obtained pre-save sampler state")

    # Create temporary directory for saving
    with tempfile.TemporaryDirectory() as temp_dir:
        save_path = os.path.join(temp_dir, "optimization.json")

        print(f"Saving coordinator to {save_path}...")
        # Save the coordinator
        coordinator.save_to_file(save_path)

        print("Loading coordinator from saved file...")
        # Load the coordinator
        loaded_coordinator = OptimizationCoordinator.load_from_file(save_path)

        # Get sampler state after loading
        post_load_sampler_state = loaded_coordinator.hypercube_sampler.get_state()
        print("Obtained post-load sampler state")

        # Verify that sampler states match
        assert pre_save_sampler_state == post_load_sampler_state, "Sampler state doesn't match after loading"
        print("Sampler state successfully verified!")

        # Get statistics after loading
        post_load_stats = {
            "total_trials": loaded_coordinator.get_total_evaluations(),
            "feasible_trials": loaded_coordinator.get_feasible_count(),
            "ranked_trials": loaded_coordinator.get_ranked_count(),
            "top_trial_id": loaded_coordinator.get_best_trial().trial_id if loaded_coordinator.get_best_trial() else None,
        }

        print(f"Post-load statistics: {post_load_stats}")

        # Verify that statistics match
        assert pre_save_stats == post_load_stats, "Statistics don't match after loading"

        print("Running 10 more trials with loaded coordinator...")
        # Run 10 more trials with the loaded coordinator
        for i in range(10):
            params_list, metadata = loaded_coordinator.suggest_parameters()
            for params in params_list:
                objectives = evaluate(**params)
                loaded_coordinator.record_evaluation(params, objectives, metadata)

        # Check that the loaded coordinator can continue optimization
        final_stats = {
            "total_trials": loaded_coordinator.get_total_evaluations(),
            "feasible_trials": loaded_coordinator.get_feasible_count(),
            "ranked_trials": loaded_coordinator.get_ranked_count(),
        }

        print(f"Final statistics after additional trials: {final_stats}")

        # Verify that trials were added
        assert final_stats["total_trials"] > post_load_stats["total_trials"], "No new trials were added"

        print("Save/load test completed successfully!")


if __name__ == "__main__":
    test_save_load()