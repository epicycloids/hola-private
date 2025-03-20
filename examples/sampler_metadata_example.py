"""
Example demonstrating the use of sampler metadata in HOLA.

This example shows how metadata from samplers is captured and recorded with
trials, allowing for analysis of how different sampling strategies perform.
"""

from hola.core.coordinator import OptimizationCoordinator
from hola.core.samplers import ExploreExploitSampler, SobolSampler, ClippedGaussianMixtureSampler

# Define a simple test function (simple quadratic function with 2 variables)
def objective_function(x, y):
    f1 = (x - 2) ** 2 + (y - 3) ** 2  # Minimum at (2, 3)
    f2 = (x - 5) ** 2 + (y - 1) ** 2  # Minimum at (5, 1)
    f3 = x ** 2 + y ** 2               # Minimum at (0, 0)
    return {"f1": f1, "f2": f2, "f3": f3}

def main():
    # Define parameters
    parameters = {
        "x": {"type": "continuous", "min": -10.0, "max": 10.0},
        "y": {"type": "continuous", "min": -10.0, "max": 10.0},
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

    # Create explore-exploit sampler
    sampler = ExploreExploitSampler(
        explore_sampler=explore_sampler,
        exploit_sampler=exploit_sampler,
    )

    # Create coordinator
    coordinator = OptimizationCoordinator.from_dict(
        hypercube_sampler=sampler,
        objectives_dict=objectives,
        parameters_dict=parameters,
        top_frac=0.2,
        minimum_fit_samples=10,  # Set lower to see exploitation kick in faster
    )

    # Run optimization
    n_iterations = 50
    for i in range(n_iterations):
        print(f"\nIteration {i+1}/{n_iterations}")

        # Get parameter suggestions with metadata
        params_list, metadata = coordinator.suggest_parameters(n_samples=2)

        # Print sampler info
        phase = metadata.get("phase", "unknown")
        sampler_class = metadata.get("sampler_class", "unknown")
        print(f"Sampling phase: {phase}, Sampler: {sampler_class}")

        # Evaluate objectives for each parameter set
        for params in params_list:
            print(f"Evaluating parameters: {params}")
            objectives = objective_function(**params)

            # Record evaluation with metadata
            coordinator.record_evaluation(params, objectives, metadata)

        # Get best trial
        best_trial = coordinator.get_best_trial()
        if best_trial:
            print(f"Best trial so far: {best_trial.parameters}")
            print(f"Best objectives: {best_trial.objectives}")

    # Print full results
    print("\nFinal Leaderboard:")
    df = coordinator.leaderboard.get_dataframe()
    print(f"DataFrame columns: {df.columns}")

    # Get metadata as a separate DataFrame
    print("\nMetadata DataFrame:")
    metadata_df = coordinator.leaderboard.get_metadata()
    print(f"Metadata columns: {metadata_df.columns}")

    # Metadata Analysis
    if not metadata_df.empty:
        print("\nMetadata Analysis:")

        # Analyze performance by sampler type
        if 'phase' in metadata_df.columns:
            print("\nPerformance by sampling phase:")
            for phase in metadata_df['phase'].unique():
                # Get trial IDs for this phase
                phase_trials = metadata_df[metadata_df['phase'] == phase].index.tolist()

                # Get the corresponding leaderboard entries
                phase_df = df[df['Trial'].isin(phase_trials)]

                print(f"Phase: {phase}")
                print(f"  Number of trials: {len(phase_df)}")

                # Performance analysis for each objective
                for obj in ['f1', 'f2', 'f3']:
                    if obj in df.columns:
                        print(f"  Average {obj}: {phase_df[obj].mean():.4f}")

                # Group scores analysis
                for col in df.columns:
                    if isinstance(col, str) and 'Group' in col and 'Score' in col:
                        print(f"  Average {col}: {phase_df[col].mean():.4f}")

        # Inner sampler analysis
        if 'inner_sampler_sampler_type' in metadata_df.columns:
            print("\nInner sampler types used:")
            inner_types = metadata_df['inner_sampler_sampler_type'].unique()
            for inner_type in inner_types:
                inner_count = len(metadata_df[metadata_df['inner_sampler_sampler_type'] == inner_type])
                print(f"  {inner_type}: {inner_count} trials")

        # GMM component analysis when present
        if 'inner_sampler_components_used' in metadata_df.columns:
            exploit_meta = metadata_df[metadata_df['phase'] == 'exploit']
            if not exploit_meta.empty and 'inner_sampler_components_used' in exploit_meta.columns:
                print("\nGMM component usage in exploitation phase:")
                # This might need custom processing based on the actual format of components_used
                components = exploit_meta['inner_sampler_components_used']
                if not components.empty:
                    print(f"  Components used: {components.iloc[0]}")

        # Find the best trial and display its metadata
        if not df.empty:
            # Find the best trial based on Group 0 Score
            for col in df.columns:
                if isinstance(col, str) and 'Group 0 Score' in col:
                    best_idx = df[col].idxmin()
                    best_trial_id = df.iloc[best_idx]['Trial']

                    print("\nBest trial metadata:")
                    if best_trial_id in metadata_df.index:
                        for col in metadata_df.columns:
                            print(f"  {col}: {metadata_df.loc[best_trial_id, col]}")

                    print("\nBest trial details:")
                    print(f"  Parameters: x={df.iloc[best_idx]['x']:.4f}, y={df.iloc[best_idx]['y']:.4f}")
                    for obj in ['f1', 'f2', 'f3']:
                        if obj in df.columns:
                            print(f"  {obj}: {df.iloc[best_idx][obj]:.4f}")
                    break

if __name__ == "__main__":
    main()