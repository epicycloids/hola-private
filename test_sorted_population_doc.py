"""Test script to run and verify examples from sorted_population.py docstrings."""

import numpy as np
from hola.core.sorted_population import ScalarSortedPopulation, VectorSortedPopulation

def print_example_output(example_name, actual_output, expected_output=None):
    """Print the results of running an example."""
    print(f"\n=== {example_name} ===")
    print("Actual output:", actual_output)
    if expected_output is not None:
        print("Expected output:", expected_output)
        if actual_output != expected_output:
            print("NOTE: Output differs from docstring")
    print()

# Example 1: Basic scalar sorting
print("Testing basic scalar sorting example...")
pop = ScalarSortedPopulation()
pop.add("sample1", np.array([0.5]))
pop.add("sample2", np.array([0.3]))
result = pop.get_sorted_labels()
print_example_output(
    "Basic Scalar Sorting",
    result,
    expected_output=["sample2", "sample1"]
)

# Example 2: Multi-group vector sorting
print("Testing multi-group vector sorting example...")
pop = VectorSortedPopulation()
pop.add("sample1", np.array([0.5, 0.3]))
pop.add("sample2", np.array([0.3, 0.4]))
front_0 = pop.get_front(0)
print_example_output(
    "Multi-group Vector Sorting - Front Size",
    len(front_0),
    expected_output=2
)

# Example 3: Fast/slow config example
print("Testing fast/slow config example...")
pop = VectorSortedPopulation()
# Add sample with 95% accuracy, 120s training time
pop.add("config1", np.array([0.95, 120.0]))
# Add sample with 90% accuracy, 60s training time
pop.add("config2", np.array([0.90, 60.0]))
# Check front 0 size
front_0 = pop.get_front(0)
print_example_output(
    "Fast/Slow Config - Front Size",
    len(front_0),
    expected_output=2
)

# Example 4: ScalarSortedPopulation score retrieval
print("Testing scalar score retrieval example...")
pop = ScalarSortedPopulation()
pop.add("config_1", np.array([120.0]))
score = pop.get_score("config_1")
print_example_output(
    "Scalar Score Retrieval",
    f"Training time: {score[0]:.1f}s",
    expected_output="Training time: 120.0s"
)

# Example 5: ScalarSortedPopulation score update
print("Testing scalar score update example...")
pop = ScalarSortedPopulation()
pop.add("config_1", np.array([120.0]))
pop.update_score("config_1", np.array([90.0]))
score = pop.get_score("config_1")
print_example_output(
    "Scalar Score Update",
    score[0],
    expected_output=90.0
)

# Example 6: ScalarSortedPopulation top samples
print("Testing scalar top samples example...")
pop = ScalarSortedPopulation()
pop.add("fast_config", np.array([60.0]))  # 60 seconds
pop.add("slow_config", np.array([120.0])) # 120 seconds
best = pop.get_top_samples(1)[0]
print_example_output(
    "Scalar Top Samples",
    best,
    expected_output="fast_config"
)

# Example 7: VectorSortedPopulation score retrieval
print("Testing vector score retrieval example...")
pop = VectorSortedPopulation()
pop.add("config_1", np.array([0.95, 120.0]))
score = pop.get_score("config_1")
print_example_output(
    "Vector Score Retrieval",
    f"Accuracy: {score[0]:.1%}, Training time: {score[1]:.1f}s",
    expected_output="Accuracy: 95.0%, Training time: 120.0s"
)

# Example 8: VectorSortedPopulation score update
print("Testing vector score update example...")
pop = VectorSortedPopulation()
pop.add("config_1", np.array([0.95, 120.0]))
pop.update_score("config_1", np.array([0.97, 90.0]))
score = pop.get_score("config_1")
print_example_output(
    "Vector Score Update",
    f"Accuracy: {score[0]:.1%}, Training time: {score[1]:.1f}s",
    expected_output="Accuracy: 97.0%, Training time: 90.0s"
)

# Example 9: VectorSortedPopulation Pareto front
print("Testing vector Pareto front example...")
pop = VectorSortedPopulation()
pop.add("config1", np.array([0.95, 120.0]))
pop.add("config2", np.array([0.90, 60.0]))
pareto_front = pop.get_front(0)
print_example_output(
    "Vector Pareto Front",
    f"Found {len(pareto_front)} Pareto-optimal configs",
    expected_output="Found 2 Pareto-optimal configs"
)

# Example 10: VectorSortedPopulation top samples
print("Testing vector top samples example...")
pop = VectorSortedPopulation()
pop.add("config1", np.array([0.95, 120.0]))
pop.add("config2", np.array([0.90, 60.0]))
best_configs = pop.get_top_samples(2)
output = []
for label in best_configs:
    score = pop.get_score(label)
    output.append(f"{label}: {score[0]:.1%} acc, {score[1]:.1f}s")
print_example_output(
    "Vector Top Samples",
    output
    # Note: No expected output specified in docstring
)

if __name__ == "__main__":
    print("\nAll examples have been run!")