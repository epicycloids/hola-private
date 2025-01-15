import asyncio
import os
from typing import Any

import numpy as np
import pytest

from hola.core.coordinator import OptimizationCoordinator
from hola.core.leaderboard import Leaderboard, Trial
from hola.core.objectives import ObjectiveConfig, ObjectiveScorer
from hola.core.parameters import (
    CategoricalParameterConfig,
    ContinuousParameterConfig,
    IntegerParameterConfig,
    LatticeParameterConfig,
    ParameterTransformer,
    Scale,
)
from hola.core.samplers import (
    ClippedGaussianMixtureSampler,
    ExploreExploitSampler,
    SobolSampler,
    UniformSampler,
)
from hola.server.messages.base import Evaluation

# Example objective configurations for testing
objective_configs = {
    "obj1": ObjectiveConfig(target=0.0, limit=1.0, direction="minimize"),
    "obj2": ObjectiveConfig(target=10.0, limit=0.0, direction="maximize"),
}

# Example parameter configurations for testing
parameter_configs = {
    "param1": ContinuousParameterConfig(min=0.0, max=1.0, scale=Scale.LINEAR),
    "param2": IntegerParameterConfig(min=0, max=10),
    "param3": CategoricalParameterConfig(categories=["A", "B", "C"]),
    "param4": LatticeParameterConfig(min=0.0, max=1.0, num_values=5),
}

@pytest.fixture
def coordinator() -> OptimizationCoordinator:
    """Fixture to create an OptimizationCoordinator instance for testing."""
    explore_sampler = UniformSampler(dimension=4)
    exploit_sampler = ClippedGaussianMixtureSampler(
        dimension=4, n_components=2, hypercube_sampler=UniformSampler(dimension=5)
    )
    ee_sampler = ExploreExploitSampler(
        explore_sampler=explore_sampler, exploit_sampler=exploit_sampler
    )
    return OptimizationCoordinator(
        objective_config=objective_configs,
        parameter_config=parameter_configs,
        hypercube_sampler=ee_sampler,
    )

@pytest.mark.asyncio
async def test_end_to_end_optimization(coordinator: OptimizationCoordinator, tmp_path: str):
    """End-to-end test for the optimization process."""
    # 1. Initial parameter suggestions
    initial_params = await coordinator.suggest_parameters(n_samples=5)
    assert len(initial_params) == 5
    assert all(len(p) == 4 for p in initial_params)  # 4 parameters

    # 2. Record evaluations
    evaluations = [
        Evaluation(
            objectives={"obj1": 0.5, "obj2": 5.0},
            parameters=params,
        )
        for i, params in enumerate(initial_params)
    ]
    for eval in evaluations:
        await coordinator.record_evaluation(eval)

    # 3. Check leaderboard
    best_result = await coordinator.get_best_result()
    assert best_result is not None
    assert "obj1" in best_result.objectives
    assert "param1" in best_result.parameters

    # 4. Update objective configuration
    new_objective_config = {
        "obj1": ObjectiveConfig(target=0.2, limit=1.2, direction="minimize"),
        "obj2": ObjectiveConfig(target=8.0, limit=2.0, direction="maximize"),
    }
    await coordinator.update_objective_config(new_objective_config)

    # 5. Update parameter configuration
    new_parameter_config = {
        "param1": ContinuousParameterConfig(min=-1.0, max=2.0, scale=Scale.LINEAR),
        "param2": IntegerParameterConfig(min=1, max=8),
        "param3": CategoricalParameterConfig(categories=["A", "B", "C", "D"]),
        "param4": LatticeParameterConfig(min=-1.0, max=1.0, num_values=10),
    }
    await coordinator.update_parameter_config(new_parameter_config)

    # 6. Suggest new parameters after updates
    new_params = await coordinator.suggest_parameters(n_samples=3)
    assert len(new_params) == 3
    assert all(len(p) == 4 for p in new_params)

    # 7. Record more evaluations
    new_evaluations = [
        Evaluation(
            objectives={"obj1": 0.3, "obj2": 7.0},
            parameters=params,
        )
        for i, params in enumerate(new_params)
    ]
    for eval in new_evaluations:
        await coordinator.record_evaluation(eval)

    # 8. Pause and resume
    await coordinator.pause()
    assert not coordinator.is_active()
    await coordinator.resume()
    assert coordinator.is_active()

    # 9. Save and load state
    save_path = os.path.join(tmp_path, "state.joblib")
    await coordinator.save_state(save_path)

    new_coordinator = OptimizationCoordinator(
        objective_config=objective_configs,  # Provide dummy configs for initialization
        parameter_config=parameter_configs,
        hypercube_sampler=ExploreExploitSampler(
            explore_sampler=UniformSampler(dimension=4),
            exploit_sampler=ClippedGaussianMixtureSampler(
                dimension=4, n_components=2, hypercube_sampler=UniformSampler(dimension=5)
            ),
        ),
    )
    await new_coordinator.load_state(save_path)

    # 10. Verify loaded state
    assert new_coordinator.is_active()
    loaded_best_result = await new_coordinator.get_best_result()
    assert loaded_best_result is not None
    assert np.allclose(
        loaded_best_result.objectives["obj1"], best_result.objectives["obj1"]
    )  # Compare objective values