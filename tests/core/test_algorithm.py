"""Tests for HOLA optimization algorithm."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from hola.core.algorithm import HOLA, SampleStorageMode
from hola.core.objective import Direction, ObjectiveConfig, ObjectiveName, create_objective
from hola.core.params import (
    ContinuousParamConfig,
    IntegerParamConfig,
    ParamConfig,
    ParamName,
    Scale,
    create_param_config,
)


@pytest.fixture
def simple_params() -> dict[ParamName, ParamConfig]:
    """Simple parameter space with one continuous parameter."""
    return {
        ParamName("lr"): ContinuousParamConfig(
            min=1e-4,
            max=1e-1,
            scale=Scale.LOG,
        )
    }


@pytest.fixture
def complex_params() -> dict[ParamName, ParamConfig]:
    """More complex parameter space with multiple parameters."""
    return {
        ParamName("lr"): ContinuousParamConfig(
            min=1e-4,
            max=1e-1,
            scale=Scale.LOG,
        ),
        ParamName("batch_size"): IntegerParamConfig(
            min=16,
            max=256,
        ),
    }


@pytest.fixture
def single_objective() -> dict[ObjectiveName, ObjectiveConfig]:
    """Single objective configuration."""
    return {
        ObjectiveName("accuracy"): ObjectiveConfig(
            target=0.95,
            limit=0.80,
            direction=Direction.MAXIMIZE,
        )
    }


@pytest.fixture
def dual_objective() -> dict[ObjectiveName, ObjectiveConfig]:
    """Dual objective configuration with priorities."""
    return {
        ObjectiveName("accuracy"): ObjectiveConfig(
            target=0.95,
            limit=0.80,
            direction=Direction.MAXIMIZE,
            priority=2.0,
        ),
        ObjectiveName("time"): ObjectiveConfig(
            target=60.0,
            limit=300.0,
            direction=Direction.MINIMIZE,
            priority=1.0,
        ),
    }


@pytest.fixture
def pareto_objective() -> dict[ObjectiveName, ObjectiveConfig]:
    """Multi-objective configuration for Pareto optimization."""
    return {
        ObjectiveName("accuracy"): ObjectiveConfig(
            target=0.95,
            limit=0.80,
            direction=Direction.MAXIMIZE,
            comparison_group=0,
        ),
        ObjectiveName("time"): ObjectiveConfig(
            target=60.0,
            limit=300.0,
            direction=Direction.MINIMIZE,
            comparison_group=1,
        ),
    }

@pytest.fixture
def optimizer(
    complex_params: dict[ParamName, ParamConfig],
    dual_objective: dict[ObjectiveName, ObjectiveConfig],
) -> HOLA:
    return HOLA(
        params_config=complex_params,
        objectives_config=dual_objective,
        min_samples=25,
        min_fit_samples=5,
    )


class TestHOLAInitialization:
    def test_valid_initialization(
        self,
        simple_params: dict[ParamName, ParamConfig],
        single_objective: dict[ObjectiveName, ObjectiveConfig],
    ) -> None:
        hola = HOLA(
            params_config=simple_params,
            objectives_config=single_objective,
        )
        assert hola._param_transformer is not None
        assert hola._objective_scorer is not None
        assert hola._leaderboard is not None
        assert hola._mixture_sampler is not None

    @pytest.mark.parametrize(
        "top_fraction",
        [-0.1, 0.0, 1.0, 1.1],  # Invalid values
    )
    def test_invalid_top_fraction(
        self,
        simple_params: dict[ParamName, ParamConfig],
        single_objective: dict[ObjectiveName, ObjectiveConfig],
        top_fraction: float,
    ) -> None:
        with pytest.raises(ValidationError):
            HOLA(
                params_config=simple_params,
                objectives_config=single_objective,
                top_fraction=top_fraction,
            )

    @pytest.mark.parametrize(
        "min_samples",
        [-1, 0],  # Invalid values
    )
    def test_invalid_min_samples(
        self,
        simple_params: dict[ParamName, ParamConfig],
        single_objective: dict[ObjectiveName, ObjectiveConfig],
        min_samples: int,
    ) -> None:
        with pytest.raises(ValidationError):
            HOLA(
                params_config=simple_params,
                objectives_config=single_objective,
                min_samples=min_samples,
            )


class TestHOLAOperation:

    def test_sampling(self, optimizer: HOLA) -> None:
        sample = optimizer.sample()
        assert isinstance(sample, dict)
        assert "lr" in sample
        assert "batch_size" in sample
        assert 1e-4 <= sample["lr"] <= 1e-1
        assert 16 <= sample["batch_size"] <= 256

    def test_add_results(self, optimizer: HOLA) -> None:
        params = optimizer.sample()
        optimizer.add(
            objectives={"accuracy": 0.90, "time": 120.0},
            params=params,
        )
        assert optimizer._leaderboard.num_samples() == 1

    def test_update_params_config(self, optimizer: HOLA) -> None:
        # Update learning rate bounds
        new_lr_config = create_param_config(
            "continuous",
            min=1e-5,
            max=1e-2,
            scale=Scale.LOG,
        )
        optimizer.update_params_config({"lr": new_lr_config})

        # Sample and verify new bounds
        for _ in range(10):
            sample = optimizer.sample()
            assert 1e-5 <= sample["lr"] <= 1e-2

    def test_update_objectives_config(self, optimizer: HOLA) -> None:
        # Update accuracy target
        new_accuracy_config = create_objective(
            "maximize",
            target=0.98,  # Changed from 0.95
            limit=0.80,
            priority=2.0,
        )
        optimizer.update_objectives_config({"accuracy": new_accuracy_config})

        # Add sample and verify scoring
        params = optimizer.sample()
        optimizer.add(
            objectives={"accuracy": 0.96, "time": 120.0},
            params=params,
        )

    def test_exploitation_transition(self, optimizer: HOLA) -> None:
        # Add enough samples to trigger exploitation
        for _ in range(12):  # More than min_samples
            params = optimizer.sample()
            optimizer.add(
                objectives={"accuracy": 0.90, "time": 120.0},
                params=params,
            )

        # Sample should now potentially use GMM
        optimizer.sample()  # This will use GMM if enough elite samples

    def test_get_dataframe(self, optimizer: HOLA) -> None:
        # Add some samples
        for _ in range(3):
            params = optimizer.sample()
            optimizer.add(
                objectives={"accuracy": 0.90, "time": 120.0},
                params=params,
            )

        df = optimizer.get_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "param_lr" in df.columns
        assert "param_batch_size" in df.columns
        assert "objective_accuracy" in df.columns
        assert "objective_time" in df.columns


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_parallel_optimization(
    n_jobs: int,
    complex_params: dict[ParamName, ParamConfig],
    dual_objective: dict[ObjectiveName, ObjectiveConfig],
) -> None:
    def objective_function(**params: Any) -> dict[str, float]:
        lr = params["lr"]
        batch_size = params["batch_size"]
        # Simple mock objective function
        accuracy = 0.8 + 0.1 * np.random.random()
        time = 100 + 50 * np.random.random()
        return {"accuracy": accuracy, "time": time}

    hola = HOLA.tune(
        func=objective_function,
        params=complex_params,
        objectives=dual_objective,
        num_runs=10,
        n_jobs=n_jobs,
        min_samples=25,
    )

    assert hola._leaderboard.num_samples() == 10
    df = hola.get_dataframe()
    assert len(df) == 10


@pytest.fixture
def optimizer_with_samples(optimizer: HOLA) -> HOLA:
    """Create optimizer with some samples for testing archiving."""
    params1 = {"lr": 1e-3, "batch_size": 32}
    params2 = {"lr": 1e-2, "batch_size": 64}

    optimizer.add(
        objectives={"accuracy": 0.90, "time": 120.0},
        params=params1,
    )
    optimizer.add(
        objectives={"accuracy": 0.92, "time": 140.0},
        params=params2,
    )

    return optimizer

class TestHOLAArchiving:
    def test_archive_discard_mode(self, optimizer_with_samples: HOLA) -> None:
        """Test parameter update with DISCARD mode."""
        initial_count = optimizer_with_samples.num_samples
        assert initial_count > 0  # Verify we have samples to start with

        # Update lr bounds to make samples infeasible
        new_lr_config = create_param_config(
            "continuous",
            min=0.1,  # Much higher than existing samples
            max=0.5,
            scale=Scale.LOG,
        )

        optimizer_with_samples.update_params_config(
            {"lr": new_lr_config},
            storage_mode=SampleStorageMode.DISCARD
        )

        # Check samples were discarded
        assert optimizer_with_samples.num_samples == 0
        assert optimizer_with_samples.num_archived_samples == 0

    def test_archive_store_mode(self, optimizer_with_samples: HOLA) -> None:
        """Test parameter update with ARCHIVE mode."""
        initial_count = optimizer_with_samples.num_samples
        assert initial_count > 0  # Verify we have samples to start with

        # Update lr bounds to make samples infeasible
        new_lr_config = create_param_config(
            "continuous",
            min=0.1,  # Much higher than existing samples
            max=0.5,
            scale=Scale.LOG,
        )

        optimizer_with_samples.update_params_config(
            {"lr": new_lr_config},
            storage_mode=SampleStorageMode.ARCHIVE
        )

        # Check samples were archived
        assert optimizer_with_samples.num_samples == 0
        assert optimizer_with_samples.num_archived_samples == initial_count

    def test_archive_recovery(self, optimizer_with_samples: HOLA) -> None:
        """Test recovery of archived samples when they become feasible again."""
        initial_count = optimizer_with_samples.num_samples
        assert initial_count > 0

        # First archive samples
        new_lr_config_1 = create_param_config(
            "continuous",
            min=0.1,
            max=0.5,
            scale=Scale.LOG,
        )

        optimizer_with_samples.update_params_config(
            {"lr": new_lr_config_1},
            storage_mode=SampleStorageMode.ARCHIVE
        )

        archived_count = optimizer_with_samples.num_archived_samples
        assert archived_count == initial_count
        assert optimizer_with_samples.num_samples == 0

        # Then make them feasible again
        new_lr_config_2 = create_param_config(
            "continuous",
            min=1e-4,
            max=1.0,
            scale=Scale.LOG,
        )

        optimizer_with_samples.update_params_config(
            {"lr": new_lr_config_2},
            storage_mode=SampleStorageMode.ARCHIVE
        )

        # Check samples were recovered
        assert optimizer_with_samples.num_samples == archived_count
        assert optimizer_with_samples.num_archived_samples == 0

    def test_sampler_state_after_update(self, optimizer_with_samples: HOLA) -> None:
        """Test sampler state is correctly updated after parameter changes."""
        # Generate enough samples to enable exploitation
        min_samples = optimizer_with_samples._mixture_sampler.min_explore_samples

        # Add enough diverse samples to trigger exploitation
        base_lr = 1e-3
        base_batch = 32
        for i in range(min_samples):
            # Add some variation to avoid GMM convergence warnings
            params = {
                "lr": base_lr * (1 + 0.1 * (i / min_samples)),
                "batch_size": base_batch + i * 8
            }
            objectives = {
                "accuracy": 0.90 + 0.01 * (i / min_samples),
                "time": 120.0 + 5 * (i / min_samples)
            }
            optimizer_with_samples.add(
                objectives=objectives,
                params=params,
            )

        # Verify we're in exploitation
        assert optimizer_with_samples._mixture_sampler.is_using_exploitation()

        # Make all samples infeasible
        new_lr_config = create_param_config(
            "continuous",
            min=0.1,
            max=0.5,
            scale=Scale.LOG,
        )

        optimizer_with_samples.update_params_config(
            {"lr": new_lr_config},
            storage_mode=SampleStorageMode.DISCARD
        )

        # Check sampler returned to exploration phase
        assert not optimizer_with_samples._mixture_sampler.is_using_exploitation()
        assert optimizer_with_samples._mixture_sampler.sample_count == 0


def test_error_handling(
    complex_params: dict[ParamName, ParamConfig],
    dual_objective: dict[ObjectiveName, ObjectiveConfig],
) -> None:
    hola = HOLA(
        params_config=complex_params,
        objectives_config=dual_objective,
        min_samples=25,
    )

    # Invalid parameter names
    with pytest.raises(KeyError):
        hola.add(
            objectives={"accuracy": 0.90, "time": 120.0},
            params={"invalid_param": 0.001},
        )

    # Invalid objective names
    with pytest.raises(KeyError):
        hola.add(
            objectives={"invalid_objective": 0.90},
            params={"lr": 0.001, "batch_size": 32},
        )

    # Non-existent save file
    with pytest.raises(FileNotFoundError):
        hola.load("nonexistent.csv")


if __name__ == "__main__":
    pytest.main([__file__])
