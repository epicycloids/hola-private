"""Hyperparameter Optimization, Lightweight and Automatic (HOLA).

This module implements the core HOLA algorithm for hyperparameter optimization,
which uses an adaptive mixture of exploration and exploitation sampling strategies
to efficiently search the parameter space. Key features include:

- Initial exploration using uniform or Sobol sequence sampling
- Exploitation using Gaussian Mixture Models fitted to elite samples
- Multi-objective optimization with Pareto front ranking
- Parallel evaluation support for expensive objective functions
- Dynamic adaptation of sampling strategies
- Support for parameter and objective updates during optimization

Example:
    >>> # Define parameter search space
    >>> params = {
    ...     "learning_rate": create_param_config(
    ...         "continuous", min=1e-4, max=1e-1, scale=Scale.LOG
    ...     ),
    ...     "batch_size": create_param_config(
    ...         "integer", min=16, max=256
    ...     )
    ... }
    >>>
    >>> # Define objectives
    >>> objectives = {
    ...     "accuracy": create_objective(
    ...         "maximize",
    ...         target=0.95,    # Target 95% accuracy
    ...         limit=0.80,     # At least 80% required
    ...         priority=2.0     # Higher priority than training time
    ...     ),
    ...     "training_time": create_objective(
    ...         "minimize",
    ...         target=60.0,    # Target: 1 minute
    ...         limit=300.0,    # Limit: 5 minutes
    ...         priority=1.0
    ...     )
    ... }
    >>>
    >>> # Create and run optimizer
    >>> hola = HOLA(params, objectives)
    >>> for i in range(100):
    ...     params = hola.sample()
    ...     results = evaluate_model(params)
    ...     hola.add(results, params)
    >>>
    >>> # Get best results
    >>> best = hola.get_dataframe()
    >>> print(f"Best accuracy: {best.best_objectives['accuracy']:.2f}")
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import logging
from multiprocess import Lock, Pool
from multiprocess.synchronize import Lock as LockBase
import os
from pathlib import Path
from typing import Any, Callable, ClassVar, Iterator

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, SkipValidation, model_validator
from tqdm import tqdm

from hola.core.leaderboard import OBJECTIVE_PREFIX, Leaderboard
from hola.core.objective import ObjectiveName, ObjectiveConfig, ObjectiveScorer
from hola.core.params_old import ParamInfeasibleError, ParamName, ParamConfig, ParameterTransformer, create_param_config
from hola.core.sample import MixtureSampler, SampleType, Sampler

logging.basicConfig(level=logging.ERROR)


class SampleStorageMode(str, Enum):
    """Mode for handling samples that become infeasible after parameter updates.

    Controls how the optimizer handles existing samples when parameter
    configurations change and some samples become infeasible under the new
    configuration.

    :cvar DISCARD: Discard infeasible samples completely, removing them from
        consideration
    :cvar ARCHIVE: Store infeasible samples for potential recovery if they
        become feasible again under future parameter updates
    """
    DISCARD = "discard"
    ARCHIVE = "archive"


@dataclass
class ArchivedTrial:
    """Container for archived trials with their original parameters.

    Stores complete trial information for samples that become infeasible
    under parameter updates, allowing potential recovery if they become
    feasible again.

    :param params: Original parameter values (not normalized)
    :type params: dict[ParamName, float]
    :param objectives: Achieved objective values
    :type objectives: dict[ObjectiveName, float]
    :param param_config: Parameter configuration when trial was generated
    :type param_config: dict[ParamName, ParamConfig]
    """
    params: dict[ParamName, float]
    objectives: dict[ObjectiveName, float]
    param_config: dict[ParamName, ParamConfig]


class SavedConfiguration(BaseModel):
    """Container for serialized HOLA configuration.

    Stores all configuration data needed to reconstruct a HOLA instance,
    including parameter and objective configurations.
    """
    params_config: dict[ParamName, dict[str, Any]]
    objectives_config: dict[ObjectiveName, dict[str, Any]]
    hola_config: dict[str, Any]


def _evaluate_single(task_data: tuple[Callable, dict]) -> tuple[dict, dict]:
    """Evaluate a single parameter set.

    Helper function for parallel evaluation that executes the objective
    function with given parameters and handles errors.

    :param task_data: Tuple of (objective_function, parameter dictionary)
    :type task_data: tuple[Callable, dict]
    :return: Tuple of (parameters, objectives) if successful, None if failed
    :rtype: tuple[dict, dict] | None
    """
    func, params = task_data
    try:
        objectives = func(**params)
        return params, objectives
    except Exception as e:
        logging.error(f"Error evaluating parameters {params}: {e}", exc_info=True)
        return None


def _param_generator(
    hola: 'HOLA',
    lock: LockBase,
    num_runs: int,
) -> Iterator[dict]:
    """Generate parameter sets as needed with a buffer to maintain sampling efficiency.

    Thread-safe generator that produces parameter sets for parallel evaluation
    while ensuring proper synchronization of the optimizer's state.

    :param hola: HOLA instance to generate samples
    :type hola: HOLA
    :param lock: Lock for thread-safe sampling
    :type lock: LockBase
    :param num_runs: Total number of runs to perform
    :type num_runs: int
    :yield: Parameter dictionaries
    :rtype: Iterator[dict]
    """
    runs_remaining = num_runs
    while runs_remaining > 0:
        with lock:
            # Lock during sampling since it may read/update internal state
            params = hola.sample()
        yield params
        runs_remaining -= 1


class HOLA(BaseModel):
    """Hyperparameter Optimization, Lightweight and Automatic (HOLA).

    HOLA performs hyperparameter optimization by alternating between
    exploration and exploitation of the parameter space. Exploitation is done
    by fitting a GMM to elite samples to assign greater weight to regions
    with better objective values. The optimizer supports:

    - Single and multi-objective optimization
    - Parameter space exploration with uniform or Sobol sampling
    - Exploitation with adaptive Gaussian mixture models
    - Dynamic parameter and objective updates
    - Parallel evaluation of objective functions
    - Sample archiving and recovery

    Example:
        >>> # Create optimizer with default settings
        >>> hola = HOLA(params_config=params, objectives_config=objectives)
        >>>
        >>> # Create optimizer with custom settings
        >>> hola = HOLA(
        ...     params_config=params,
        ...     objectives_config=objectives,
        ...     top_fraction=0.2,        # Top 20% used for exploitation
        ...     min_samples=50,          # Min samples before exploitation
        ...     min_fit_samples=10,      # Min elite samples needed
        ...     n_components=3,          # GMM components
        ...     gmm_reg=0.0005,         # GMM regularization
        ...     gmm_sampler=SampleType.UNIFORM,
        ...     explore_sampler=SampleType.SOBOL,
        ... )

    :param params_config: Parameter configuration dictionary
    :type params_config: dict[ParamName, ParamConfig]
    :param objectives_config: Objective configuration dictionary
    :type objectives_config: dict[ObjectiveName, ObjectiveConfig]
    :param top_fraction: Fraction of top samples to use for exploitation
    :type top_fraction: float
    :param min_samples: Minimum samples before exploitation begins
    :type min_samples: int | None
    :param min_fit_samples: Minimum elite samples needed for GMM
    :type min_fit_samples: int | None
    :param n_components: Number of GMM components
    :type n_components: int
    :param gmm_reg: GMM covariance regularization
    :type gmm_reg: float
    :param gmm_sampler: Sampling strategy for GMM
    :type gmm_sampler: SampleType
    :param explore_sampler: Sampling strategy for exploration
    :type explore_sampler: SampleType
    """

    params: dict[ParamName, SkipValidation[ParamConfig]]
    objectives: dict[ObjectiveName, SkipValidation[ObjectiveConfig]]
    sampler: Sampler

    model_config: ClassVar[ConfigDict] = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True
    )

    @model_validator(mode="after")
    def validate_samples(self) -> "HOLA":
        """Validate minimum sample requirements.

        Ensures that min_samples and min_fit_samples are compatible with
        top_fraction to allow proper exploitation phase transitions.

        :return: Validated HOLA instance
        :rtype: HOLA
        :raises ValueError: If sample requirements are incompatible
        """
        if self.min_samples is not None and self.min_fit_samples is not None:
            if self.min_samples * self.top_fraction < self.min_fit_samples:
                raise ValueError(
                    f"min_samples * top_fraction ({self.min_samples * self.top_fraction}) "
                    f"must be >= min_fit_samples ({self.min_fit_samples})"
                )
        return self

    def model_post_init(self, __context: Any) -> None:
        """Initialize components after validation.

        Sets up samplers, transformer, scorer, and leaderboard with default
        values calculated from parameter dimension where appropriate.

        :param __context: Unused context parameter
        :type __context: Any
        """
        # Calculate defaults based on parameter dimension
        num_params = len(self.params)
        if self.min_samples is None:
            self.min_samples = max(num_params + 2, 10 * num_params)
        if self.min_fit_samples is None:
            self.min_fit_samples = 2

        # Initialize components
        self._param_transformer = ParameterTransformer(self.params)
        self._objective_scorer = ObjectiveScorer(self.objectives)
        self._leaderboard = Leaderboard(self._objective_scorer)
        self._archived_trials = []
        self._init_samplers(num_params)

    def _init_samplers(self, num_params: int) -> None:
        """Initialize exploration and exploitation samplers.

        Creates appropriate mixture sampler based on the explore_sampler
        configuration.

        :param num_params: Number of parameters to optimize
        :type num_params: int
        """
        if self.explore_sampler == SampleType.SOBOL:
            self._mixture_sampler = MixtureSampler.create_sobol_to_gmm(
                dimension=num_params,
                n_components=self.n_components,
                min_explore_samples=self.min_samples,
                min_fit_samples=self.min_fit_samples,
                reg_covar=self.gmm_reg,
                sample_type=self.gmm_sampler,
            )
        else:
            self._mixture_sampler = MixtureSampler.create_uniform_to_gmm(
                dimension=num_params,
                n_components=self.n_components,
                min_explore_samples=self.min_samples,
                min_fit_samples=self.min_fit_samples,
                reg_covar=self.gmm_reg,
                sample_type=self.gmm_sampler,
            )

    def update_params_config(
        self,
        updates: dict[ParamName, ParamConfig],
        storage_mode: SampleStorageMode = SampleStorageMode.DISCARD
    ) -> None:
        """Update specific parameter configurations.

        Updates parameter configurations and handles existing samples that may
        become infeasible under the new configuration.

        Example:
            >>> # Update single parameter
            >>> hola.update_params_config({
            ...     "learning_rate": create_param_config(
            ...         "continuous",
            ...         min=1e-5,  # Changed from 1e-4
            ...         max=1e-1,
            ...         scale="log"
            ...     )
            ... })

        :param updates: Map of parameter names to new configurations
        :type updates: dict[ParamName, ParamConfig]
        :param storage_mode: How to handle infeasible samples
        :type storage_mode: SampleStorageMode
        :raises ValueError: If new configuration is invalid or parameters
            don't exist
        """
        # Create new merged config
        new_config = self.params.copy()
        new_config.update(updates)

        # Verify all parameters being updated exist
        invalid_params = set(updates.keys()) - set(self.params.keys())
        if invalid_params:
            raise ValueError(f"Cannot update non-existent parameters: {invalid_params}")

        # Validate and create new transformer
        new_transformer = ParameterTransformer(new_config)

        # First try to recover any previously archived trials that are now feasible
        recovered_count = 0
        still_archived = []

        for archived in self._archived_trials:
            try:
                # Check if parameters are feasible under new config
                new_transformer.back_transform_param_dict(archived.params)
                # If feasible, we'll add it back later
                recovered_count += 1
            except ParamInfeasibleError:
                still_archived.append(archived)

        recovered_trials = [t for t in self._archived_trials if t not in still_archived]
        self._archived_trials = still_archived

        # Process existing trials
        valid_indices = []
        filtered_count = 0

        for i in range(self.num_samples):
            params, objectives = self.get_sample(i)
            try:
                # Verify parameters are feasible under new config
                new_transformer.back_transform_param_dict(params)
                valid_indices.append(i)
            except ParamInfeasibleError:
                filtered_count += 1
                if storage_mode == SampleStorageMode.ARCHIVE:
                    self._archived_trials.append(ArchivedTrial(
                        params=params,
                        objectives=objectives,
                        param_config=self.params
                    ))

        # Update state
        self.params = new_config
        self._param_transformer = new_transformer

        # Create new leaderboard with only valid samples
        new_leaderboard = Leaderboard(self._objective_scorer)

        # Add valid existing samples
        for idx in valid_indices:
            params, objectives = self.get_sample(idx)
            new_leaderboard.add(params, objectives)

        # Add recovered samples
        for trial in recovered_trials:
            new_leaderboard.add(trial.params, trial.objectives)

        self._leaderboard = new_leaderboard

        # Update sampler's sample count to match current state
        self._mixture_sampler.adjust_sample_count(self.num_samples)

        # Update sampler fitting if needed
        self._update_sampler()

        # Log changes
        if recovered_count > 0:
            print(f"Recovered {recovered_count} previously archived samples")
        if filtered_count > 0:
            action = "Archived" if storage_mode == SampleStorageMode.ARCHIVE else "Filtered"
            print(
                f"{action} {filtered_count} samples that became "
                "infeasible under new parameter configuration"
            )

    def update_objectives_config(
        self,
        updates: dict[ObjectiveName, ObjectiveConfig]
    ) -> None:
        """Update specific objective configurations.

        Updates objective configurations and recomputes scores for all
        existing samples under the new configuration.

        Example:
            >>> # Update single objective
            >>> hola.update_objectives_config({
            ...     "accuracy": create_objective(
            ...         "maximize",
            ...         target=0.98,  # Changed from 0.95
            ...         limit=0.80,
            ...         priority=2.0
            ...     )
            ... })

        :param updates: Map of objective names to new configurations
        :type updates: dict[ObjectiveName, ObjectiveConfig]
        :raises ValueError: If new configuration is invalid or objectives
            don't exist
        """
        # Create new merged config
        new_config = self.objectives.copy()
        new_config.update(updates)

        # Create new scorer
        new_scorer = ObjectiveScorer(new_config)

        # Create new leaderboard with new scorer
        new_leaderboard = Leaderboard(new_scorer)

        # Transfer samples
        for i in range(self.num_samples):
            params, objectives = self.get_sample(i)
            new_leaderboard.add(params, objectives)

        # Update state
        self.objectives = new_config
        self._objective_scorer = new_scorer
        self._leaderboard = new_leaderboard
        self._update_sampler()

    def get_best_sample(self) -> tuple[dict[ParamName, float], dict[ObjectiveName, float]]:
        """Get parameters and objectives for best sample.

        :return: (parameters, objectives) for best sample
        :rtype: tuple[dict[str, float], dict[str, float]]
        :raises IndexError: If no samples exist
        """
        return self._leaderboard.get_best_sample()

    def get_top_samples(
        self,
        num_samples: int
    ) -> list[tuple[dict[ParamName, float], dict[ObjectiveName, float]]]:
        """Get parameters and objectives for top samples.

        For multi-group objectives, returns samples ordered by Pareto
        dominance and crowding distance.

        :param num_samples: Number of samples to return
        :type num_samples: int
        :return: List of (parameters, objectives) tuples for top samples
        :rtype: list[tuple[dict[str, float], dict[str, float]]]
        """
        return self._leaderboard.get_top_samples(num_samples)

    @property
    def num_samples(self) -> int:
        """Get total number of samples in the optimization.

        :return: Number of samples
        :rtype: int
        """
        return self._leaderboard.num_samples()

    @property
    def num_archived_samples(self) -> int:
        """Get number of archived samples.

        :return: Number of archived samples
        :rtype: int
        """
        return len(self._archived_trials)

    def get_sample(self, index: int) -> tuple[dict[ParamName, float], dict[ObjectiveName, float]]:
        """Get parameters and objectives for specific sample.

        :param index: Sample index
        :type index: int
        :return: (parameters, objectives) for sample
        :rtype: tuple[dict[str, float], dict[str, float]]
        :raises IndexError: If index is invalid
        """
        return self._leaderboard.get_sample(index)

    def add(
        self,
        objectives: dict[ObjectiveName, float],
        params: dict[ParamName, float],
    ) -> None:
        """Add a new run to the optimization history.

        :param objectives: Objective values from the run
        :type objectives: dict[str, float]
        :param params: Parameter values used in the run
        :type params: dict[str, float]
        :raises ValueError: If parameter or objective names don't match config
        """
        # Validate parameter names
        invalid_params = set(params.keys()) - set(self.params.keys())
        if invalid_params:
            raise KeyError(f"Invalid parameter names: {invalid_params}")

        # Validate objective names
        invalid_objectives = set(objectives.keys()) - set(self.objectives.keys())
        if invalid_objectives:
            raise KeyError(f"Invalid objective names: {invalid_objectives}")

        self._leaderboard.add(params, objectives)
        self._update_sampler()

    def _update_sampler(self) -> None:
        """Update exploitation sampler with elite samples if ready.

        Internal method that updates sample counts and fits the GMM to
        elite samples when appropriate conditions are met.
        """
        self._mixture_sampler.adjust_sample_count(self.num_samples)

        # Calculate number of elite samples
        num_elite = int(self.num_samples * self.top_fraction)

        # Check if sampler is ready for fitting
        if self._mixture_sampler.is_ready_to_fit(num_elite):
            # Get elite samples
            elite_samples = []
            for params, _ in self.get_top_samples(num_elite):
                elite_params = self._param_transformer.back_transform_param_dict(params)
                elite_samples.append(elite_params)

            # Fit sampler if we have samples
            if elite_samples:
                self._mixture_sampler.fit(np.array(elite_samples))

    def sample(self) -> dict[ParamName, Any]:
        """Generate next set of parameters to try.

        :return: Dictionary of parameter names and values
        :rtype: dict[ParamName, Any]
        """
        normalized_params = self._mixture_sampler.sample()
        return self._param_transformer.transform_normalized_params(normalized_params)

    def get_dataframe(self) -> pd.DataFrame:
        """Get current optimization results.

        :return: The optimization results as a DataFrame
        :rtype: pd.DataFrame
        """
        return self._leaderboard.get_dataframe()

    def save(self, filename: Path | str, save_config: bool = False) -> None:
        """Save optimization state to file.

        :param filename: Path to save file
        :type filename: Path | str
        :param save_config: Whether to save configuration data, defaults to
            False
        :type save_config: bool, optional
        """
        filename = Path(filename)
        # Save leaderboard data
        self._leaderboard.save(filename.with_suffix('.csv'))

        if save_config:
            # Prepare configuration data
            config = SavedConfiguration(
                params_config={
                    name: {"type": param_config.param_type, **param_config.model_dump()}
                    for name, param_config in self.params.items()
                },
                objectives_config={
                    name: obj_config.model_dump()
                    for name, obj_config in self.objectives.items()
                },
                hola_config={
                    "top_fraction": self.top_fraction,
                    "min_samples": self.min_samples,
                    "min_fit_samples": self.min_fit_samples,
                    "n_components": self.n_components,
                    "gmm_reg": self.gmm_reg,
                    "gmm_sampler": self.gmm_sampler.value,  # Use enum value
                    "explore_sampler": self.explore_sampler.value,  # Use enum value
                }
            )
            # Save configuration
            with open(filename.with_suffix('.json'), 'w') as f:
                json.dump(config.model_dump(), f, indent=2)

    @classmethod
    def load(
        cls,
        filename: Path | str,
        params_config: dict[ParamName, ParamConfig] | None = None,
        objectives_config: dict[ObjectiveName, ObjectiveConfig] | None = None,
        hola_config: dict[str, Any] | None = None,
        load_config: bool = False
    ) -> HOLA:
        """Load optimization state from file.

        :param filename: Path to load file
        :type filename: Path | str
        :param params_config: Parameter configurations if not loading from file
        :type params_config: dict[ParamName, ParamConfig] | None
        :param objectives_config: Objective configurations if not loading from
            file
        :type objectives_config: dict[ObjectiveName, ObjectiveConfig] | None
        :param hola_config: HOLA settings if not loading from file
        :type hola_config: dict[str, Any] | None
        :param load_config: Whether to load configuration from file, defaults
            to False
        :type load_config: bool
        :return: HOLA instance with loaded state
        :rtype: HOLA
        :raises ValueError: If neither configuration file nor parameters
            provided
        :raises FileNotFoundError: If required files do not exist
        """
        filename = Path(filename)

        if load_config:
            # Try to load config first
            json_file = filename.with_suffix('.json')
            if not json_file.exists():
                raise ValueError(f"Configuration file not found: {json_file}")

            with open(json_file) as f:
                config_data = json.load(f)
            config = SavedConfiguration.model_validate(config_data)

            # Reconstruct configurations
            params_config = {}
            for name, param_data in config.params_config.items():
                param_type = param_data.pop("type")
                params_config[name] = create_param_config(param_type, **param_data)

            objectives_config = {
                name: ObjectiveConfig.model_validate(obj_data)
                for name, obj_data in config.objectives_config.items()
            }
            hola_config = config.hola_config
        else:
            # Check if configs were provided
            if params_config is None or objectives_config is None:
                raise ValueError(
                    "Must provide params_config and objectives_config when load_config=False"
                )

        # Check if CSV exists after getting configs
        csv_file = filename.with_suffix('.csv')
        if not csv_file.exists():
            raise FileNotFoundError(f"Data file not found: {csv_file}")

        # Create HOLA instance
        hola = cls(
            params_config=params_config,
            objectives_config=objectives_config,
            **(hola_config or {})
        )

        # Load leaderboard data
        hola._leaderboard = Leaderboard.load(csv_file, hola._objective_scorer)

        # Update sampler state
        num_samples = hola.num_samples
        for _ in range(num_samples):
            hola._mixture_sampler.sample()
        hola._update_sampler()

        return hola

    @classmethod
    def tune(
        cls,
        func: Callable[..., dict[str, float]],
        params: dict[ParamName, ParamConfig],
        objectives: dict[ObjectiveName, ObjectiveConfig],
        *,
        num_runs: int = 100,
        n_jobs: int = 1,
        min_samples: int | None = None,
        **kwargs: Any,
    ) -> HOLA:
        """Run hyperparameter optimization with dynamic parallel evaluation.

        This implementation:
        1. Processes evaluations dynamically as workers become available
        2. Updates the leaderboard as results come in
        3. Handles stragglers gracefully

        Example:
            >>> def evaluate(learning_rate: float, batch_size: int) -> dict[str, float]:
            ...     model = train_model(learning_rate, batch_size)
            ...     return {
            ...         "accuracy": model.evaluate(),
            ...         "training_time": model.train_time
            ...     }
            >>>
            >>> hola = HOLA.tune(
            ...     evaluate,
            ...     params=params_config,
            ...     objectives=objectives_config,
            ...     num_runs=100,
            ...     n_jobs=4  # Parallel evaluation
            ... )
            >>> results = hola.get_dataframe()

        :param func: Objective function that takes parameter values and returns
            objectives
        :type func: Callable[..., dict[str, float]]
        :param params: Parameter search space configuration
        :type params: dict[ParamName, ParamConfig]
        :param objectives: Objective configuration with groups and priorities
        :type objectives: dict[ObjectiveName, ObjectiveConfig]
        :param num_runs: Number of evaluations to perform
        :type num_runs: int
        :param n_jobs: Number of parallel jobs (-1 for all CPUs)
        :type n_jobs: int
        :param min_samples: Minimum samples before exploitation
        :type min_samples: int | None
        :param kwargs: Additional arguments passed to HOLA constructor
        :type kwargs: Any
        :return: Optimized HOLA instance
        :rtype: HOLA
        :raises ValueError: If configuration parameters are invalid
        """
        if num_runs < 1:
            raise ValueError(f"num_runs must be >= 1, got {num_runs}")

        if n_jobs < 1 and n_jobs != -1:
            raise ValueError(f"n_jobs must be >= 1 or -1, got {n_jobs}")

        # Create optimizer
        hola = cls(
            params_config=params,
            objectives_config=objectives,
            min_samples=min_samples,
            **kwargs
        )

        if n_jobs == 1:
            # Single process optimization
            for i in range(num_runs):
                params = hola.sample()
                result = func(**params)
                hola.add(result, params)
                if (i + 1) % 10 == 0:
                    print(f"Completed {i + 1}/{num_runs} runs")
        else:
            # Parallel optimization
            n_workers = os.cpu_count() if n_jobs == -1 else n_jobs
            print(f"Running parallel optimization with {n_workers} workers")

            # Create lock for thread-safe operations
            lock = Lock()

            with Pool(processes=n_workers) as pool, tqdm(total=num_runs) as progress_bar:
                # Create parameter generator with buffer
                param_gen = _param_generator(hola, lock, num_runs)

                # Create task iterator that pairs function with parameters
                tasks = ((func, params) for params in param_gen)

                # Process tasks as they complete
                for result in pool.imap_unordered(_evaluate_single, tasks):
                    if result is not None:  # Skip failed evaluations
                        params, objectives = result

                        # Lock during leaderboard update and related state changes
                        with lock:
                            hola.add(objectives, params)

                        progress_bar.update(1)

        # Print summary
        results = hola.get_dataframe()
        best_params, best_objectives = hola._leaderboard.get_best_sample()
        print("\nOptimization complete.")
        print(f"Best objectives: {best_objectives}")
        print(f"Best parameters: {best_params}")

        return hola

    def __repr__(self) -> str:
        """Return string representation including configuration settings.

        :return: String representation of HOLA instance
        :rtype: str
        """
        return (
            f"{self.__class__.__name__}("
            f"top_fraction={self.top_fraction}, "
            f"min_samples={self.min_samples}, "
            f"min_fit_samples={self.min_fit_samples}, "
            f"n_components={self.n_components}, "
            f"gmm_reg={self.gmm_reg}, "
            f"gmm_sampler='{self.gmm_sampler}', "
            f"explore_sampler='{self.explore_sampler}', "
            f"num_params={len(self.params)}, "
            f"num_objectives={len(self.objectives)}"
            ")"
        )