"""Sampling utilities for hyperparameter optimization.

This module provides a collection of sampling strategies for hyperparameter
optimization:

- Uniform random sampling for initial exploration
- Sobol sequence sampling for low-discrepancy coverage
- Gaussian Mixture Model (GMM) sampling for exploitation
- Mixture sampling that combines exploration and exploitation phases

Each sampler implements the common SamplerProtocol interface, making them
interchangeable within the optimization process. The samplers operate in the
[0, 1] hypercube space, with results intended to be transformed to actual
hyperparameter values by the caller.

Example:
    >>> # Create a mixture sampler that transitions from Sobol to GMM
    >>> mixture = MixtureSampler.create_sobol_to_gmm(
    ...     dimension=3,
    ...     n_components=2,
    ...     min_explore_samples=10,
    ...     min_fit_samples=5
    ... )
    >>> # Sample during exploration phase
    >>> elite_samples = np.empty((5, 3))
    >>> for i in range(5):
    ...     elite_samples[i] = mixture.sample()
    ...
    >>> # Fit exploitation sampler with elite samples
    >>> mixture.fit(elite_samples)
    >>> # Continue sampling
    >>> mixture.sample()
    array([0.06419425, 0.0301052 , 0.14682435])
"""

from __future__ import annotations

from abc import ABC
from enum import Enum
from typing import Protocol

import numpy as np
import numpy.typing as npt
from scipy.stats import norm
from scipy.stats.qmc import Sobol
from sklearn.mixture import GaussianMixture  # type: ignore[import-untyped]

from hola.core.utils import uniform_to_category


class SampleType(str, Enum):
    """Type of sampling strategy to use.

    :cvar UNIFORM: Use uniform random sampling for generating samples
    :cvar SOBOL: Use Sobol sequence for low-discrepancy quasi-random sampling
    """

    UNIFORM = "uniform"
    SOBOL = "sobol"


class Sampler(Protocol):
    """Protocol defining the interface for all samplers.

    All sampler implementations must conform to this protocol to ensure
    consistency and interchangeability within the optimization process.
    Each sampler operates in the [0, 1] hypercube space.
    """

    @property
    def dimension(self) -> int:
        """Get the dimension of the sampler.

        :return: The number of dimensions in the sampling space
        :rtype: int
        """
        ...

    def sample(self) -> npt.NDArray[np.float64]:
        """Generate samples from the sampler's distribution.

        :return: Array of samples with shape (dimension,)
        :rtype: numpy.ndarray
        """
        ...

    def fit(self, samples: npt.NDArray[np.float64]) -> None:
        """Fit the sampler to observed data if necessary.

        :param samples: Array of samples to fit to, shape (n_samples,
            dimension)
        :type samples: numpy.ndarray
        """
        ...


class UniformSampler:
    """Uniform random sampler in the [0, 1]^dimension hypercube.

    Generates samples uniformly at random from the [0, 1]^dimension hypercube.
    This sampler is typically used during the initial exploration phase when
    no prior knowledge about good solutions exists.

    Example:
        >>> sampler = UniformSampler(dimension=2)
        >>> sample = sampler.sample()
        array([0.81269556, 0.46055788])
    """

    def __init__(self, dimension: int):
        """Initialize uniform sampler.

        :param dimension: The number of dimensions in the sampling space
        :type dimension: int
        :raises ValueError: If dimension is not positive
        """
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        """Get the dimension of the sampling space.

        :return: Number of dimensions
        :rtype: int
        """
        return self._dimension

    def sample(self) -> npt.NDArray[np.float64]:
        """Generate uniform random samples.

        :return: Array of uniform random samples in [0, 1]^dimension with
            shape (dimension,)
        :rtype: numpy.ndarray
        """
        return np.random.rand(self.dimension)

    def fit(self, samples: npt.NDArray[np.float64]) -> None:
        """No-op since uniform sampling doesn't require fitting.

        :param samples: Array of samples (unused in this implementation)
        :type samples: numpy.ndarray
        """
        pass


class SobolSampler:
    """Low-discrepancy sequence sampler using the Sobol sequence.

    This sampler provides more uniform coverage of the sampling space compared
    to pure random sampling by using a quasi-random low-discrepancy sequence.
    This is particularly beneficial during the initial exploration phase for
    more systematic coverage of the parameter space.

    Example:
        >>> sampler = SobolSampler(dimension=2)
        >>> sampler.sample()
        array([0.40065987, 0.86596148])
        >>> sampler.sample()
        array([0.61976991, 0.02656887])

    Note:
        For optimal uniformity properties, the total number of samples
        drawn from a Sobol sequence should ideally be a power of 2.
    """

    def __init__(self, dimension: int):
        """Initialize the Sobol sampler.

        :param dimension: The number of dimensions in the sampling space
        :type dimension: int
        :raises ValueError: If dimension is not positive
        """
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._dimension = dimension
        self._sampler = Sobol(dimension)

    @property
    def dimension(self) -> int:
        """Get the dimension of the sampling space.

        :return: Number of dimensions
        :rtype: int
        """
        return self._dimension

    def sample(self) -> npt.NDArray[np.float64]:
        """Generate samples from the Sobol sequence.

        Each call advances the sequence and returns the next point. Points
        are generated deterministically to ensure good space coverage.

        :return: Array of Sobol sequence samples in [0, 1]^dimension with
            shape (dimension,)
        :rtype: numpy.ndarray
        """
        return self._sampler.random().ravel()

    def fit(self, samples: npt.NDArray[np.float64]) -> None:
        """No-op since Sobol sequence doesn't require fitting.

        :param samples: Array of samples (unused in this implementation)
        :type samples: numpy.ndarray
        """
        pass


class ClippedGaussianMixtureSampler:
    """Clipped Gaussian Mixture Model (GMM) sampler for exploitation.

    This sampler fits a GMM to elite samples and generates new samples from the
    fitted distribution. All samples are clipped to the [0, 1] hypercube to
    ensure valid hyperparameter values. The sampler supports both uniform
    random sampling and Sobol sequences for generating base samples that are
    then transformed through the GMM.

    Example:
        >>> # Create and fit a GMM sampler
        >>> sampler = ClippedGaussianMixtureSampler(
        ...     dimension=2,
        ...     n_components=3,
        ...     sample_type="sobol"
        ... )
        ...
        >>> elite_samples = np.random.standard_normal((10, 2))
        >>> sampler.fit(elite_samples)
        >>> sampler.sample()
        array([0., 1.])
    """

    def __init__(
        self,
        dimension: int,
        n_components: int,
        reg_covar: float = 1e-6,
        hypercube_sampler: Sampler | None = None
    ):
        """Initialize GMM sampler.

        :param dimension: The number of dimensions in the sampling space
        :type dimension: int
        :param n_components: Number of Gaussian components in the mixture
        :type n_components: int
        :param reg_covar: Regularization added to covariance matrices to ensure
            numerical stability, defaults to 1e-6
        :type reg_covar: float, optional
        :raises ValueError: If dimension or n_components are not positive, or if
            reg_covar is negative
        """
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        if n_components <= 0:
            raise ValueError("n_components must be positive")
        if reg_covar < 0:
            raise ValueError("reg_covar must be non-negative")

        # GMM data
        self._gmm = GaussianMixture(
            n_components=n_components,
            reg_covar=reg_covar,
            covariance_type="full",
        )
        self._gmm_chols: tuple[npt.NDArray[np.float64], ...] | None = None
        self._n_components = n_components

        # Sampler data
        self._dimension = dimension
        if hypercube_sampler is None:
            hypercube_sampler = UniformSampler(dimension)
        self._hypercube_sampler = hypercube_sampler

    @property
    def dimension(self) -> int:
        """Get the dimension of the sampling space.

        :return: Number of dimensions
        :rtype: int
        """
        return self._dimension

    @property
    def n_components(self) -> int:
        """Get the number of GMM components.

        :return: Number of GMM components
        :rtype: int
        """
        return self._n_components

    def fit(self, samples: npt.NDArray[np.float64]) -> None:
        """Fit the GMM to provided samples.

        :param samples: Array of samples to fit the GMM to, shape
            (n_samples, dimension)
        :type samples: numpy.ndarray
        :raises ValueError: If samples array has wrong dimension or if there
            are too few samples to fit the requested number of components
        """
        if samples.shape[1] != self.dimension:
            raise ValueError(
                f"Expected samples with dimension = {self.dimension}, "
                f"got samples with dimension = {samples.shape[1]}"
            )
        self._gmm.fit(samples)
        self._gmm_chols = tuple(
            np.linalg.cholesky(cov) for cov in self._gmm.covariances_
        )

    def sample(self) -> npt.NDArray[np.float64]:
        """Generate samples from the fitted GMM.

        :return: Array of samples from the GMM, clipped to [0, 1]^dimension,
            with shape (dimension,)
        :rtype: numpy.ndarray
        :raises ValueError: If the GMM has not been fitted or if sample_type
            is invalid
        """
        if self._gmm_chols is None:
            raise ValueError("The sampler must be fitted before sampling")

        u_sample = self._hypercube_sampler.sample()
        component = uniform_to_category(u_sample[0], self.n_components)
        u_gmm = u_sample[1:]

        mean = self._gmm.means_[component]
        chol = self._gmm_chols[component]

        z_gmm = norm.ppf(u_gmm)
        sample = mean + chol @ z_gmm
        np.clip(sample, 0.0, 1.0, out=sample)
        return sample


class MixtureSampler:
    """Adaptive sampler that combines exploration and exploitation strategies.

    This sampler implements a transition from exploration (using uniform or
    Sobol sampling) to exploitation (using GMM sampling). The transition occurs
    after a specified number of samples and only if sufficient elite samples
    are available for fitting the exploitation sampler.

    The sampler follows these rules:
    1. Uses exploration sampling until min_explore_samples is reached
    2. Switches to exploitation sampling if:
       - min_explore_samples has been reached
       - At least min_fit_samples elite samples are provided
       - The exploitation sampler has been successfully fitted
    3. Falls back to exploration sampling if any conditions aren't met

    Example:
        >>> # Create a mixture sampler that transitions from Sobol to GMM
        >>> mixture = MixtureSampler.create_sobol_to_gmm(
        ...     dimension=3,
        ...     n_components=2,
        ...     min_explore_samples=10,
        ...     min_fit_samples=5
        ... )
        >>> # Sample during exploration phase
        >>> elite_samples = np.empty((5, 3))
        >>> for i in range(5):
        ...     elite_samples[i] = mixture.sample()
        ...
        >>> # Fit exploitation sampler with elite samples
        >>> mixture.fit(elite_samples)
        >>> # Continue sampling
        >>> mixture.sample()
        array([0.06419425, 0.0301052 , 0.14682435])
    """

    @classmethod
    def create_uniform_to_gmm(
        cls: type["MixtureSampler"],
        dimension: int,
        n_components: int,
        min_explore_samples: int = 10,
        min_fit_samples: int = 5,
        reg_covar: float = 1e-6,
        hypercube_sampler: Sampler | None = None,
    ) -> "MixtureSampler":
        """Create a mixture sampler that transitions from uniform to GMM sampling.

        This is a convenience method that creates and configures a
        MixtureSampler using UniformSampler for exploration and
        ClippedGaussianMixtureSampler for exploitation.

        :param dimension: The number of dimensions in the sampling space
        :type dimension: int
        :param n_components: Number of Gaussian components in the mixture
        :type n_components: int
        :param min_explore_samples: Minimum number of exploration samples,
            defaults to 10
        :type min_explore_samples: int, optional
        :param min_fit_samples: Minimum samples before fitting GMM, defaults
            to 5
        :type min_fit_samples: int, optional
        :param reg_covar: GMM covariance regularization, defaults to 1e-6
        :type reg_covar: float, optional
        :param sample_type: Method for GMM sampling, defaults to UNIFORM
        :type sample_type: SampleType, optional
        :return: Configured mixture sampler
        :rtype: MixtureSampler
        """
        explore_sampler = UniformSampler(dimension=dimension)
        exploit_sampler = ClippedGaussianMixtureSampler(
            dimension=dimension,
            n_components=n_components,
            reg_covar=reg_covar,
            hypercube_sampler=hypercube_sampler,
        )

        return cls(
            sampler_explore=explore_sampler,
            sampler_exploit=exploit_sampler,
            min_explore_samples=min_explore_samples,
            min_fit_samples=min_fit_samples,
        )

    @classmethod
    def create_sobol_to_gmm(
        cls: type["MixtureSampler"],
        dimension: int,
        n_components: int,
        min_explore_samples: int = 10,
        min_fit_samples: int = 5,
        reg_covar: float = 1e-6,
        hypercube_sampler: Sampler | None = None,
    ) -> "MixtureSampler":
        """Create a mixture sampler that transitions from Sobol to GMM sampling.

        This is a convenience method that creates and configures a
        MixtureSampler using SobolSampler for exploration and
        ClippedGaussianMixtureSampler for exploitation.

        :param dimension: The number of dimensions in the sampling space
        :type dimension: int
        :param n_components: Number of Gaussian components in the mixture
        :type n_components: int
        :param min_explore_samples: Minimum number of exploration samples,
            defaults to 10
        :type min_explore_samples: int, optional
        :param min_fit_samples: Minimum samples before fitting GMM, defaults to
            5
        :type min_fit_samples: int, optional
        :param reg_covar: GMM covariance regularization, defaults to 1e-6
        :type reg_covar: float, optional
        :param sample_type: Method for GMM sampling, defaults to UNIFORM
        :type sample_type: SampleType, optional
        :return: Configured mixture sampler
        :rtype: MixtureSampler
        """
        explore_sampler = SobolSampler(dimension=dimension)
        exploit_sampler = ClippedGaussianMixtureSampler(
            dimension=dimension,
            n_components=n_components,
            reg_covar=reg_covar,
            hypercube_sampler=hypercube_sampler,
        )

        return cls(
            sampler_explore=explore_sampler,
            sampler_exploit=exploit_sampler,
            min_explore_samples=min_explore_samples,
            min_fit_samples=min_fit_samples,
        )

    def __init__(
        self,
        sampler_explore: Sampler,
        sampler_exploit: Sampler,
        min_explore_samples: int = 10,
        min_fit_samples: int = 5,
    ):
        """Initialize mixture sampler.

        :param sampler_explore: Sampler used for exploration phase
        :type sampler_explore: SamplerProtocol
        :param sampler_exploit: Optional sampler used for exploitation phase
        :type sampler_exploit: SamplerProtocol | None
        :param min_explore_samples: Minimum number of samples to generate during
            exploration
        :type min_explore_samples: int
        :param min_fit_samples: Minimum number of samples required before fitting
            exploitation sampler
        :type min_fit_samples: int
        :raises ValueError: If min_samples values are invalid or if samplers
            have different dimensions
        """
        if min_explore_samples <= 0:
            raise ValueError("min_explore_samples must be positive")
        if min_fit_samples <= 0:
            raise ValueError("min_fit_samples must be positive")
        if min_fit_samples > min_explore_samples:
            raise ValueError("min_fit_samples cannot be greater than min_explore_samples")
        if sampler_exploit.dimension != sampler_explore.dimension:
            raise ValueError("Explore and exploit samplers must have same dimension")

        self._sampler_explore = sampler_explore
        self._sampler_exploit = sampler_exploit
        self._min_explore_samples = min_explore_samples
        self._min_fit_samples = min_fit_samples
        self._generated_samples = 0
        self._external_samples = 0
        self._is_fitted = False

    @property
    def dimension(self) -> int:
        """Get the dimension of the sampling space.

        :return: Number of dimensions
        :rtype: int
        """
        return self._sampler_explore.dimension

    @property
    def sample_count(self) -> int:
        """Get the total number of samples generated.

        :return: Total number of samples
        :rtype: int
        """
        return self._generated_samples + self._external_samples

    @property
    def min_explore_samples(self) -> int:
        """Get the minimum number of exploration samples required.

        :return: Minimum exploration samples
        :rtype: int
        """
        return self._min_explore_samples

    @property
    def min_fit_samples(self) -> int:
        """Get the minimum number of samples needed for fitting.

        :return: Minimum fitting samples
        :rtype: int
        """
        return self._min_fit_samples

    def is_ready_to_fit(self, num_elite_samples: int) -> bool:
        """Check if the sampler is ready to transition to exploitation.

        The sampler is ready to fit when:
        1. We have generated at least min_explore_samples
        2. We have at least min_fit_samples elite samples
        3. We have an exploitation sampler to fit

        :param num_elite_samples: Number of elite samples available for fitting
        :type num_elite_samples: int
        :return: True if ready to fit exploitation sampler
        :rtype: bool
        """
        return (
            self.sample_count >= self._min_explore_samples
            and num_elite_samples >= self._min_fit_samples
        )

    def is_using_exploitation(self) -> bool:
        """Check if currently using exploitation sampling.

        :return: True if using exploitation sampler
        :rtype: bool
        """
        return (
            self.sample_count >= self._min_explore_samples
            and self._is_fitted
        )

    def sample(self) -> npt.NDArray[np.float64]:
        """Generate samples using either exploration or exploitation sampler.

        The sampler automatically switches between exploration and exploitation
        based on the number of samples generated and whether the exploitation
        sampler has been fitted.

        :return: Array of samples with shape (dimension,)
        :rtype: numpy.ndarray
        """
        sampler = self._sampler_exploit if self.is_using_exploitation() else self._sampler_explore
        if sampler is None:
            raise ValueError("Exploitation mode active but no exploitation sampler exists.")
        sample = sampler.sample()
        return sample

    def fit(self, samples: npt.NDArray[np.float64]) -> None:
        """Fit the exploitation sampler if possible.

        :param samples: Array of samples to fit the exploitation sampler,
            shape (n_samples, dimension)
        :type samples: npt.NDArray[np.float64]
        :raises ValueError: If samples array has wrong shape or dimension
        """
        if self._sampler_exploit is None:
            return

        if samples.ndim != 2:
            raise ValueError(
                f"Expected 2D array with shape (n_samples, dimension), "
                f"got array with ndim = {samples.ndim}"
            )
        if samples.shape[1] != self.dimension:
            raise ValueError(
                f"Expected samples with dimension = {self.dimension}, "
                f"got samples with dimension = {samples.shape[1]}"
            )
        if len(samples) < self._min_fit_samples:
            return

        try:
            self._sampler_exploit.fit(samples)
            self._is_fitted = True
        except Exception:
            self._is_fitted = False

    def add_external_sample(self, sample: npt.NDArray[np.float64]) -> None:
        """Add external sample"""
        if len(sample) != self.dimension:
            raise ValueError("Sample does not match expected dimension")

        self._external_samples += 1
