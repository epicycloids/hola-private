"""Sampling utilities for hyperparameter optimization.

This module provides a collection of sampling strategies for hyperparameter
optimization:

- Uniform random sampling for initial exploration
- Sobol sequence sampling for low-discrepancy coverage
- Gaussian Mixture Model (GMM) sampling for exploitation
- Mixture sampling that combines exploration and exploitation phases

Each sampler implements the common SamplerProtocol interface, making them
interchangeable within the optimization process. The samplers use the [0, 1]
hypercube as their sampling space, with results intended to be transformed to
actual hyperparameter values by the caller.

Example:
    >>> # Create a mixture sampler that transitions from Sobol to GMM
    >>> mixture = MixtureSampler.create_sobol_to_gmm(
    ...     dimension=3,
    ...     n_components=2,
    ...     min_explore_samples=10,
    ...     min_fit_samples=5
    ... )
    >>> # Sample during exploration phase
    >>> initial_samples = mixture.sample(5)
    >>> # Fit exploitation sampler with elite samples
    >>> mixture.fit(elite_samples)
    >>> # Continue sampling
    >>> next_samples = mixture.sample(5)
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, Protocol

import numpy as np
import numpy.typing as npt
from pydantic import Field, model_validator
from scipy.stats import norm
from scipy.stats.qmc import Sobol
from sklearn.mixture import GaussianMixture

from hola.core.utils import BaseConfig


class SampleType(str, Enum):
    """Type of sampling strategy to use.

    :cvar UNIFORM: Use uniform random sampling
    :cvar SOBOL: Use Sobol sequence for low-discrepancy sampling
    """

    UNIFORM = "uniform"
    SOBOL = "sobol"


class SamplerProtocol(Protocol):
    """Protocol defining the interface for all samplers.

    All sampler implementations must conform to this protocol to ensure
    consistency and interchangeability.
    """

    @property
    def dimension(self) -> int:
        """Get the dimension of the sampler.

        :return: The number of dimensions in the sampling space.
        :rtype: int
        """
        ...

    def sample(self, num_samples: int = 1) -> npt.NDArray[np.float64]:
        """Generate samples from the sampler's distribution.

        :param num_samples: The number of samples to generate, defaults to 1
        :type num_samples: int, optional
        :return: Array of samples with shape (num_samples, dimension)
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
    This sampler is typically used during the initial exploration phase.

    :param dimension: The number of dimensions in the sampling space
    :type dimension: int
    :raises ValueError: If dimension is not positive
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

    def sample(self, num_samples: int = 1) -> npt.NDArray[np.float64]:
        """Generate uniform random samples.

        :param num_samples: The number of samples to generate, defaults to 1
        :type num_samples: int, optional
        :return: Array of uniform random samples in [0, 1]^dimension
        :rtype: numpy.ndarray
        """
        return np.random.rand(num_samples, self.dimension)

    def fit(self, samples: npt.NDArray[np.float64]) -> None:
        """No-op since uniform sampling doesn't require fitting.

        :param samples: Array of samples (unused in this implementation)
        :type samples: numpy.ndarray
        """
        pass


class SobolSampler:
    """Low-discrepancy sequence sampler using the Sobol sequence.

    This sampler provides more uniform coverage of the sampling space compared
    to pure random sampling, which is particularly beneficial during the
    initial exploration phase.

    Example:
        >>> sampler = SobolSampler(dimension=2)
        >>> samples = sampler.sample(10)  # Get first 10 points
        >>> more = sampler.sample(5)      # Continue sequence with next 5 points

    :param dimension: The number of dimensions in the sampling space
    :type dimension: int
    :raises ValueError: If dimension is not positive
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

    def sample(self, num_samples: int = 1) -> npt.NDArray[np.float64]:
        """Generate samples from the Sobol sequence.

        Each call advances the sequence and returns the next set of points.

        :param num_samples: Number of samples to generate, defaults to 1
        :type num_samples: int, optional
        :return: Array of Sobol sequence samples in [0, 1]
        :rtype: numpy.ndarray

        Note:
            For optimal uniformity properties, the total number of samples
            drawn should be a power of 2.
        """
        return self._sampler.random(num_samples)

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
    ensure valid hyperparameter values. The sampler can use either uniform
    random sampling or Sobol sequences for the underlying sampling process.

    Example:
        >>> # Create and fit a GMM sampler
        >>> sampler = ClippedGaussianMixtureSampler(
        ...     dimension=2,
        ...     n_components=3,
        ...     sample_type=SampleType.SOBOL
        ... )
        >>> sampler.fit(elite_samples)
        >>> # Generate new samples
        >>> samples = sampler.sample(10)

    :param dimension: The number of dimensions in the sampling space
    :type dimension: int
    :param n_components: Number of Gaussian components in the mixture
    :type n_components: int
    :param reg_covar: Regularization added to covariance matrices, defaults to
        1e-6
    :type reg_covar: float, optional
    :param sample_type: Method for sampling, defaults to UNIFORM
    :type sample_type: SampleType, optional
    :raises ValueError: If dimension or n_components are not positive, or if
        reg_covar is negative
    """

    def __init__(
        self,
        dimension: int,
        n_components: int,
        reg_covar: float = 1e-6,
        sample_type: SampleType = SampleType.UNIFORM,
    ):
        """Initialize GMM sampler.

        :param dimension: The number of dimensions in the sampling space
        :type dimension: int
        :param n_components: Number of Gaussian components in the mixture
        :type n_components: int
        :param reg_covar: Regularization added to covariance matrices, defaults
            to 1e-6
        :type reg_covar: float, optional
        :param sample_type: Method for sampling, defaults to UNIFORM
        :type sample_type: SampleType, optional
        :raises ValueError: If dimension or n_components are not positive, or
            if reg_covar is negative
        """
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        if n_components <= 0:
            raise ValueError("n_components must be positive")
        if reg_covar < 0:
            raise ValueError("reg_covar must be non-negative")

        self._dimension = dimension
        self._gmm = GaussianMixture(
            n_components=n_components,
            reg_covar=reg_covar,
            covariance_type="full",
        )
        self._sample_type = sample_type
        self._sobol = Sobol(dimension + 1) if sample_type == SampleType.SOBOL else None
        self._gmm_chols: Optional[tuple[npt.NDArray[np.float64]]] = None

    @property
    def dimension(self) -> int:
        """Get the dimension of the sampling space.

        :return: Number of dimensions
        :rtype: int
        """
        return self._dimension

    def fit(self, samples: npt.NDArray[np.float64]) -> None:
        """Fit the GMM to provided samples.

        :param samples: Array of samples to fit the GMM to
        :type samples: numpy.ndarray
        :raises ValueError: If samples array has wrong dimension
        """
        if samples.shape[1] != self.dimension:
            raise ValueError(
                f"Expected samples with dimension = {self.dimension}, "
                f"got samples with dimension = {samples.shape[1]}"
            )
        self._gmm.fit(samples)
        self._gmm_chols = tuple(np.linalg.cholesky(cov) for cov in self._gmm.covariances_)

    def sample(self, num_samples: int = 1) -> npt.NDArray[np.float64]:
        """Generate samples from the fitted GMM.

        :param num_samples: The number of samples to generate, defaults to 1
        :type num_samples: int, optional
        :return: Array of samples from the GMM, clipped to [0, 1]
        :rtype: numpy.ndarray
        :raises ValueError: If the GMM has not been fitted or if sample_type is
            invalid
        """
        if self._gmm_chols is None:
            raise ValueError("The sampler must be fitted before sampling")

        if self._sample_type == SampleType.UNIFORM:
            return self._uniform_sample(num_samples)
        elif self._sample_type == SampleType.SOBOL:
            return self._sobol_sample(num_samples)
        else:
            raise ValueError(f"Invalid sample_type: {self._sample_type}")

    def _uniform_sample(self, num_samples: int) -> npt.NDArray[np.float64]:
        """Generate samples using uniform random sampling.

        :param num_samples: The number of samples to generate
        :type num_samples: int
        :return: Array of samples from the GMM
        :rtype: numpy.ndarray
        """
        samples, _ = self._gmm.sample(num_samples)
        np.clip(samples, 0.0, 1.0, out=samples)
        return samples

    def _sobol_sample(self, num_samples: int) -> npt.NDArray[np.float64]:
        """Generate samples using Sobol sequence sampling.

        :param num_samples: The number of samples to generate
        :type num_samples: int
        :return: Array of samples from the GMM
        :rtype: numpy.ndarray
        :raises ValueError: If Sobol sampler is not initialized
        """
        if self._sobol is None:
            raise ValueError("Sobol sampler not initialized")

        u_samples = self._sobol.random(num_samples)
        components = np.floor(u_samples[:, 0] * self._gmm.n_components).astype(int)
        u_gmm = u_samples[:, 1:]

        means = self._gmm.means_[components]
        chols = np.array([self._gmm_chols[c] for c in components])

        z_gmm = norm.ppf(u_gmm)
        samples = means + np.einsum("nij,nj->ni", chols, z_gmm)
        np.clip(samples, 0.0, 1.0, out=samples)
        return samples


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
        >>> # Create a sampler that transitions from Sobol to GMM
        >>> mixture = create_sobol_to_gmm_sampler(
        ...     dimension=2,
        ...     n_components=2,
        ...     min_explore_samples=10,
        ...     min_fit_samples=5
        ... )
        >>> # Initial exploration phase
        >>> samples = mixture.sample(5)
        >>> # Fit exploitation sampler when elite samples are available
        >>> mixture.fit(elite_samples)
        >>> # Continue sampling - will switch to exploitation if conditions are met
        >>> more_samples = mixture.sample(5)

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
    :raises ValueError: If min_samples values are invalid or if samplers have
        different dimensions
    """

    def __init__(
        self,
        sampler_explore: SamplerProtocol,
        sampler_exploit: SamplerProtocol | None = None,
        min_explore_samples: int = 10,
        min_fit_samples: int = 5,
    ):
        """Initialize mixture sampler.

        :param sampler_explore: Sampler used for exploration phase
        :type sampler_explore: SamplerProtocol
        :param sampler_exploit: Optional sampler used for exploitation phase
        :type sampler_exploit: SamplerProtocol | None
        :param min_explore_samples: Minimum number of samples to generate
            during exploration
        :type min_explore_samples: int
        :param min_fit_samples: Minimum number of samples required before
            fitting exploitation sampler
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

        if sampler_exploit is not None and sampler_exploit.dimension != sampler_explore.dimension:
            raise ValueError("Explore and exploit samplers must have same dimension")

        self._sampler_explore = sampler_explore
        self._sampler_exploit = sampler_exploit
        self._min_explore_samples = min_explore_samples
        self._min_fit_samples = min_fit_samples
        self._sample_count = 0

    @property
    def dimension(self) -> int:
        """Get the dimension of the sampling space.

        :return: Number of dimensions
        :rtype: int
        """
        return self._sampler_explore.dimension

    @property
    def sample_count(self) -> int:
        """Get the total number of samples generated."""
        return self._sample_count

    @property
    def min_explore_samples(self) -> int:
        """Get the minimum number of exploration samples required."""
        return self._min_explore_samples

    @property
    def min_fit_samples(self) -> int:
        """Get the minimum number of samples needed for fitting."""
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
            self._sample_count >= self._min_explore_samples
            and num_elite_samples >= self._min_fit_samples
            and self._sampler_exploit is not None
        )

    def is_using_exploitation(self) -> bool:
        """Check if currently using exploitation sampling.

        :return: True if using exploitation sampler
        :rtype: bool
        """
        return (
            self._sample_count >= self._min_explore_samples
            and self._sampler_exploit is not None
            and self._is_fitted
        )

    def sample(self, num_samples: int = 1) -> npt.NDArray[np.float64]:
        """Generate samples using either exploration or exploitation sampler.

        The sampler automatically switches between exploration and exploitation
        based on the number of samples generated and whether the exploitation
        sampler has been fitted.

        :param num_samples: The number of samples to generate, defaults to 1
        :type num_samples: int, optional
        :return: Array of samples
        :rtype: numpy.ndarray
        """
        sampler = (
            self._sampler_exploit
            if self.is_using_exploitation()
            else self._sampler_explore
        )
        samples = sampler.sample(num_samples)
        self._sample_count += num_samples
        return samples

    def fit(self, samples: npt.NDArray[np.float64]) -> None:
        """Fit the exploitation sampler if possible.

        :param samples: Array of samples to fit the exploitation sampler
        :type samples: npt.NDArray[np.float64]
        :return: True if fitting was successful
        :rtype: bool
        :raises ValueError: If samples array has wrong shape or dimension
        """
        if self._sampler_exploit is None:
            return False

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
            return False

        try:
            self._sampler_exploit.fit(samples)
            self._is_fitted = True
            return True
        except Exception:
            self._is_fitted = False
            return False


    @classmethod
    def create_uniform_to_gmm(
        cls,
        dimension: int,
        n_components: int,
        min_explore_samples: int = 10,
        min_fit_samples: int = 5,
        reg_covar: float = 1e-6,
        sample_type: SampleType = SampleType.UNIFORM,
    ) -> "MixtureSampler":
        """Create a mixture sampler that transitions from uniform to GMM
        sampling.

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
        explore_sampler = UniformSampler(dimension=dimension)
        exploit_sampler = ClippedGaussianMixtureSampler(
            dimension=dimension,
            n_components=n_components,
            reg_covar=reg_covar,
            sample_type=sample_type,
        )

        return cls(
            sampler_explore=explore_sampler,
            sampler_exploit=exploit_sampler,
            min_explore_samples=min_explore_samples,
            min_fit_samples=min_fit_samples,
        )

    @classmethod
    def create_sobol_to_gmm(
        cls,
        dimension: int,
        n_components: int,
        min_explore_samples: int = 10,
        min_fit_samples: int = 5,
        reg_covar: float = 1e-6,
        sample_type: SampleType = SampleType.UNIFORM,
    ) -> "MixtureSampler":
        """Create a mixture sampler that transitions from Sobol to GMM
        sampling.

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
            sample_type=sample_type,
        )

        return cls(
            sampler_explore=explore_sampler,
            sampler_exploit=exploit_sampler,
            min_explore_samples=min_explore_samples,
            min_fit_samples=min_fit_samples,
        )
