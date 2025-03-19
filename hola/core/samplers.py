"""
Sampling strategies for hyperparameter optimization in HOLA.

This module implements various sampling strategies for exploring the
hyperparameter space, including uniform, Sobol sequence, Gaussian mixture, and
explore-exploit samplers. All samplers operate in the unit hypercube [0,1]^d,
where d is the dimension of the parameter space.

The samplers follow a common interface defined by the HypercubeSampler
protocol, allowing them to be used interchangeably in the optimization process.
"""

import logging
from typing import Protocol, TypeAlias, Union

import numpy as np
import numpy.typing as npt
import scipy.stats
from scipy.stats.qmc import Sobol
from sklearn.mixture import GaussianMixture

from hola.core.utils import uniform_to_category

logger = logging.getLogger(__name__)


class HypercubeSampler(Protocol):
    """
    Protocol defining the interface for hypercube sampling strategies.

    All samplers must operate in the unit hypercube [0,1]^d and implement
    methods for sampling, fitting to data, and resetting internal state.
    """

    @property
    def dimension(self) -> int:
        """
        The dimension of the sampling space.

        :return: Number of dimensions in the unit hypercube
        :rtype: int
        """
        ...

    def sample(self, n_samples: int = 1) -> npt.NDArray[np.float64]:
        """
        Generate samples from the unit hypercube.

        :param n_samples: Number of samples to generate
        :type n_samples: int
        :return: Array of shape (n_samples, dimension) containing samples
        :rtype: npt.NDArray[np.float64]
        """
        ...

    def fit(self, samples: npt.NDArray[np.float64]) -> None:
        """
        Update the sampler's state based on observed samples.

        :param samples: Array of shape (n_samples, dimension) containing
            previous samples
        :type samples: npt.NDArray[np.float64]
        """
        ...

    def reset(self) -> None:
        """Reset the sampler's internal state."""
        ...


class UniformSampler:
    """
    Simple uniform random sampler over the unit hypercube.

    This sampler draws samples independently and uniformly from [0,1]^d.
    """

    def __init__(self, dimension: int):
        """
        Initialize the uniform sampler.

        :param dimension: Dimension of the sampling space
        :type dimension: int
        :raises ValueError: If dimension <= 0
        """
        if dimension <= 0:
            raise ValueError("dimension must be positive.")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        """
        :return: Dimension of the sampling space
        :rtype: int
        """
        return self._dimension

    def sample(self, n_samples: int = 1) -> npt.NDArray[np.float64]:
        """
        Generate uniform random samples from [0,1]^d.

        :param n_samples: Number of samples to generate
        :type n_samples: int
        :return: Array of shape (n_samples, dimension) containing uniform
            samples
        :rtype: npt.NDArray[np.float64]
        :raises ValueError: If n_samples <= 0
        """
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        return np.random.rand(n_samples, self.dimension)

    def fit(self, samples: npt.NDArray[np.float64]) -> None:
        """No-op as uniform sampling is non-adaptive."""
        pass

    def reset(self) -> None:
        """No-op as uniform sampling maintains no state."""
        pass


class SobolSampler:
    """
    Quasi-random sampler using Sobol sequences.

    This sampler generates low-discrepancy sequences that provide better space
    coverage than uniform random sampling.
    """

    def __init__(self, dimension: int, sampler: Sobol | None = None):
        """
        Initialize the Sobol sequence sampler.

        :param dimension: Dimension of the sampling space
        :type dimension: int
        :param sampler: Pre-configured Sobol sampler, or None to create new
        :type sampler: Sobol | None
        :raises ValueError: If dimension <= 0
        """
        if dimension <= 0:
            raise ValueError("dimension must be positive.")
        self._dimension = dimension
        self._sampler = sampler if sampler is not None else Sobol(dimension)

    @property
    def dimension(self) -> int:
        """
        :return: Dimension of the sampling space
        :rtype: int
        """
        return self._dimension

    def sample(self, n_samples: int = 1) -> npt.NDArray[np.float64]:
        """
        Generate samples from the Sobol sequence.

        :param n_samples: Number of samples to generate
        :type n_samples: int
        :return: Array of shape (n_samples, dimension) containing Sobol
            sequence samples
        :rtype: npt.NDArray[np.float64]
        :raises ValueError: If n_samples <= 0
        """
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        return self._sampler.random(n_samples).astype(np.float64)

    def fit(self, samples: npt.NDArray[np.float64]) -> None:
        """No-op as Sobol sampling is non-adaptive."""
        pass

    def reset(self) -> None:
        """Reset the Sobol sequence to its initial state."""
        self._sampler.reset()


class ClippedGaussianMixtureSampler:
    """
    Adaptive sampler using a Gaussian Mixture Model (GMM) clipped to the unit
    hypercube.

    This sampler fits a GMM to observed samples and generates new samples by:

    1. Selecting a mixture component
    2. Sampling from the chosen Gaussian
    3. Clipping the samples to [0,1]^d
    """

    def __init__(
        self,
        dimension: int,
        n_components: int,
        reg_covar: float = 1e-6,
        hypercube_sampler: HypercubeSampler | None = None,  # TODO: add a static HypercubeSampler protocol
    ):
        """
        Initialize the GMM sampler.

        :param dimension: Dimension of the sampling space
        :type dimension: int
        :param n_components: Number of Gaussian components in the mixture
        :type n_components: int
        :param reg_covar: Regularization for component covariances
        :type reg_covar: float
        :param hypercube_sampler: Sampler for component selection and latent
            space
        :type hypercube_sampler: HypercubeSampler | None
        :raises ValueError: If dimension <= 0, n_components <= 0, reg_covar <
            0, or if hypercube_sampler dimension != dimension + 1
        """
        if dimension <= 0:
            raise ValueError("dimension must be positive.")
        if n_components <= 0:
            raise ValueError("n_components must be positive.")
        if reg_covar < 0:
            raise ValueError("reg_covar must be non-negative.")

        self._gmm = GaussianMixture(
            n_components=n_components,
            reg_covar=reg_covar,
            covariance_type="full",
        )
        self._gmm_means: npt.NDArray[np.float64] | None = None
        self._gmm_chols: npt.NDArray[np.float64] | None = None
        self._n_components = n_components
        self._reg_covar = reg_covar
        self._dimension = dimension

        if hypercube_sampler is None:
            hypercube_sampler = UniformSampler(dimension + 1)
        if hypercube_sampler.dimension != dimension + 1:
            raise ValueError("The hypercube_sampler must have dimension+1 for mixture selection.")
        self._hypercube_sampler = hypercube_sampler

    @property
    def dimension(self) -> int:
        """
        :return: Dimension of the sampling space
        :rtype: int
        """
        return self._dimension

    @property
    def n_components(self) -> int:
        """
        :return: Number of Gaussian components in the mixture
        :rtype: int
        """
        return self._n_components

    def sample(self, n_samples: int = 1) -> npt.NDArray[np.float64]:
        """
        Generate samples from the fitted GMM, clipped to [0,1]^d.

        :param n_samples: Number of samples to generate
        :type n_samples: int
        :return: Array of shape (n_samples, dimension) containing samples
        :rtype: npt.NDArray[np.float64]
        :raises ValueError: If sampler hasn't been fitted yet
        """
        if self._gmm_chols is None:
            raise ValueError("Call fit(...) before generating samples.")

        # Sample from the (d+1)-dim hypercube to pick mixture comp + latents
        u_samples = self._hypercube_sampler.sample(n_samples)

        # Component selection from the first coordinate
        comps = uniform_to_category(u_samples[:, 0], self.n_components)

        # Remaining d coords for latent GMM sampling
        u_gmms = u_samples[:, 1:]

        # Convert uniform draws -> normal draws via inverse CDF
        z_gmms = scipy.stats.norm.ppf(u_gmms)

        # Mean + Cholesky to transform normal draws
        means = self._gmm_means[comps]
        chols = self._gmm_chols[comps]
        samples = means + np.einsum("kij,kj->ki", chols, z_gmms)

        # Clip to [0,1]^d
        np.clip(samples, 0.0, 1.0, out=samples)
        return samples

    def fit(self, samples: npt.NDArray[np.float64]) -> None:
        """
        Fit the GMM to the provided samples.

        :param samples: Array of shape (n_samples, dimension) containing
            samples
        :type samples: npt.NDArray[np.float64]
        :raises ValueError: If samples have wrong shape or lie outside [0,1]^d
        """
        if samples.ndim != 2 or samples.shape[1] != self.dimension:
            raise ValueError("samples must be of shape (n, dimension).")
        if np.any(samples < 0) or np.any(samples > 1):
            raise ValueError("Samples must lie in [0,1]^d.")

        self._gmm.fit(samples)
        self._gmm_means = self._gmm.means_
        self._gmm_chols = np.linalg.cholesky(self._gmm.covariances_)

    def reset(self) -> None:
        """Reset the sampler to its initial state."""
        self._gmm = GaussianMixture(
            n_components=self._n_components,
            reg_covar=self._reg_covar,
            covariance_type="full",
        )
        self._gmm_chols = None
        self._gmm_means = None
        self._hypercube_sampler.reset()


class ExploreExploitSampler:
    """
    Two-phase sampler that switches from exploration to exploitation.

    This sampler initially uses an exploration strategy, then switches to an
    exploitation strategy once sufficient samples have been collected and the
    exploit sampler has been successfully fitted to elite samples.
    """

    def __init__(
        self,
        explore_sampler: HypercubeSampler,
        exploit_sampler: HypercubeSampler,
        min_explore_samples: int = 10,
        min_fit_samples: int = 5,
    ):
        """
        Initialize the explore-exploit sampler.

        :param explore_sampler: Sampler to use during exploration phase
        :type explore_sampler: HypercubeSampler
        :param exploit_sampler: Sampler to use during exploitation phase
        :type exploit_sampler: HypercubeSampler
        :param min_explore_samples: Minimum samples before exploitation can
            begin
        :type min_explore_samples: int
        :param min_fit_samples: Minimum elite samples needed to fit exploit
            sampler
        :type min_fit_samples: int
        :raises ValueError: If min_explore_samples <= 0, min_fit_samples <= 0,
            min_fit_samples > min_explore_samples, or if samplers have
            different dimensions
        """
        if min_explore_samples <= 0:
            raise ValueError("min_explore_samples must be positive.")
        if min_fit_samples <= 0:
            raise ValueError("min_fit_samples must be positive.")
        if min_fit_samples > min_explore_samples:
            raise ValueError("min_fit_samples cannot exceed min_explore_samples.")
        if exploit_sampler.dimension != explore_sampler.dimension:
            raise ValueError("Both samplers must have the same dimension.")

        self._explore_sampler = explore_sampler
        self._exploit_sampler = exploit_sampler
        self._min_explore_samples = min_explore_samples
        self._min_fit_samples = min_fit_samples

        self._generated_samples = 0
        self._is_fitted = False

    @property
    def dimension(self) -> int:
        """
        :return: Dimension of the sampling space
        :rtype: int
        """
        return self._explore_sampler.dimension

    @property
    def sample_count(self) -> int:
        """
        :return: Total number of samples generated
        :rtype: int
        """
        return self._generated_samples

    @property
    def min_explore_samples(self) -> int:
        """
        :return: Minimum samples required before exploitation can begin
        :rtype: int
        """
        return self._min_explore_samples

    @property
    def min_fit_samples(self) -> int:
        """
        :return: Minimum elite samples needed to fit exploit sampler
        :rtype: int
        """
        return self._min_fit_samples

    def is_ready_to_fit(self, num_elite_samples: int) -> bool:
        """
        Check if the sampler is ready to fit the exploitation strategy.

        :param num_elite_samples: Number of elite samples available
        :type num_elite_samples: int
        :return: True if ready to fit, False otherwise
        :rtype: bool
        """
        return (
            self.sample_count >= self.min_explore_samples
            and num_elite_samples >= self.min_fit_samples
        )

    def is_using_exploitation(self) -> bool:
        """
        Check if currently in exploitation phase.

        :return: True if using exploitation sampler, False if exploring
        :rtype: bool
        """
        return self.sample_count >= self.min_explore_samples and self._is_fitted

    def sample(self, n_samples: int = 1) -> npt.NDArray[np.float64]:
        """
        Generate samples using either exploration or exploitation strategy.

        :param n_samples: Number of samples to generate
        :type n_samples: int
        :return: Array of shape (n_samples, dimension) containing samples
        :rtype: npt.NDArray[np.float64]
        """
        if self.is_using_exploitation():
            sampler = self._exploit_sampler
        else:
            sampler = self._explore_sampler

        results = sampler.sample(n_samples)
        self._generated_samples += n_samples
        return results

    def fit(self, samples: npt.NDArray[np.float64]) -> None:
        """
        Attempt to fit the exploitation sampler to the provided samples.

        :param samples: Array of shape (n_samples, dimension) containing elite
            samples
        :type samples: npt.NDArray[np.float64]
        :raises ValueError: If samples have wrong shape
        """
        if samples.ndim != 2 or samples.shape[1] != self.dimension:
            raise ValueError("samples must be 2D and match the sampler dimension.")

        if not self.is_ready_to_fit(len(samples)):
            return

        try:
            self._exploit_sampler.fit(samples)
            self._is_fitted = True
        except (ValueError, np.linalg.LinAlgError) as e:
            logger.warning(f"Failed to fit exploit sampler: {str(e)}")
            self._is_fitted = False

    def reset(self) -> None:
        """Reset the sampler to its initial state."""
        self._generated_samples = 0
        self._is_fitted = False
        self._explore_sampler.reset()
        self._exploit_sampler.reset()


PredefinedSamplers: TypeAlias = Union[
    UniformSampler,
    SobolSampler,
    ClippedGaussianMixtureSampler,
    ExploreExploitSampler,
]
"""Type alias for all predefined sampler classes in this module."""
