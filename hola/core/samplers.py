from typing import Protocol, TypeAlias, Union

import numpy as np
import numpy.typing as npt
import scipy.stats
from scipy.stats.qmc import Sobol
from sklearn.mixture import GaussianMixture

from hola.core.utils import uniform_to_category


class HypercubeSampler(Protocol):
    @property
    def dimension(self) -> int: ...

    def sample(self, n_samples: int = 1) -> npt.NDArray[np.float64]: ...

    def fit(self, samples: npt.NDArray[np.float64]) -> None: ...

    def reset(self) -> None: ...


class UniformSampler:
    def __init__(self, dimension: int):
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def sample(self, n_samples: int = 1) -> npt.NDArray[np.float64]:
        return np.random.rand(n_samples, self.dimension)

    def fit(self, samples: npt.NDArray[np.float64]) -> None:
        pass

    def reset(self) -> None:
        pass


class SobolSampler:
    def __init__(self, dimension: int, sampler: Sobol | None = None):
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._dimension = dimension
        if sampler is None:
            sampler = Sobol(dimension)
        self._sampler = sampler

    @property
    def dimension(self) -> int:
        return self._dimension

    def sample(self, n_samples: int = 1) -> npt.NDArray[np.float64]:
        return self._sampler.random(n_samples)

    def fit(self, samples: npt.NDArray[np.float64]) -> None:
        pass

    def reset(self) -> None:
        self._sampler.reset()


class ClippedGaussianMixtureSampler:
    def __init__(
        self,
        dimension: int,
        n_components: int,
        reg_covar: float = 1e-6,
        hypercube_sampler: HypercubeSampler | None = None,
    ):
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
        self._gmm_chols: npt.NDArray[np.float64] | None = None
        self._n_components = n_components
        self._reg_covar = reg_covar

        # Sampler data
        self._dimension = dimension
        if hypercube_sampler is None:
            hypercube_sampler = UniformSampler(dimension + 1)
        if hypercube_sampler.dimension != dimension + 1:
            raise ValueError(f"hypercube_sampler must have one more dimension than dimension")
        self._hypercube_sampler = hypercube_sampler

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def n_components(self) -> int:
        return self._n_components

    def sample(self, n_samples: int = 1) -> npt.NDArray[np.float64]:
        if self._gmm_chols is None:
            raise ValueError("The sampler must be fitted before sampling")

        u_samples = self._hypercube_sampler.sample(n_samples)
        components = uniform_to_category(u_samples[:, 0])
        u_gmms = u_samples[:, 1:]

        means = self._gmm.means_[components]
        chols = self._gmm_chols[components]

        z_gmms = scipy.stats.norm.ppf(u_gmms)
        samples = means + np.einsum("kij,kj->ki", chols, z_gmms)
        np.clip(samples, 0.0, 1.0, out=samples)
        return samples

    def fit(self, samples: npt.NDArray[np.float64]) -> None:
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
        self._gmm.fit(samples)
        self._gmm_chols = np.linalg.cholesky(self._gmm.covariances_)

    def reset(self) -> None:
        self._gmm = GaussianMixture(
            n_components=self.n_components, reg_covar=self._reg_covar, covariance_type="full"
        )
        self._gmm_chols = None
        self._hypercube_sampler.reset()


class ExploreExploitSampler:
    def __init__(
        self,
        explore_sampler: HypercubeSampler,
        exploit_sampler: HypercubeSampler,
        min_explore_samples: int = 10,
        min_fit_samples: int = 5,
    ):
        if min_explore_samples <= 0:
            raise ValueError("min_explore_samples must be positive")
        if min_fit_samples <= 0:
            raise ValueError("min_fit_samples must be positive")
        if min_fit_samples > min_explore_samples:
            raise ValueError("min_fit_samples cannot be greater than min_explore_samples")
        if exploit_sampler.dimension != explore_sampler.dimension:
            raise ValueError("Explore and exploit samplers must have same dimension")

        self._explore_sampler = explore_sampler
        self._exploit_sampler = exploit_sampler
        self._min_explore_samples = min_explore_samples
        self._min_fit_samples = min_fit_samples
        self._generated_samples = 0
        self._is_fitted = False

    @property
    def dimension(self) -> int:
        return self._explore_sampler.dimension

    @property
    def sample_count(self) -> int:
        return self._generated_samples

    @property
    def min_explore_samples(self) -> int:
        return self._min_explore_samples

    @property
    def min_fit_samples(self) -> int:
        return self._min_fit_samples

    def is_ready_to_fit(self, num_elite_samples: int) -> bool:
        return (
            self.sample_count >= self.min_explore_samples
            and num_elite_samples >= self.min_fit_samples
        )

    def is_using_exploitation(self) -> bool:
        return self.sample_count >= self.min_explore_samples and self._is_fitted

    def sample(self, n_samples: int = 1) -> npt.NDArray[np.float64]:
        sampler = self._exploit_sampler if self.is_using_exploitation() else self._explore_sampler
        samples = sampler.sample(n_samples)
        # TODO should we increment generated samples here, or after these
        # parameter choices have been evaluated?
        self._generated_samples += n_samples
        return samples

    def fit(self, samples: npt.NDArray[np.float64]) -> None:
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
        if len(samples) < self.min_fit_samples:
            return

        try:
            self._exploit_sampler.fit(samples)
            self._is_fitted = True
        except Exception:
            self._is_fitted = False

    def reset(self) -> None:
        self._generated_samples = 0
        self._is_fitted = False
        self._explore_sampler.reset()
        self._exploit_sampler.reset()


PredefinedSamplers: TypeAlias = Union[
    UniformSampler, SobolSampler, ClippedGaussianMixtureSampler, ExploreExploitSampler
]
