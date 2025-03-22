"""
Sampling strategies for hyperparameter optimization in HOLA.

This module implements various sampling strategies for exploring the
hyperparameter space, including uniform, Sobol sequence, Gaussian mixture, and
explore-exploit samplers. All samplers operate in the unit hypercube [0,1]^d,
where d is the dimension of the parameter space.

The samplers follow a common interface defined by the HypercubeSampler
protocol, allowing them to be used interchangeably in the optimization process.
"""

from typing import Protocol, TypeAlias, Union, Dict, Any, Tuple

import numpy as np
import numpy.typing as npt
import scipy.stats
from scipy.stats.qmc import Sobol
from sklearn.mixture import GaussianMixture

from hola.core.utils import uniform_to_category


# Permanent tags for sampler types to ensure serialization compatibility
# if class names change in the future
SAMPLER_TAGS = {
    "uniform": "UniformSampler",
    "sobol": "SobolSampler",
    "gmm": "ClippedGaussianMixtureSampler",
    "explore_exploit": "ExploreExploitSampler"
}

# Reverse mapping for lookup
SAMPLER_CLASSES = {v: k for k, v in SAMPLER_TAGS.items()}


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

    def sample(self, n_samples: int = 1) -> Tuple[npt.NDArray[np.float64], Dict[str, Any]]:
        """
        Generate samples from the unit hypercube.

        :param n_samples: Number of samples to generate
        :type n_samples: int
        :return: Tuple containing:
            - Array of shape (n_samples, dimension) containing samples
            - Dictionary of metadata about the samples
        :rtype: Tuple[npt.NDArray[np.float64], Dict[str, Any]]
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

    def get_state(self) -> Dict[str, Any]:
        """
        Get a serializable representation of the sampler's state.

        :return: Dictionary containing the sampler state
        :rtype: Dict[str, Any]
        """
        ...

    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Restore the sampler's state from a serialized representation.

        :param state: Dictionary containing the sampler state
        :type state: Dict[str, Any]
        """
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

    def sample(self, n_samples: int = 1) -> Tuple[npt.NDArray[np.float64], Dict[str, Any]]:
        """
        Generate uniform random samples from [0,1]^d.

        :param n_samples: Number of samples to generate
        :type n_samples: int
        :return: Tuple containing:
            - Array of shape (n_samples, dimension) containing uniform samples
            - Dictionary of metadata about the samples
        :rtype: Tuple[npt.NDArray[np.float64], Dict[str, Any]]
        :raises ValueError: If n_samples <= 0
        """
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        samples = np.random.rand(n_samples, self.dimension)
        metadata = {
            "sampler_type": "uniform",
            "sampler_class": self.__class__.__name__,
        }
        return samples, metadata

    def fit(self, samples: npt.NDArray[np.float64]) -> None:
        """No-op as uniform sampling is non-adaptive."""
        pass

    def reset(self) -> None:
        """No-op as uniform sampling maintains no state."""
        pass

    def get_state(self) -> Dict[str, Any]:
        """
        Get a serializable representation of the sampler's state.

        Since UniformSampler doesn't maintain state, this just captures
        the dimension.

        :return: Dictionary containing the sampler state
        :rtype: Dict[str, Any]
        """
        return {
            "type": SAMPLER_CLASSES[self.__class__.__name__],
            "dimension": self.dimension
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Restore the sampler's state from a serialized representation.

        For UniformSampler, this is a no-op since it's stateless.

        :param state: Dictionary containing the sampler state
        :type state: Dict[str, Any]
        """
        # No state to restore for uniform sampler
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

    def sample(self, n_samples: int = 1) -> Tuple[npt.NDArray[np.float64], Dict[str, Any]]:
        """
        Generate samples from the Sobol sequence.

        :param n_samples: Number of samples to generate
        :type n_samples: int
        :return: Tuple containing:
            - Array of shape (n_samples, dimension) containing Sobol sequence samples
            - Dictionary of metadata about the samples
        :rtype: Tuple[npt.NDArray[np.float64], Dict[str, Any]]
        :raises ValueError: If n_samples <= 0
        """
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")
        samples = self._sampler.random(n_samples).astype(np.float64)
        metadata = {
            "sampler_type": "quasi-random",
            "sampler_class": self.__class__.__name__,
            "sequence_type": "sobol",
        }
        return samples, metadata

    def fit(self, samples: npt.NDArray[np.float64]) -> None:
        """No-op as Sobol sampling is non-adaptive."""
        pass

    def reset(self) -> None:
        """Reset the Sobol sequence to its initial state."""
        self._sampler.reset()

    def get_state(self) -> Dict[str, Any]:
        """
        Get a serializable representation of the sampler's state.

        For SobolSampler, the state includes the dimension and the
        number of samples generated so far.

        :return: Dictionary containing the sampler state
        :rtype: Dict[str, Any]
        """
        return {
            "type": SAMPLER_CLASSES[self.__class__.__name__],
            "dimension": self.dimension,
            # Store the Sobol sampler's state - specifically the number of points generated
            "num_generated": int(self._sampler.num_generated)
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Restore the sampler's state from a serialized representation.

        For SobolSampler, this restores the state of the Sobol sequence.

        :param state: Dictionary containing the sampler state
        :type state: Dict[str, Any]
        """
        # Create a new Sobol sampler
        self._sampler = Sobol(self.dimension)

        # Fast-forward the sequence by generating and discarding points
        num_generated = state.get("num_generated", 0)
        if num_generated > 0:
            # Generate dummy samples to advance the sequence
            chunk_size = min(10000, num_generated)  # Process in chunks to avoid memory issues
            remaining = num_generated

            while remaining > 0:
                current_chunk = min(chunk_size, remaining)
                _ = self._sampler.random(current_chunk)
                remaining -= current_chunk


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
    ):
        """
        Initialize the GMM sampler.

        :param dimension: Dimension of the sampling space
        :type dimension: int
        :param n_components: Number of Gaussian components in the mixture
        :type n_components: int
        :param reg_covar: Regularization for component covariances
        :type reg_covar: float
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

        self._hypercube_sampler = UniformSampler(dimension + 1)  # Sobol' does not really make sense here because the GMM may be refitted

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

    def sample(self, n_samples: int = 1) -> Tuple[npt.NDArray[np.float64], Dict[str, Any]]:
        """
        Generate samples from the fitted GMM, clipped to [0,1]^d.

        :param n_samples: Number of samples to generate
        :type n_samples: int
        :return: Tuple containing:
            - Array of shape (n_samples, dimension) containing samples
            - Dictionary of metadata about the samples
        :rtype: Tuple[npt.NDArray[np.float64], Dict[str, Any]]
        :raises ValueError: If sampler hasn't been fitted yet
        """
        if self._gmm_chols is None:
            raise ValueError("Call fit(...) before generating samples.")

        # Sample from the (d+1)-dim hypercube to pick mixture comp + latents
        u_samples, _ = self._hypercube_sampler.sample(n_samples)

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

        metadata = {
            "sampler_type": "adaptive",
            "sampler_class": self.__class__.__name__,
            "model_type": "gaussian_mixture",
            "n_components": self.n_components,
            "is_fitted": True,
            "components_used": comps.tolist(),
        }

        return samples, metadata

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

    def get_state(self) -> Dict[str, Any]:
        """
        Get a serializable representation of the sampler's state.

        For ClippedGaussianMixtureSampler, the state includes the dimension,
        number of components, regularization, and the fitted GMM parameters
        if available.

        :return: Dictionary containing the sampler state
        :rtype: Dict[str, Any]
        """
        state = {
            "type": SAMPLER_CLASSES[self.__class__.__name__],
            "dimension": self.dimension,
            "n_components": self.n_components,
            "reg_covar": self._reg_covar,
            "is_fitted": self._gmm_means is not None
        }

        # Save fitted model parameters if available
        if self._gmm_means is not None:
            state.update({
                "weights": self._gmm.weights_.tolist(),
                "means": self._gmm_means.tolist(),
                "covariances": self._gmm.covariances_.tolist(),
            })

        return state

    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Restore the sampler's state from a serialized representation.

        For ClippedGaussianMixtureSampler, this restores the fitted GMM if available.

        :param state: Dictionary containing the sampler state
        :type state: Dict[str, Any]
        """
        # Initialize the GMM
        self._gmm = GaussianMixture(
            n_components=self._n_components,
            reg_covar=self._reg_covar,
            covariance_type="full",
        )

        # If the GMM was fitted, restore its parameters
        if state.get("is_fitted", False):
            weights = np.array(state["weights"])
            means = np.array(state["means"])
            covariances = np.array(state["covariances"])

            # Set GMM parameters directly
            self._gmm.weights_ = weights
            self._gmm.means_ = means
            self._gmm.covariances_ = covariances
            self._gmm.precisions_cholesky_ = np.array([
                np.linalg.cholesky(np.linalg.inv(cov)) for cov in covariances
            ])

            # Set the derived parameters used by the sampler
            self._gmm_means = means
            self._gmm_chols = np.array([np.linalg.cholesky(cov) for cov in covariances])

            # Initialize other required GMM attributes
            self._gmm.converged_ = True
            self._gmm.n_iter_ = 1


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
    ):
        """
        Initialize the explore-exploit sampler.

        :param explore_sampler: Sampler to use during exploration phase
        :type explore_sampler: HypercubeSampler
        :param exploit_sampler: Sampler to use during exploitation phase
        :type exploit_sampler: HypercubeSampler
        :raises ValueError: If samplers have different dimensions
        """
        if explore_sampler.dimension != exploit_sampler.dimension:
            raise ValueError("Explore and exploit samplers must have the same dimensions")

        self._explore_sampler = explore_sampler
        self._exploit_sampler = exploit_sampler
        self._is_fitted = False

    @property
    def dimension(self) -> int:
        """
        :return: Dimension of the sampling space
        :rtype: int
        """
        return self._explore_sampler.dimension

    def is_using_exploitation(self) -> bool:
        """
        Check if currently in exploitation phase.

        :return: True if using exploitation sampler, False if exploring
        :rtype: bool
        """
        return self._is_fitted

    def sample(self, n_samples: int = 1) -> Tuple[npt.NDArray[np.float64], Dict[str, Any]]:
        """
        Generate samples using either exploration or exploitation strategy.

        :param n_samples: Number of samples to generate
        :type n_samples: int
        :return: Tuple containing:
            - Array of shape (n_samples, dimension) containing samples
            - Dictionary of metadata about the samples
        :rtype: Tuple[npt.NDArray[np.float64], Dict[str, Any]]
        """
        is_exploit = self.is_using_exploitation()
        sampler = self._exploit_sampler if is_exploit else self._explore_sampler

        samples, inner_metadata = sampler.sample(n_samples)

        metadata = {
            "sampler_type": "explore_exploit",
            "sampler_class": self.__class__.__name__,
            "phase": "exploit" if is_exploit else "explore",
            "inner_sampler": inner_metadata,
        }

        return samples, metadata

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

        try:
            self._exploit_sampler.fit(samples)
            self._is_fitted = True
        except (ValueError, np.linalg.LinAlgError) as e:
            # Warn the user
            print(f"Failed to fit exploit sampler: {str(e)}")
            self._is_fitted = False

    def reset(self) -> None:
        """Reset the sampler to its initial state."""
        self._is_fitted = False
        self._explore_sampler.reset()
        self._exploit_sampler.reset()

    def get_state(self) -> Dict[str, Any]:
        """
        Get a serializable representation of the sampler's state.

        For ExploreExploitSampler, this includes the state of both the explore
        and exploit samplers, as well as whether exploitation is being used.

        :return: Dictionary containing the sampler state
        :rtype: Dict[str, Any]
        """
        return {
            "type": SAMPLER_CLASSES[self.__class__.__name__],
            "is_fitted": self._is_fitted,
            "explore_sampler": self._explore_sampler.get_state(),
            "exploit_sampler": self._exploit_sampler.get_state()
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Restore the sampler's state from a serialized representation.

        For ExploreExploitSampler, this restores the state of both sub-samplers
        and the exploitation flag.

        :param state: Dictionary containing the sampler state
        :type state: Dict[str, Any]
        """
        self._is_fitted = state.get("is_fitted", False)

        # Restore the state of both sub-samplers
        self._explore_sampler.set_state(state.get("explore_sampler", {}))
        self._exploit_sampler.set_state(state.get("exploit_sampler", {}))


PredefinedSamplers: TypeAlias = Union[
    UniformSampler,
    SobolSampler,
    ClippedGaussianMixtureSampler,
    ExploreExploitSampler,
]
"""Type alias for all predefined sampler classes in this module."""

def create_sampler_from_state(state: Dict[str, Any]) -> HypercubeSampler:
    """
    Create a sampler instance from a serialized state.

    This function reconstructs the appropriate sampler type based on the
    serialized state dictionary and initializes it with the saved parameters.

    :param state: Dictionary containing the sampler state
    :type state: Dict[str, Any]
    :return: Initialized sampler with restored state
    :rtype: HypercubeSampler
    :raises ValueError: If the sampler type is not recognized
    """
    sampler_tag = state.get("type")
    dimension = state.get("dimension")

    # Get the class name from the tag
    sampler_type = SAMPLER_TAGS.get(sampler_tag)
    if not sampler_type:
        # Fallback for backward compatibility - might be using the class name directly
        sampler_type = sampler_tag

    if sampler_type == "ExploreExploitSampler":
        # For ExploreExploitSampler, we need to handle sub-samplers first
        explore_state = state.get("explore_sampler", {})
        exploit_state = state.get("exploit_sampler", {})

        # Check if both sub-sampler states are valid
        if not explore_state.get("type") or not explore_state.get("dimension"):
            raise ValueError("Invalid explore_sampler state in ExploreExploitSampler")

        if not exploit_state.get("type") or not exploit_state.get("dimension"):
            raise ValueError("Invalid exploit_sampler state in ExploreExploitSampler")

        # Create the sub-samplers
        explore_sampler = create_sampler_from_state(explore_state)
        exploit_sampler = create_sampler_from_state(exploit_state)

        # Create the ExploreExploitSampler
        sampler = ExploreExploitSampler(
            explore_sampler=explore_sampler,
            exploit_sampler=exploit_sampler
        )

        # Set the is_fitted flag
        sampler._is_fitted = state.get("is_fitted", False)

        return sampler

    # For other sampler types, check required fields
    if not sampler_type or not dimension:
        raise ValueError(f"Invalid sampler state: missing type or dimension. State: {state}")

    if sampler_type == "UniformSampler":
        sampler = UniformSampler(dimension)

    elif sampler_type == "SobolSampler":
        sampler = SobolSampler(dimension)

    elif sampler_type == "ClippedGaussianMixtureSampler":
        n_components = state.get("n_components", 1)
        reg_covar = state.get("reg_covar", 1e-6)
        sampler = ClippedGaussianMixtureSampler(
            dimension=dimension,
            n_components=n_components,
            reg_covar=reg_covar
        )

    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")

    # Restore the sampler's state
    sampler.set_state(state)

    return sampler
