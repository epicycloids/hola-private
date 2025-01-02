"""Tests for sampling utilities."""

import numpy as np
import pytest
from scipy.stats import kstest

from hola.core.sample import (
    ClippedGaussianMixtureSampler,
    MixtureSampler,
    SampleType,
    SobolSampler,
    UniformSampler,
)


class TestUniformSampler:
    @pytest.mark.parametrize("dimension", [1, 3, 10])
    def test_instantiation_valid(self, dimension: int) -> None:
        sampler = UniformSampler(dimension=dimension)
        assert sampler.dimension == dimension

    @pytest.mark.parametrize("dimension", [0, -1])
    def test_instantiation_invalid(self, dimension: int) -> None:
        with pytest.raises(ValueError):
            UniformSampler(dimension=dimension)

    def test_sample(self) -> None:
        sampler = UniformSampler(dimension=3)
        sample = sampler.sample()
        assert isinstance(sample, np.ndarray)
        assert sample.shape == (3,)
        assert np.all(sample >= 0) and np.all(sample <= 1)

    def test_sample_distribution(self) -> None:
        """Test if samples follow uniform distribution by generating multiple samples"""
        sampler = UniformSampler(dimension=2)
        samples = np.array([sampler.sample() for _ in range(1000)])

        assert np.all(samples >= 0) and np.all(samples <= 1)

        # Check basic statistical properties
        # Mean should be close to 0.5
        assert 0.4 < np.mean(samples) < 0.6
        # Standard deviation should be close to 1/sqrt(12) ≈ 0.289
        assert 0.2 < np.std(samples) < 0.4
        # Min and max should cover most of [0,1]
        assert np.min(samples) < 0.1
        assert np.max(samples) > 0.9

    def test_fit_noop(self) -> None:
        sampler = UniformSampler(dimension=2)
        dummy_samples = np.random.random((10, 2))
        sampler.fit(dummy_samples)


class TestSobolSampler:
    @pytest.mark.parametrize("dimension", [1, 3, 10])
    def test_instantiation_valid(self, dimension: int) -> None:
        sampler = SobolSampler(dimension=dimension)
        assert sampler.dimension == dimension

    @pytest.mark.parametrize("dimension", [0, -1])
    def test_instantiation_invalid(self, dimension: int) -> None:
        with pytest.raises(ValueError):
            SobolSampler(dimension=dimension)

    def test_sample(self) -> None:
        sampler = SobolSampler(dimension=3)
        sample = sampler.sample()
        assert isinstance(sample, np.ndarray)
        assert sample.shape == (3,)
        assert np.all(sample >= 0) and np.all(sample <= 1)

    def test_samples_are_sequential(self) -> None:
        """Test if sequential samples follow Sobol sequence"""
        sampler = SobolSampler(dimension=2)
        sample1 = sampler.sample()
        sample2 = sampler.sample()

        # Verify samples are different
        assert not np.array_equal(sample1, sample2)

        # Verify samples are within bounds
        assert np.all(sample1 >= 0) and np.all(sample1 <= 1)
        assert np.all(sample2 >= 0) and np.all(sample2 <= 1)

    def test_fit_noop(self) -> None:
        sampler = SobolSampler(dimension=2)
        dummy_samples = np.random.random((10, 2))
        sampler.fit(dummy_samples)


class TestClippedGaussianMixtureSampler:
    @pytest.fixture
    def valid_samples(self) -> np.ndarray:
        return np.random.random((20, 2))

    @pytest.mark.parametrize(
        "dimension, n_components, reg_covar, sample_type",
        [
            (2, 2, 1e-6, SampleType.UNIFORM),
            (3, 1, 1e-3, SampleType.SOBOL),
            (5, 3, 1e-6, SampleType.UNIFORM),
        ],
    )
    def test_instantiation_valid(
        self, dimension: int, n_components: int, reg_covar: float, sample_type: SampleType
    ) -> None:
        sampler = ClippedGaussianMixtureSampler(
            dimension=dimension,
            n_components=n_components,
            reg_covar=reg_covar,
            sample_type=sample_type,
        )
        assert sampler.dimension == dimension

    @pytest.mark.parametrize(
        "dimension, n_components, reg_covar",
        [
            (0, 2, 1e-6),  # Invalid dimension
            (2, 0, 1e-6),  # Invalid n_components
            (2, 2, -1e-6),  # Invalid reg_covar
        ],
    )
    def test_instantiation_invalid(
        self, dimension: int, n_components: int, reg_covar: float
    ) -> None:
        with pytest.raises(ValueError):
            ClippedGaussianMixtureSampler(
                dimension=dimension,
                n_components=n_components,
                reg_covar=reg_covar,
            )

    def test_sample_without_fit(self) -> None:
        sampler = ClippedGaussianMixtureSampler(dimension=2, n_components=2)
        with pytest.raises(ValueError):
            sampler.sample()

    @pytest.mark.parametrize("sample_type", [SampleType.UNIFORM, SampleType.SOBOL])
    def test_sample_after_fit(self, sample_type: SampleType, valid_samples: np.ndarray) -> None:
        sampler = ClippedGaussianMixtureSampler(
            dimension=2, n_components=2, sample_type=sample_type
        )
        sampler.fit(valid_samples)

        sample = sampler.sample()
        assert sample.shape == (2,)
        assert np.all(sample >= 0) and np.all(sample <= 1)

    def test_fit_invalid_dimension(self) -> None:
        sampler = ClippedGaussianMixtureSampler(dimension=3, n_components=2)
        invalid_samples = np.random.random((10, 2))  # Wrong dimension
        with pytest.raises(ValueError):
            sampler.fit(invalid_samples)


class TestMixtureSampler:
    @pytest.fixture
    def mixture_sampler(self) -> MixtureSampler:
        return MixtureSampler.create_sobol_to_gmm(
            dimension=2,
            n_components=2,
            min_explore_samples=10,
            min_fit_samples=5,
        )

    def test_instantiation_valid(self) -> None:
        sampler = MixtureSampler.create_sobol_to_gmm(dimension=2, n_components=2)
        assert sampler.dimension == 2
        assert sampler.min_explore_samples == 10
        assert sampler.min_fit_samples == 5

    def test_instantiation_invalid_min_samples(self) -> None:
        with pytest.raises(ValueError):
            MixtureSampler.create_sobol_to_gmm(
                dimension=2,
                n_components=2,
                min_explore_samples=5,
                min_fit_samples=10,  # min_fit_samples > min_explore_samples
            )

    def test_sample_exploration_phase(self, mixture_sampler: MixtureSampler) -> None:
        sample = mixture_sampler.sample()
        assert sample.shape == (2,)
        assert mixture_sampler.sample_count == 1
        assert not mixture_sampler.is_using_exploitation()

    def test_transition_to_exploitation(self, mixture_sampler: MixtureSampler) -> None:
        # Generate exploration samples
        for _ in range(mixture_sampler.min_explore_samples):
            mixture_sampler.sample()

        # Provide sufficient elite samples
        elite_samples = np.random.random((mixture_sampler.min_fit_samples, 2))
        mixture_sampler.fit(elite_samples)

        # Verify transition
        assert mixture_sampler.is_using_exploitation()

        # Sample in exploitation phase
        sample = mixture_sampler.sample()
        assert sample.shape == (2,)
        assert np.all(sample >= 0) and np.all(sample <= 1)

    def test_insufficient_samples_no_transition(self, mixture_sampler: MixtureSampler) -> None:
        # Generate some samples but less than min_explore_samples
        for _ in range(5):
            mixture_sampler.sample()

        # Try to fit with sufficient elite samples
        elite_samples = np.random.random((mixture_sampler.min_fit_samples, 2))
        mixture_sampler.fit(elite_samples)

        # Should still be in exploration phase
        assert not mixture_sampler.is_using_exploitation()

    def test_factory_methods(self) -> None:
        uniform_mixture = MixtureSampler.create_uniform_to_gmm(
            dimension=2,
            n_components=2,
        )
        sobol_mixture = MixtureSampler.create_sobol_to_gmm(
            dimension=2,
            n_components=2,
        )

        # Test basic functionality of both samplers
        assert uniform_mixture.dimension == 2
        assert sobol_mixture.dimension == 2

        sample_uniform = uniform_mixture.sample()
        sample_sobol = sobol_mixture.sample()

        assert sample_uniform.shape == (2,)
        assert sample_sobol.shape == (2,)

    def test_adjust_sample_count(self, mixture_sampler: MixtureSampler) -> None:
        """Test sample count adjustment affects exploration/exploitation correctly."""
        # Generate enough samples to enable exploitation
        for _ in range(mixture_sampler.min_explore_samples):
            mixture_sampler.sample()

        # Fit with sufficient elite samples
        elite_samples = np.random.random((mixture_sampler.min_fit_samples, 2))
        mixture_sampler.fit(elite_samples)

        # Should be in exploitation phase
        assert mixture_sampler.is_using_exploitation()

        # Adjust count below min_explore_samples
        mixture_sampler.adjust_sample_count(mixture_sampler.min_explore_samples - 1)
        assert not mixture_sampler.is_using_exploitation()
        assert mixture_sampler.sample_count == mixture_sampler.min_explore_samples - 1

    def test_adjust_sample_count_invalid(self, mixture_sampler: MixtureSampler) -> None:
        """Test sample count adjustment with invalid values."""
        with pytest.raises(ValueError, match="Sample count cannot be negative"):
            mixture_sampler.adjust_sample_count(-1)


if __name__ == "__main__":
    pytest.main([__file__])
