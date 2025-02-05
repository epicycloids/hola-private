import numpy as np
import pytest
from numpy.testing import assert_allclose

from hola.core.samplers import (
    ClippedGaussianMixtureSampler,
    ExploreExploitSampler,
    SobolSampler,
    UniformSampler,
)


@pytest.fixture(scope="function", autouse=True)
def fixed_random_seed():
    """
    Fixture to set a fixed random seed before each test.
    """
    np.random.seed(42)


class TestUniformSampler:
    """
    Tests for UniformSampler functionality.
    """

    def test_init_valid(self):
        sampler = UniformSampler(dimension=2)
        assert sampler.dimension == 2

    def test_init_invalid(self):
        with pytest.raises(ValueError, match="dimension must be positive"):
            UniformSampler(dimension=0)

    def test_sample(self):
        sampler = UniformSampler(dimension=3)
        samples = sampler.sample(n_samples=5)
        assert samples.shape == (5, 3)
        assert np.all((samples >= 0) & (samples <= 1))

        with pytest.raises(ValueError, match="must be positive"):
            sampler.sample(n_samples=0)

    def test_fit_reset_no_ops(self):
        sampler = UniformSampler(dimension=2)
        sampler.fit(np.random.rand(5, 2))  # Should be no-op
        sampler.reset()  # Also no-op


class TestSobolSampler:
    """
    Tests for SobolSampler functionality.
    """

    def test_init_valid(self):
        sampler = SobolSampler(dimension=2)
        assert sampler.dimension == 2

    def test_init_invalid(self):
        with pytest.raises(ValueError, match="dimension must be positive"):
            SobolSampler(dimension=0)

    def test_sample(self):
        sampler = SobolSampler(dimension=2)
        samples = sampler.sample(n_samples=4)
        assert samples.shape == (4, 2)
        assert np.all((samples >= 0) & (samples <= 1))

        with pytest.raises(ValueError, match="must be positive"):
            sampler.sample(0)

    def test_reset(self):
        sampler = SobolSampler(dimension=2)
        s1 = sampler.sample(4)
        s2 = sampler.sample(4)
        sampler.reset()
        s3 = sampler.sample(4)
        # After resetting, the sequence should match the first batch
        assert_allclose(s1, s3, atol=1e-7)


class TestClippedGaussianMixtureSampler:
    """
    Tests for ClippedGaussianMixtureSampler functionality.
    """

    def test_init_valid(self):
        sampler = ClippedGaussianMixtureSampler(dimension=2, n_components=3)
        assert sampler.dimension == 2
        assert sampler.n_components == 3

    @pytest.mark.parametrize("dim", [0, -1])
    def test_init_invalid_dimension(self, dim):
        with pytest.raises(ValueError, match="dimension must be positive"):
            ClippedGaussianMixtureSampler(dimension=dim, n_components=2)

    @pytest.mark.parametrize("n_components", [0, -1])
    def test_init_invalid_n_components(self, n_components):
        with pytest.raises(ValueError, match="n_components must be positive"):
            ClippedGaussianMixtureSampler(dimension=2, n_components=n_components)

    def test_fit_and_sample(self):
        sampler = ClippedGaussianMixtureSampler(dimension=2, n_components=2)
        with pytest.raises(ValueError, match="Call fit(...)"):
            sampler.sample(3)

        training_data = np.array(
            [
                [0.2, 0.3],
                [0.4, 0.4],
                [0.5, 0.6],
                [0.1, 0.7],
            ]
        )
        sampler.fit(training_data)
        samples = sampler.sample(5)
        assert samples.shape == (5, 2)
        assert np.all(samples >= 0) and np.all(samples <= 1)

    def test_fit_invalid_data(self):
        sampler = ClippedGaussianMixtureSampler(dimension=2, n_components=2)
        bad_data = np.array(
            [
                [0.1, 1.1],  # out-of-bounds
                [0.2, 0.2],
            ]
        )
        with pytest.raises(ValueError, match="Samples must lie in"):
            sampler.fit(bad_data)

    def test_reset_clears_parameters(self):
        sampler = ClippedGaussianMixtureSampler(dimension=2, n_components=2)
        data = np.random.rand(3, 2)
        sampler.fit(data)
        sampler.reset()
        with pytest.raises(ValueError, match="Call fit(...)"):
            sampler.sample(1)


class TestExploreExploitSampler:
    """
    Tests for the ExploreExploitSampler hybrid strategy.
    """

    def test_init_valid(self):
        explore = UniformSampler(dimension=2)
        exploit = ClippedGaussianMixtureSampler(dimension=2, n_components=2)
        sampler = ExploreExploitSampler(
            explore_sampler=explore,
            exploit_sampler=exploit,
            min_explore_samples=5,
            min_fit_samples=3,
        )
        assert sampler.dimension == 2
        assert sampler.min_explore_samples == 5
        assert sampler.min_fit_samples == 3
        assert sampler.sample_count == 0

    def test_init_invalid(self):
        explore = UniformSampler(dimension=2)
        exploit = ClippedGaussianMixtureSampler(dimension=2, n_components=2)

        with pytest.raises(ValueError, match="must be positive"):
            ExploreExploitSampler(explore, exploit, min_explore_samples=0)

        with pytest.raises(ValueError, match="must be positive"):
            ExploreExploitSampler(explore, exploit, min_fit_samples=0)

        with pytest.raises(ValueError, match="cannot exceed"):
            ExploreExploitSampler(
                explore,
                exploit,
                min_explore_samples=5,
                min_fit_samples=6,
            )

        exploit_mismatch = ClippedGaussianMixtureSampler(dimension=3, n_components=2)
        with pytest.raises(ValueError, match="same dimension"):
            ExploreExploitSampler(explore, exploit_mismatch)

    def test_sampling_flow(self):
        explore = SobolSampler(dimension=2)
        exploit = ClippedGaussianMixtureSampler(dimension=2, n_components=2)
        sampler = ExploreExploitSampler(explore, exploit, min_explore_samples=5, min_fit_samples=2)

        # Start in exploration mode
        assert sampler.sample_count == 0
        s1 = sampler.sample(n_samples=3)
        assert s1.shape == (3, 2)
        assert sampler.sample_count == 3
        assert not sampler.is_using_exploitation()

        # Now 2 more => total 5 => meets min_explore_samples, but not fitted
        sampler.sample(n_samples=2)
        assert sampler.sample_count == 5
        assert not sampler.is_using_exploitation()

        # Fit exploit with 2 "elite" samples => meets min_fit_samples
        elite_data = np.random.rand(2, 2)
        sampler.fit(elite_data)
        assert sampler.is_using_exploitation()

        # Now samples come from exploit
        s2 = sampler.sample(2)
        assert s2.shape == (2, 2)
        assert sampler.sample_count == 7

    def test_reset(self):
        explore = UniformSampler(dimension=2)
        exploit = ClippedGaussianMixtureSampler(dimension=2, n_components=2)
        sampler = ExploreExploitSampler(explore, exploit, 5, 2)

        sampler.sample(3)
        assert sampler.sample_count == 3

        sampler.reset()
        assert sampler.sample_count == 0
        assert not sampler.is_using_exploitation()

        # Exploit sampler should need fitting again
        with pytest.raises(ValueError, match="Call fit(...)"):
            sampler._exploit_sampler.sample(1)
