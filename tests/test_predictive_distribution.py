
import numpy as np
import pytest

from shared_code.evaluation.predictive_distribution import (
    EmpiricalPredictiveDistribution
)


@pytest.fixture
def distribution():
    return EmpiricalPredictiveDistribution(
        forecast=250.0,
        errors=np.array([-5.0, -2.0, 0.0, 3.0, 8.0])
    )


def test_shifted_samples(distribution):
    np.testing.assert_array_equal(
        distribution.samples,
        [245.0, 248.0, 250.0, 253.0, 258.0]
    )


def test_cdf(distribution):
    assert distribution.cdf(250.0) == pytest.approx(0.6)
    assert distribution.cdf(244.0) == 0.0
    assert distribution.cdf(260.0) == 1.0


def test_mean(distribution):
    assert distribution.mean() == pytest.approx(250.8)


def test_variance(distribution):
    assert distribution.variance() == pytest.approx(
        np.var([-5.0, -2.0, 0.0, 3.0, 8.0])
    )


def test_interval(distribution):
    assert distribution.interval(0.80) == (245.0, 258.0)


def test_invalid_errors():
    with pytest.raises(ValueError):
        EmpiricalPredictiveDistribution(
            forecast=250.0,
            errors=[1.0, np.nan]
        )


def test_sampling(distribution):
    samples = distribution.sample(
        n=100,
        random_state=42
    )

    assert len(samples) == 100
    assert np.isin(samples, distribution.samples).all()
