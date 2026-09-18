import numpy as np
import pytest

from active_inference_neural_metacontrol import (
    canonicalize_posterior,
    information_loss,
    normalized_entropy,
    remap_posterior,
)


@pytest.mark.parametrize("resolution", (2, 5, 10, 20))
def test_canonical_belief_preserves_mass(resolution):
    posterior = np.arange(1, resolution**2 + 1, dtype=float)
    canonical = canonicalize_posterior(posterior, resolution)

    assert canonical.shape == (20, 20)
    assert canonical.sum() == pytest.approx(1.0)


def test_round_trip_preserves_a_coarse_belief():
    posterior = np.asarray([0.1, 0.2, 0.3, 0.4])

    fine = remap_posterior(posterior, 2, 20)
    restored = remap_posterior(fine, 20, 2)

    assert np.allclose(restored.ravel(), posterior)


def test_native_entropy_is_comparable_between_uniform_resolutions():
    assert normalized_entropy(np.ones(4), 2) == pytest.approx(1.0)
    assert normalized_entropy(np.ones(400), 20) == pytest.approx(1.0)


def test_information_loss_is_zero_when_resolution_is_retained_or_increased():
    posterior = np.asarray([0.7, 0.1, 0.1, 0.1])

    assert information_loss(posterior, 2, 2) == pytest.approx(0.0)
    assert information_loss(posterior, 2, 20) == pytest.approx(0.0)


def test_information_loss_detects_fine_to_coarse_compression():
    posterior = np.zeros(400)
    posterior[0] = 1.0

    assert information_loss(posterior, 20, 2) > 0.5
