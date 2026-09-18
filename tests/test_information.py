import numpy as np
import pytest

from active_inference_neural_metacontrol import (
    categorical_fisher_map,
    detection_probability_map,
    expected_information_gain,
)


def binary_likelihood(size=20):
    detection = np.tile(np.linspace(0.05, 0.95, size), (size, 1))
    return np.stack((1.0 - detection, detection))


def test_visibility_is_detection_probability():
    likelihood = binary_likelihood()

    assert np.allclose(detection_probability_map(likelihood), likelihood[1])


def test_fisher_map_is_canonical_and_normalized():
    fisher = categorical_fisher_map(binary_likelihood())

    assert fisher.shape == (20, 20)
    assert fisher.min() >= 0
    assert fisher.max() == pytest.approx(1.0)


def test_informative_sensor_has_positive_expected_information_gain():
    posterior = np.ones((20, 20)) / 400

    assert expected_information_gain(posterior, binary_likelihood()) > 0
