import numpy as np
import pytest

from active_inference_neural_metacontrol.expected_state_labels import (
    expected_categorical_state_terms,
)


def test_uninformative_sensor_has_no_expected_complexity():
    prior = np.asarray([0.25, 0.75])
    likelihood = np.asarray([[0.8, 0.8], [0.2, 0.2]])

    accuracy, complexity = expected_categorical_state_terms(prior, likelihood)

    expected_accuracy = 0.8 * np.log(0.8) + 0.2 * np.log(0.2)
    assert accuracy == pytest.approx(expected_accuracy)
    assert complexity == pytest.approx(0.0, abs=1e-12)


def test_perfect_binary_sensor_has_log_two_expected_complexity():
    prior = np.asarray([0.5, 0.5])
    likelihood = np.eye(2)

    accuracy, complexity = expected_categorical_state_terms(prior, likelihood)

    assert accuracy == pytest.approx(0.0, abs=1e-12)
    assert complexity == pytest.approx(np.log(2.0))
