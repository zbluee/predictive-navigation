import numpy as np
import pytest

from predictive_navigation.bayes_filter import (
    predict_belief,
    update_belief,
)
from predictive_navigation.environment import RingWorld


def test_forward_movement():
    belief = np.array([0.5, 0.0, 0.5, 0.0])

    result = predict_belief(belief, 1)

    np.testing.assert_allclose(
        result, [0.0, 0.5, 0.0, 0.5]
    )


def test_backward_movement():
    belief = np.array([0.7, 0.2, 0.1, 0.0])

    result = predict_belief(belief, -1)

    np.testing.assert_allclose(
        result, [0.2, 0.1, 0.0, 0.7]
    )


def test_stay():
    belief = np.array([0.1, 0.2, 0.3, 0.4])

    result = predict_belief(belief, 0)

    np.testing.assert_allclose(result, belief)


def test_probabilities_still_sum_to_one():
    belief = np.array([0.1, 0.2, 0.3, 0.4])

    result = predict_belief(belief, 1)

    assert np.isclose(result.sum(), 1.0)


def test_invalid_action():
    belief = np.array([0.5, 0.0, 0.5, 0.0])

    with pytest.raises(ValueError):
        predict_belief(belief, 3)


def test_update_with_c():
    world = RingWorld()
    belief = np.array([0.0, 0.5, 0.0, 0.5])

    result = update_belief(belief, "C", world)

    expected = np.array([0.0, 1 / 9, 0.0, 8 / 9])
    np.testing.assert_allclose(result, expected)


def test_update_with_b():
    world = RingWorld()
    belief = np.array([0.0, 0.5, 0.0, 0.5])

    result = update_belief(belief, "B", world)

    expected = np.array([0.0, 8 / 9, 0.0, 1 / 9])
    np.testing.assert_allclose(result, expected)


def test_ambiguous_observation():
    world = RingWorld()
    belief = np.array([0.5, 0.0, 0.5, 0.0])

    result = update_belief(belief, "A", world)

    # A cannot distinguish positions 0 and 2.
    np.testing.assert_allclose(result, belief)


def test_invalid_observation():
    world = RingWorld()
    belief = np.array([0.25, 0.25, 0.25, 0.25])

    with pytest.raises(ValueError):
        update_belief(belief, "D", world)

def test_ambiguous_observation_preserves_prior():
    world = RingWorld()
    belief = np.array([0.9, 0.0, 0.1, 0.0])

    result = update_belief(belief, "A", world)

    np.testing.assert_allclose(result, belief)