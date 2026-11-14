
import numpy as np
import pytest

from predictive_navigation.environment import RingWorld


def test_movement_wraps_around():
    world = RingWorld()

    assert world.move(3, 1) == 0
    assert world.move(0, -1) == 3
    assert world.move(2, 0) == 2


def test_ambiguous_positions():
    world = RingWorld()

    np.testing.assert_array_equal(
        world.observation_probs(0),
        world.observation_probs(2),
    )


def test_observation_probabilities():
    world = RingWorld()

    for state in range(4):
        probs = world.observation_probs(state)

        assert np.all(probs >= 0)
        assert np.isclose(probs.sum(), 1.0)

    assert np.isclose(
        world.observation_probs(1)[1], 0.8
    )


def test_same_seed_same_observations():
    world = RingWorld()

    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)

    states = [0, 1, 2, 3, 0]

    observations_a = [
        world.observe(s, rng_a) for s in states
    ]
    observations_b = [
        world.observe(s, rng_b) for s in states
    ]

    assert observations_a == observations_b


def test_invalid_action():
    world = RingWorld()

    with pytest.raises(ValueError):
        world.move(0, 5)
