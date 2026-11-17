
import numpy as np
import pytest

from predictive_navigation.data import (
    ACTION_VALUES,
    make_dataset,
)
from predictive_navigation.environment import RingWorld


def test_dataset_shapes():
    world = RingWorld()

    data = make_dataset(
        world, np.random.default_rng(42), 3, moves=5
    )

    assert data["observations"].shape == (3, 6)
    assert data["actions"].shape == (3, 5)
    assert data["states"].shape == (3, 6)


def test_movement_alignment():
    world = RingWorld()

    data = make_dataset(
        world, np.random.default_rng(42), 5, moves=10
    )

    for states, actions in zip(
        data["states"], data["actions"]
    ):
        for t, action_id in enumerate(actions):
            action = ACTION_VALUES[action_id]

            assert states[t + 1] == world.move(
                states[t], action
            )


def test_prediction_target_alignment():
    world = RingWorld()

    data = make_dataset(
        world, np.random.default_rng(42), 3, moves=5
    )

    observations = data["observations"]

    inputs = observations[:, :-1]
    next_targets = observations[:, 1:]
    reconstruction_targets = observations[:, :-1]

    assert inputs.shape == (3, 5)
    assert next_targets.shape == (3, 5)

    np.testing.assert_array_equal(
        reconstruction_targets, inputs
    )
    np.testing.assert_array_equal(
        next_targets[:, 0], observations[:, 1]
    )


def test_reproducible_dataset():
    world = RingWorld()

    first = make_dataset(
        world, np.random.default_rng(42), 5
    )
    second = make_dataset(
        world, np.random.default_rng(42), 5
    )

    for key in first:
        np.testing.assert_array_equal(
            first[key], second[key]
        )


def test_invalid_dataset_size():
    world = RingWorld()

    with pytest.raises(ValueError):
        make_dataset(
            world, np.random.default_rng(42), 0
        )
