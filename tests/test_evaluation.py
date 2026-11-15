
import numpy as np

from predictive_navigation.environment import RingWorld
from predictive_navigation.evaluation import (
    generate_trajectory,
    evaluate_trajectory,
)


def test_same_seed_same_trajectory():
    world = RingWorld()

    first = generate_trajectory(
        world, np.random.default_rng(42), moves=10
    )
    second = generate_trajectory(
        world, np.random.default_rng(42), moves=10
    )

    assert first == second


def test_filter_matches_manual_example():
    world = RingWorld()

    trajectory = [
        {
            "time": 0,
            "action": None,
            "true_state": 0,
            "observation": "A",
        },
        {
            "time": 1,
            "action": 1,
            "true_state": 1,
            "observation": "B",
        },
    ]

    result = evaluate_trajectory(world, trajectory)

    posterior = [
        result[1][f"belief_{i}"]
        for i in range(4)
    ]

    np.testing.assert_allclose(
        posterior,
        np.array([1, 64, 1, 8]) / 74,
    )


def test_posteriors_are_valid():
    world = RingWorld()

    trajectory = generate_trajectory(
        world, np.random.default_rng(3), moves=10
    )

    for row in evaluate_trajectory(world, trajectory):
        posterior = np.array([
            row[f"belief_{i}"] for i in range(4)
        ])

        assert np.all(posterior >= 0)
        assert np.isclose(posterior.sum(), 1)
        assert 0 <= row["prob_true"] <= 1


def test_true_state_does_not_affect_belief():
    world = RingWorld()

    first = [{
        "time": 0,
        "action": None,
        "true_state": 0,
        "observation": "A",
    }]

    second = [{
        **first[0],
        "true_state": 2,
    }]

    result_a = evaluate_trajectory(world, first)[0]
    result_b = evaluate_trajectory(world, second)[0]

    posterior_a = [
        result_a[f"belief_{i}"] for i in range(4)
    ]
    posterior_b = [
        result_b[f"belief_{i}"] for i in range(4)
    ]

    np.testing.assert_allclose(posterior_a, posterior_b)
