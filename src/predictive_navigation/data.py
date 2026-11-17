
import numpy as np

from .evaluation import generate_trajectory


ACTION_VALUES = (-1, 0, 1)


def make_dataset(world, rng, n_trajectories, moves=20):
    #Generate fixed-length navigation sequences

    if n_trajectories <= 0 or moves <= 0:
        raise ValueError("Trajectories and moves must be positive")

    observations = np.empty(
        (n_trajectories, moves + 1), dtype=np.int64
    )
    actions = np.empty(
        (n_trajectories, moves), dtype=np.int64
    )
    states = np.empty(
        (n_trajectories, moves + 1), dtype=np.int64
    )

    observation_ids = {
        label: i for i, label in enumerate(world.OBSERVATIONS)
    }
    action_ids = {
        action: i for i, action in enumerate(ACTION_VALUES)
    }

    for episode in range(n_trajectories):
        trajectory = generate_trajectory(world, rng, moves)

        observations[episode] = [
            observation_ids[step["observation"]]
            for step in trajectory
        ]

        states[episode] = [
            step["true_state"]
            for step in trajectory
        ]

        # Action stored at time t+1 moves state t to state t+1.
        actions[episode] = [
            action_ids[step["action"]]
            for step in trajectory[1:]
        ]

    return {
        "observations": observations,
        "actions": actions,
        "states": states,
    }

