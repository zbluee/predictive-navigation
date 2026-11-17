
from pathlib import Path

import numpy as np


def load_data(split):
    path = Path("outputs/datasets") / f"{split}.npz"

    with np.load(path, allow_pickle=False) as data:
        observations = data["observations"]
        actions = data["actions"]

    return (
        observations[:, :-1],  # Current observation
        actions,               # Upcoming action
        observations[:, 1:],   # Next observation
    )


def main():
    train_obs, train_actions, train_targets = load_data("train")
    val_obs, val_actions, val_targets = load_data("val")

    # Three current observations, three actions,
    # and three possible next observations.
    # Start with 1 in every cell (Laplace smoothing).
    counts = np.ones((3, 3, 3), dtype=np.float64)

    # Count next observations for each (observation, action) pair.
    np.add.at(
        counts,
        (
            train_obs.ravel(),
            train_actions.ravel(),
            train_targets.ravel(),
        ),
        1,
    )

    # Convert counts into conditional probabilities.
    probabilities = counts / counts.sum(axis=-1, keepdims=True)

    # Look up the predictions for the validation dataset.
    predictions = probabilities[val_obs, val_actions]

    # Probability assigned to each actual next observation.
    prob_true = np.take_along_axis(
        predictions,
        val_targets[..., None],
        axis=-1,
    ).squeeze(-1)

    accuracy = np.mean(
        predictions.argmax(axis=-1) == val_targets
    )
    log_loss = -np.log(prob_true).mean()

    print(f"Conditional baseline loss: {log_loss:.4f}")
    print(f"Conditional baseline accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()
