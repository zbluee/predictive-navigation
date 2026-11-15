
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from predictive_navigation.environment import RingWorld
from predictive_navigation.bayes_filter import (
    predict_belief,
    update_belief,
)


def main():
    rng = np.random.default_rng(42)
    world = RingWorld()

    actions = [1, 1, 1, 0, -1, 1, 1]

    # The simulator knows the true position.
    true_state = 0

    # The Bayesian observer starts with no location preference.
    belief = np.full(world.num_states, 1 / world.num_states)

    history = []

    for t in range(len(actions) + 1):
        if t == 0:
            action = None  # No movement before the first observation.
        else:
            action = actions[t - 1]

            # Simulate actual movement.
            true_state = world.move(true_state, action)

            # Predict the belief after movement.
            belief = predict_belief(belief, action)

        # The simulator generates an observation.
        observation = world.observe(true_state, rng)

        # The filter only uses its belief and the observation.
        belief = update_belief(belief, observation, world)

        # Copy the belief so later updates don't change saved results.
        history.append(
            (t, action, true_state, observation, belief.copy())
        )

    print("t  action  true  observation  posterior (states 0,1,2,3)")

    for t, action, state, obs, posterior in history:
        movement = "-" if action is None else f"{action:+d}"
        probs = np.array2string(
            posterior, precision=3, suppress_small=True
        )

        print(f"{t:>1}  {movement:>6}  {state:>4}  {obs:>11}  {probs}")

    # Plot the posterior over time.
    beliefs = np.stack([row[4] for row in history])
    true_states = [row[2] for row in history]
    times = np.arange(len(history))

    fig, ax = plt.subplots(figsize=(9, 4))

    im = ax.imshow(
        beliefs.T,
        origin="lower",
        aspect="auto",
        vmin=0,
        vmax=1,
        cmap="Blues",
    )

    ax.scatter(
        times,
        true_states,
        marker="x",
        color="red",
        label="True position",
    )

    ax.set(
        xticks=times,
        yticks=np.arange(world.num_states),
        xlabel="Time step",
        ylabel="Position",
        title="Bayesian belief over positions",
    )

    ax.legend(loc="upper right")
    fig.colorbar(im, ax=ax, label="Posterior probability")
    fig.tight_layout()

    output = Path("outputs/bayes_trajectory_seed42.png")
    output.parent.mkdir(exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)

    print(f"\nSaved plot to {output}")


if __name__ == "__main__":
    main()
