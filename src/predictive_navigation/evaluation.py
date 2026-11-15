
import numpy as np

from .bayes_filter import predict_belief, update_belief


def generate_trajectory(world, rng, moves=20):
    #Generate a trajectory with noisy observations

    if moves < 0:
        raise ValueError("moves must be nonnegative")

    state = int(rng.integers(world.num_states))
    steps = []

    for t in range(moves + 1):
        action = None if t == 0 else int(rng.choice([-1, 0, 1]))

        if action is not None:
            state = world.move(state, action)

        steps.append({
            "time": t,
            "action": action,
            "true_state": state,
            "observation": world.observe(state, rng),
        })

    return steps


def evaluate_trajectory(world, trajectory):
    #Infer beliefs, then compare them with the true positions

    belief = np.full(world.num_states, 1 / world.num_states)
    records = []

    for step in trajectory:
        # Prediction: use the movement.
        if step["action"] is not None:
            belief = predict_belief(belief, step["action"])

        # Update: use the new observation.
        belief = update_belief(
            belief, step["observation"], world
        )

        estimate = int(np.argmax(belief))

        # Entropy measures how spread out the belief is.
        nonzero = belief[belief > 0]
        entropy = -float(np.sum(nonzero * np.log(nonzero)))

        records.append({
            **step,
            "estimate": estimate,
            "correct": int(estimate == step["true_state"]),
            "prob_true": float(belief[step["true_state"]]),
            "confidence": float(belief.max()),
            "entropy": entropy,
            **{
                f"belief_{i}": float(p)
                for i, p in enumerate(belief)
            },
        })

    return records
