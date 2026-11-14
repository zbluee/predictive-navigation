import numpy as np


def predict_belief(belief: np.ndarray, action: int) -> np.ndarray:
    #Predict the new position belief after an exact movement
    if action not in (-1, 0, 1):
        raise ValueError("Action must be -1, 0, or 1")

    return np.roll(belief, action)


def update_belief(belief: np.ndarray, observation: str, world) -> np.ndarray:
    #Update the position belief using a new observation.

    if observation not in world.OBSERVATIONS:
        raise ValueError(f"Unknown observation: {observation}")

    if (
        belief.shape != (world.num_states,)
        or not np.all(np.isfinite(belief))
        or np.any(belief < 0)
        or not np.isclose(belief.sum(), 1.0)
    ):
        raise ValueError("Belief must be a valid probability distribution")

    # Find the column corresponding to the observation.
    obs_index = world.OBSERVATIONS.index(observation)

    # P(observation | state) for all possible positions.
    likelihoods = world.observation_model[:, obs_index]

    # Multiply the prior by the observation likelihoods.
    unnormalized = belief * likelihoods

    # Convert the weights into probabilities that sum to 1.
    evidence = unnormalized.sum()

    if evidence <= 0:
        raise ValueError("Observation has zero probability under the belief")

    return unnormalized / evidence
