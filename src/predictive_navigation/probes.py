
import torch


def extract_hidden_states(model, observations, actions):
    """
    observations: (episodes, moves + 1)
    actions:      (episodes, moves)

    Returns:
        Flattened hidden states: (episodes * moves, hidden_size)
    """
    if observations.ndim != 2 or actions.ndim != 2:
        raise ValueError("Expected 2D observations and actions")

    if observations.shape != (
        actions.shape[0], actions.shape[1] + 1
    ):
        raise ValueError("Observation/action shapes are misaligned")

    model.eval()

    with torch.no_grad():
        _, hidden = model(observations[:, :-1], actions)

    return hidden.reshape(-1, hidden.shape[-1]).detach()


def location_targets(states, actions):
    """
    Match h_t with the true location s_t, not s_(t+1).
    """
    if states.ndim != 2 or states.shape != (
        actions.shape[0], actions.shape[1] + 1
    ):
        raise ValueError("State/action shapes are misaligned")

    return states[:, :-1].reshape(-1)
