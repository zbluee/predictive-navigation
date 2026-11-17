
import torch
from torch.nn import functional as F


def predict_without_history(model, observations, actions):
    #Run the trained GRU with memory reset at every time step

    if observations.ndim != 2 or actions.shape != observations.shape:
        raise ValueError("Expected matching (batch, steps) arrays")

    # Match the input encoding used by NavigationRNN.
    previous_actions = torch.full_like(actions, 3)
    previous_actions[:, 1:] = actions[:, :-1]

    obs_input = F.one_hot(observations, num_classes=3)
    prev_action_input = F.one_hot(
        previous_actions, num_classes=4
    )

    rnn_input = torch.cat(
        [obs_input, prev_action_input], dim=-1
    ).float()

    hidden_states = []

    for t in range(observations.shape[1]):
        # No previous hidden state is supplied.
        # The GRU starts from zero at every step.
        hidden, _ = model.gru(rnn_input[:, t:t + 1])

        hidden_states.append(hidden)

    hidden_states = torch.cat(hidden_states, dim=1)

    upcoming_action = F.one_hot(
        actions, num_classes=3
    ).float()

    prediction_input = torch.cat(
        [hidden_states, upcoming_action], dim=-1
    )

    features = model.head_features(prediction_input)
    logits = model.head(features)

    return logits
