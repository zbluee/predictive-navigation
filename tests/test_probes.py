
import pytest
import torch

from predictive_navigation.models import NavigationRNN
from predictive_navigation.probes import (
    extract_hidden_states,
    location_targets,
)


def test_hidden_state_and_target_alignment():
    model = NavigationRNN(head_type="mlp")

    observations = torch.tensor([
        [0, 1, 2, 0],
        [2, 0, 1, 2],
    ])

    actions = torch.tensor([
        [2, 1, 0],
        [0, 2, 1],
    ])

    states = torch.tensor([
        [0, 1, 1, 0],
        [3, 2, 3, 3],
    ])

    features = extract_hidden_states(
        model, observations, actions
    )
    targets = location_targets(states, actions)

    assert features.shape == (6, 16)

    # There are three hidden states per episode.
    # The final state in each episode is excluded.
    torch.testing.assert_close(
        targets,
        torch.tensor([0, 1, 1, 3, 2, 3]),
    )


def test_invalid_sequence_alignment():
    model = NavigationRNN(head_type="mlp")

    observations = torch.tensor([[0, 1, 2]])
    actions = torch.tensor([[2, 1, 0]])

    with pytest.raises(ValueError):
        extract_hidden_states(
            model, observations, actions
        )
