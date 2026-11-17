
import torch

from predictive_navigation.models import NavigationRNN
from predictive_navigation.ablations import predict_without_history


def test_first_step_is_unchanged():
    model = NavigationRNN(head_type="mlp")
    model.eval()

    observations = torch.tensor([[0, 1, 2]])
    actions = torch.tensor([[2, 2, 1]])

    with torch.no_grad():
        normal, _ = model(observations, actions)
        reset = predict_without_history(
            model, observations, actions
        )

    torch.testing.assert_close(normal[:, 0], reset[:, 0])


def test_reset_ignores_earlier_observation_history():
    model = NavigationRNN(head_type="mlp")
    model.eval()

    observations = torch.tensor([
        [0, 1, 2],
        [1, 0, 2],
    ])

    actions = torch.tensor([
        [1, 2, 0],
        [0, 2, 0],
    ])

    # At t=2, both sequences have the same current
    # observation, previous action, and upcoming action.
    with torch.no_grad():
        logits = predict_without_history(
            model, observations, actions
        )

    torch.testing.assert_close(
        logits[0, 2], logits[1, 2]
    )
