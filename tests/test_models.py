
import torch
from torch.nn import functional as F

from predictive_navigation.models import NavigationRNN
from predictive_navigation.models import ObservationActionMLP


def test_output_shapes():
    model = NavigationRNN(hidden_size=16)

    observations = torch.tensor([
        [0, 1, 0],
        [2, 0, 2],
    ])
    actions = torch.tensor([
        [2, 2, 1],
        [0, 1, 2],
    ])

    logits, hidden = model(observations, actions)

    assert logits.shape == (2, 3, 3)
    assert hidden.shape == (2, 3, 16)


def test_no_future_observation_leakage():
    model = NavigationRNN()
    model.eval()

    # Both sequences begin with A and action +1.
    # Later observations are different.
    observations = torch.tensor([
        [0, 1, 2],
        [0, 2, 1],
    ])
    actions = torch.tensor([
        [2, 2, 1],
        [2, 1, 0],
    ])

    with torch.no_grad():
        logits, hidden = model(observations, actions)

    torch.testing.assert_close(logits[0, 0], logits[1, 0])
    torch.testing.assert_close(hidden[0, 0], hidden[1, 0])


def test_loss_and_gradients():
    model = NavigationRNN()

    observations = torch.tensor([[0, 1, 0]])
    actions = torch.tensor([[2, 2, 1]])

    # Example next-observation targets.
    targets = torch.tensor([[1, 0, 2]])

    logits, _ = model(observations, actions)

    loss = F.cross_entropy(
        logits.reshape(-1, 3),
        targets.reshape(-1),
    )

    loss.backward()

    assert torch.isfinite(loss)
    assert model.gru.weight_ih_l0.grad is not None
    assert model.head.weight.grad is not None

def test_mlp_head():
    model = NavigationRNN(
        hidden_size=16,
        head_type="mlp",
    )

    observations = torch.tensor([[0, 1, 2]])
    actions = torch.tensor([[2, 1, 0]])
    targets = torch.tensor([[1, 2, 0]])

    logits, hidden = model(observations, actions)

    assert logits.shape == (1, 3, 3)
    assert hidden.shape == (1, 3, 16)

    loss = F.cross_entropy(
        logits.transpose(1, 2), targets
    )
    loss.backward()

    assert model.head_features[0].weight.grad is not None
    assert model.gru.weight_ih_l0.grad is not None


def test_feedforward_shapes():
    model = ObservationActionMLP()

    observations = torch.tensor([
        [0, 1, 2],
        [2, 0, 1],
    ])
    actions = torch.tensor([
        [2, 1, 0],
        [0, 2, 1],
    ])

    logits, hidden = model(observations, actions)

    assert logits.shape == (2, 3, 3)
    assert hidden is None


def test_feedforward_does_not_use_history():
    model = ObservationActionMLP()
    model.eval()

    # Different histories, same current observation and action at t=2
    observations = torch.tensor([
        [0, 1, 2],
        [1, 0, 2],
    ])
    actions = torch.tensor([
        [2, 1, 0],
        [0, 2, 0],
    ])

    with torch.no_grad():
        logits, _ = model(observations, actions)

    # Predictions at t=2 must be identical.
    torch.testing.assert_close(
        logits[0, 2],
        logits[1, 2],
    )


def test_feedforward_gradients():
    model = ObservationActionMLP()

    observations = torch.tensor([[0, 1, 2]])
    actions = torch.tensor([[2, 1, 0]])
    targets = torch.tensor([[1, 2, 0]])

    logits, _ = model(observations, actions)

    loss = F.cross_entropy(
        logits.transpose(1, 2),
        targets,
    )
    loss.backward()

    assert model.network[0].weight.grad is not None
    assert model.network[-1].weight.grad is not None
