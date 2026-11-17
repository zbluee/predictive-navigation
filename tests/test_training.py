
import torch
from torch.utils.data import DataLoader, TensorDataset

from predictive_navigation.models import NavigationRNN
from predictive_navigation.training import run_epoch


def make_loader():
    observations = torch.tensor([
        [0, 1, 0],
        [2, 0, 1],
    ])
    actions = torch.tensor([
        [2, 2, 1],
        [0, 1, 2],
    ])
    targets = torch.tensor([
        [1, 0, 2],
        [0, 1, 2],
    ])

    return DataLoader(
        TensorDataset(observations, actions, targets),
        batch_size=2,
    )


def test_training_updates_weights():
    torch.manual_seed(0)

    model = NavigationRNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    before = model.head.weight.detach().clone()

    loss, accuracy = run_epoch(
        model, make_loader(), optimizer
    )

    assert torch.isfinite(torch.tensor(loss))
    assert 0 <= accuracy <= 1
    assert not torch.equal(before, model.head.weight)


def test_validation_does_not_update_weights():
    model = NavigationRNN()

    before = model.head.weight.detach().clone()

    loss, accuracy = run_epoch(
        model, make_loader()
    )

    assert torch.isfinite(torch.tensor(loss))
    assert 0 <= accuracy <= 1
    torch.testing.assert_close(before, model.head.weight)
