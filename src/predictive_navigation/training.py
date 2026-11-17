
import torch
from torch.nn import functional as F


def run_epoch(model, loader, optimizer=None):
    #Train for one epoch, or evaluate without updating weights

    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_predictions = 0

    with torch.set_grad_enabled(training):
        for observations, actions, targets in loader:
            logits, _ = model(observations, actions)

            # Cross-entropy expects (batch, classes, steps).
            loss = F.cross_entropy(
                logits.transpose(1, 2),
                targets,
            )

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            n = targets.numel()

            total_loss += loss.item() * n
            total_correct += (
                (logits.argmax(dim=-1) == targets)
                .sum()
                .item()
            )
            total_predictions += n

    return (
        total_loss / total_predictions,
        total_correct / total_predictions,
    )
