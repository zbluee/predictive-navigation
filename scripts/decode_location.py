
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from predictive_navigation.models import NavigationRNN
from predictive_navigation.probes import (
    extract_hidden_states,
    location_targets,
)


PROBE_SEED = 23
EPOCHS = 200
LEARNING_RATE = 0.05

CHECKPOINTS = {
    "predictive": Path(
        "outputs/predictive_mlp_seed7_epochs120/best_model.pt"
    ),
    "reconstruction": Path(
        "outputs/reconstruction_mlp_seed7_epochs120/best_model.pt"
    ),
}

OUTPUT_DIR = Path("outputs/location_probes_seed23")


def load_split(split):
    with np.load(
        f"outputs/datasets/{split}.npz",
        allow_pickle=False,
    ) as data:
        observations = torch.from_numpy(
            data["observations"].copy()
        )
        actions = torch.from_numpy(
            data["actions"].copy()
        )
        states = torch.from_numpy(
            data["states"].copy()
        )

    return observations, actions, states


def load_frozen_model(path):
    # Load only checkpoints produced by our own training script.
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    model = NavigationRNN(
        hidden_size=checkpoint["hidden_size"],
        head_type="mlp",
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return model


def fit_linear_probe(train_features, train_labels,
                     val_features, val_labels):
    # Same decoder initialization for both representations.
    torch.manual_seed(PROBE_SEED)

    probe = nn.Linear(train_features.shape[1], 4)
    optimizer = torch.optim.Adam(
        probe.parameters(), lr=LEARNING_RATE
    )

    best_loss = float("inf")
    best_accuracy = None
    best_epoch = None
    best_weights = None

    for epoch in range(1, EPOCHS + 1):
        probe.train()

        logits = probe(train_features)
        loss = F.cross_entropy(logits, train_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        probe.eval()

        with torch.no_grad():
            val_logits = probe(val_features)
            val_loss = F.cross_entropy(
                val_logits, val_labels
            ).item()

            val_accuracy = (
                (val_logits.argmax(dim=-1) == val_labels)
                .float().mean().item()
            )

        if val_loss < best_loss:
            best_loss = val_loss
            best_accuracy = val_accuracy
            best_epoch = epoch
            best_weights = copy.deepcopy(probe.state_dict())

    return {
        "best_epoch": best_epoch,
        "val_loss": best_loss,
        "val_accuracy": best_accuracy,
        "weights": best_weights,
    }


def observation_only_baseline(train_data, val_data):
    """
    Predict location using only the current observation.
    No movement or history.
    """
    train_obs, _, train_states = train_data
    val_obs, _, val_states = val_data

    train_x = train_obs[:, :-1].numpy().ravel()
    train_y = train_states[:, :-1].numpy().ravel()

    val_x = val_obs[:, :-1].numpy().ravel()
    val_y = val_states[:, :-1].numpy().ravel()

    counts = np.ones((3, 4), dtype=np.float64)

    np.add.at(counts, (train_x, train_y), 1)

    probabilities = counts / counts.sum(
        axis=1, keepdims=True
    )

    predicted_probabilities = probabilities[val_x]
    predictions = predicted_probabilities.argmax(axis=-1)

    accuracy = np.mean(predictions == val_y)

    correct_probabilities = predicted_probabilities[
        np.arange(len(val_y)), val_y
    ]
    log_loss = -np.log(correct_probabilities).mean()

    return {
        "val_loss": float(log_loss),
        "val_accuracy": float(accuracy),
    }


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    train_data = load_split("train")
    val_data = load_split("val")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    baseline = observation_only_baseline(
        train_data, val_data
    )

    print("Observation-only location baseline:")
    print(
        f"  Loss: {baseline['val_loss']:.4f}, "
        f"Accuracy: {baseline['val_accuracy']:.2%}"
    )

    results = {
        "probe_seed": PROBE_SEED,
        "probe_epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "observation_only_baseline": baseline,
        "models": {},
    }

    for name, checkpoint_path in CHECKPOINTS.items():
        print(f"\nEvaluating: {name}")

        model = load_frozen_model(checkpoint_path)

        train_obs, train_actions, train_states = train_data
        val_obs, val_actions, val_states = val_data

        train_features = extract_hidden_states(
            model, train_obs, train_actions
        )
        val_features = extract_hidden_states(
            model, val_obs, val_actions
        )

        train_labels = location_targets(
            train_states, train_actions
        )
        val_labels = location_targets(
            val_states, val_actions
        )

        print(f"  Train features: {tuple(train_features.shape)}")
        print(f"  Val features: {tuple(val_features.shape)}")

        result = fit_linear_probe(
            train_features,
            train_labels,
            val_features,
            val_labels,
        )

        torch.save(
            {
                "probe_state_dict": result["weights"],
                "input_dim": train_features.shape[1],
                "best_epoch": result["best_epoch"],
                "source_checkpoint": str(checkpoint_path),
            },
            OUTPUT_DIR / f"{name}_probe.pt",
        )

        results["models"][name] = {
            "checkpoint": str(checkpoint_path),
            "best_epoch": result["best_epoch"],
            "val_loss": result["val_loss"],
            "val_accuracy": result["val_accuracy"],
        }

        print(f"  Best epoch: {result['best_epoch']}")
        print(f"  Val loss: {result['val_loss']:.4f}")
        print(f"  Val accuracy: {result['val_accuracy']:.2%}")

    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )

    print(f"\nSaved results to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
