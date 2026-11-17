
import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from predictive_navigation.data import learning_targets
from predictive_navigation.models import (
    NavigationRNN,
    ObservationActionMLP,
)
from predictive_navigation.training import run_epoch


SEED = 7
BATCH_SIZE = 64
HIDDEN_SIZE = 16
LEARNING_RATE = 0.003


def load_dataset(split, objective):
    path = Path("outputs/datasets") / f"{split}.npz"

    with np.load(path, allow_pickle=False) as data:
        observations = torch.from_numpy(
            data["observations"].copy()
        )
        actions = torch.from_numpy(
            data["actions"].copy()
        )

    inputs = observations[:, :-1]
    targets = learning_targets(observations, objective)

    return TensorDataset(inputs, actions, targets)


def main():
    # 1. Parse command-line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=30)

    parser.add_argument(
        "--head",
        choices=["linear", "mlp"],
        default="linear",
    )

    parser.add_argument(
        "--model",
        choices=["gru", "feedforward"],
        default="gru",
    )

    parser.add_argument(
        "--objective",
        choices=["predictive", "reconstruction"],
        default="predictive",
    )

    args = parser.parse_args()

    if args.epochs <= 0:
        parser.error("--epochs must be positive")

    if args.objective == "reconstruction":
        if args.model != "gru" or args.head != "mlp":
            parser.error(
                "Use --model gru --head mlp for reconstruction"
            )

    # 2. Reproducibility
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    # 3. Load datasets
    train_data = load_dataset("train", args.objective)
    val_data = load_dataset("val", args.objective)

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )

    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # 4. Initialize model
    if args.model == "gru":
        model = NavigationRNN(
            hidden_size=HIDDEN_SIZE,
            head_type=args.head,
        )
    else:
        model = ObservationActionMLP(hidden_size=40)

    num_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Model: {args.model}")
    print(f"Head: {args.head if args.model == 'gru' else 'N/A'}")
    print(f"Objective: {args.objective}")
    print(f"Trainable parameters: {num_parameters}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # 5. Baselines
    if args.objective == "predictive":
        train_targets = train_data.tensors[2].reshape(-1)
        val_targets = val_data.tensors[2].reshape(-1)

        frequencies = torch.bincount(
            train_targets,
            minlength=3,
        ).float()

        frequencies /= frequencies.sum()

        baseline_loss = -torch.log(
            frequencies[val_targets]
        ).mean().item()

        baseline_accuracy = (
            (val_targets == frequencies.argmax())
            .float()
            .mean()
            .item()
        )

        baseline_info = {
            "baseline_name": "observation_frequency",
            "baseline_loss": baseline_loss,
            "baseline_accuracy": baseline_accuracy,
        }

        print(f"Frequency baseline loss: {baseline_loss:.4f}")
        print(f"Majority-class accuracy: {baseline_accuracy:.2%}")

    else:
        # The target is already present in the input.
        # An ideal direct-copy baseline is perfect.
        baseline_info = {
            "baseline_name": "copy_current_observation",
            "baseline_loss": 0.0,
            "baseline_accuracy": 1.0,
        }

        print("Reconstruction baseline: copy current observation")
        print("Ideal copy accuracy: 100%")

    # 6. Training
    best_val_loss = float("inf")
    best_val_accuracy = None
    best_epoch = None
    best_weights = None

    history = []

    print("\nEpoch | Train loss | Val loss | Val accuracy")
    print("-" * 47)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            optimizer,
        )

        val_loss, val_accuracy = run_epoch(
            model,
            val_loader,
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        })

        # Keep the checkpoint with lowest validation loss.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            best_weights = copy.deepcopy(
                model.state_dict()
            )

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"{epoch:>5} | "
                f"{train_loss:>10.4f} | "
                f"{val_loss:>8.4f} | "
                f"{val_accuracy:>11.2%}"
            )

    # 7. Choose an output directory
    if args.objective == "reconstruction":
        output_dir = Path(
            f"outputs/reconstruction_{args.head}_"
            f"seed{SEED}_epochs{args.epochs}"
        )

    elif args.model == "gru":
        output_dir = Path(
            f"outputs/predictive_{args.head}_"
            f"seed{SEED}_epochs{args.epochs}"
        )

    else:
        output_dir = Path(
            f"outputs/feedforward_seed{SEED}_"
            f"epochs{args.epochs}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # 8. Save best model
    checkpoint = {
        "model_state_dict": best_weights,
        "hidden_size": HIDDEN_SIZE,
        "best_epoch": best_epoch,
        "seed": SEED,
        "model_type": args.model,
        "head_type": (
            args.head if args.model == "gru" else None
        ),
        "objective": args.objective,
        "num_parameters": num_parameters,
    }

    torch.save(
        checkpoint,
        output_dir / "best_model.pt",
    )

    # 9. Save training history
    with (output_dir / "history.csv").open(
        "w", newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=history[0].keys(),
        )
        writer.writeheader()
        writer.writerows(history)

    # 10. Save experiment summary
    summary = {
        "seed": SEED,
        "epochs": args.epochs,
        "batch_size": BATCH_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "learning_rate": LEARNING_RATE,
        "device": "cpu",
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "model_type": args.model,
        "head_type": (
            args.head if args.model == "gru" else None
        ),
        "objective": args.objective,
        "num_parameters": num_parameters,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_accuracy,
        **baseline_info,
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print(f"\nBest epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation accuracy: {best_val_accuracy:.2%}")
    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()
