
import json
from pathlib import Path

import numpy as np

from predictive_navigation.environment import RingWorld
from predictive_navigation.evaluation import (
    generate_trajectory,
    evaluate_trajectory,
)


def evaluate_seed(seed, n_trajectories=200, moves=20):
    rng = np.random.default_rng(seed)
    world = RingWorld()

    records = []

    for _ in range(n_trajectories):
        trajectory = generate_trajectory(world, rng, moves)
        records.extend(evaluate_trajectory(world, trajectory))

    correct = np.array([r["correct"] for r in records])
    confidence = np.array([r["confidence"] for r in records])
    prob_true = np.array([r["prob_true"] for r in records])
    entropy = np.array([r["entropy"] for r in records])

    high_conf = confidence >= 0.9

    return {
        "seed": seed,
        "accuracy": float(correct.mean()),
        "mean_entropy": float(entropy.mean()),
        "mean_log_loss": float(-np.log(prob_true).mean()),
        "high_conf_predictions": int(high_conf.sum()),
        "high_conf_errors": int(
            ((correct == 0) & high_conf).sum()
        ),
    }


def main():
    seeds = [0, 1, 2, 3, 42]
    results = [evaluate_seed(seed) for seed in seeds]

    print("Seed | Accuracy | Entropy | Log loss | HC errors")
    print("-" * 55)

    for r in results:
        print(
            f"{r['seed']:>4} | "
            f"{r['accuracy']:>7.2%} | "
            f"{r['mean_entropy']:>7.3f} | "
            f"{r['mean_log_loss']:>8.3f} | "
            f"{r['high_conf_errors']:>9}"
        )

    accuracies = np.array([r["accuracy"] for r in results])

    summary = {
        "seeds": seeds,
        "trajectories_per_seed": 200,
        "moves_per_trajectory": 20,
        "mean_accuracy": float(accuracies.mean()),
        "std_accuracy": float(accuracies.std(ddof=1)),
        "min_accuracy": float(accuracies.min()),
        "max_accuracy": float(accuracies.max()),
        "results": results,
    }

    print(f"\nMean accuracy: {summary['mean_accuracy']:.2%}")
    print(f"Std across seeds: {summary['std_accuracy']:.2%}")
    print(
        f"Range: {summary['min_accuracy']:.2%} "
        f"to {summary['max_accuracy']:.2%}"
    )

    output = Path("outputs/bayes_multi_seed_summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
