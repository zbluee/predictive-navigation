
import csv
import json
import platform
from pathlib import Path

import numpy as np

from predictive_navigation.environment import RingWorld
from predictive_navigation.evaluation import (
    generate_trajectory,
    evaluate_trajectory,
)


def main():
    seed = 42
    n_trajectories = 200
    moves_per_trajectory = 20

    rng = np.random.default_rng(seed)
    world = RingWorld()
    rows = []

    for episode in range(n_trajectories):
        trajectory = generate_trajectory(
            world, rng, moves_per_trajectory
        )
        records = evaluate_trajectory(world, trajectory)

        rows.extend(
            {"episode": episode, **record}
            for record in records
        )

    summary = {
        "seed": seed,
        "trajectories": n_trajectories,
        "moves_per_trajectory": moves_per_trajectory,
        "evaluated_observations": len(rows),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "observation_model": world.observation_model.tolist(),
        "accuracy": float(np.mean([
            r["correct"] for r in rows
        ])),
        "mean_probability_true_state": float(np.mean([
            r["prob_true"] for r in rows
        ])),
        "mean_confidence": float(np.mean([
            r["confidence"] for r in rows
        ])),
        "mean_entropy_nats": float(np.mean([
            r["entropy"] for r in rows
        ])),
        "mean_log_loss": float(np.mean([
            -np.log(r["prob_true"]) for r in rows
        ])),
        "high_confidence_errors": sum(
            r["correct"] == 0 and r["confidence"] >= 0.9
            for r in rows
        ),
    }

    out_dir = Path("outputs") / "bayes_eval_seed42"
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(rows)

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print(json.dumps(summary, indent=2))
    print(f"\nFiles saved in {out_dir}")


if __name__ == "__main__":
    main()
