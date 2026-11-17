
import json
import platform
from pathlib import Path

import numpy as np

from predictive_navigation.data import make_dataset
from predictive_navigation.environment import RingWorld


def main():
    world = RingWorld()
    moves = 20

    splits = {
        "train": {"count": 600, "seed": 100},
        "val": {"count": 100, "seed": 101},
        "test": {"count": 100, "seed": 102},
    }

    output_dir = Path("outputs/datasets")
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, config in splits.items():
        rng = np.random.default_rng(config["seed"])

        dataset = make_dataset(
            world,
            rng,
            n_trajectories=config["count"],
            moves=moves,
        )

        np.savez_compressed(
            output_dir / f"{name}.npz",
            **dataset,
        )

        print(
            f"{name}: "
            f"observations={dataset['observations'].shape}, "
            f"actions={dataset['actions'].shape}, "
            f"states={dataset['states'].shape}"
        )

    metadata = {
        "splits": splits,
        "moves": moves,
        "observation_labels": list(world.OBSERVATIONS),
        "action_values_by_id": [-1, 0, 1],
        "observation_model": world.observation_model.tolist(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }

    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    print(f"\nSaved datasets to {output_dir}")


if __name__ == "__main__":
    main()
