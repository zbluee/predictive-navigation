
import csv
from pathlib import Path

import numpy as np

from predictive_navigation.environment import RingWorld
from predictive_navigation.bayes_filter import predict_belief
from collections import Counter


def main():
    path = Path("outputs/bayes_eval_seed42/results.csv")

    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))

    world = RingWorld()

    high_confidence = [
        row for row in rows
        if float(row["confidence"]) >= 0.9
    ]

    errors = [
        row for row in high_confidence
        if int(row["correct"]) == 0
    ]

    error_counts = Counter(
    int(row["episode"]) for row in errors
    )

    print(f"\nEpisodes with high-confidence errors: {len(error_counts)}")

    for episode, count in error_counts.most_common():
        times = [
            int(row["time"])
            for row in errors
            if int(row["episode"]) == episode
        ]
        print(f"Episode {episode}: {count} errors at t={times}")

    print(f"High-confidence predictions: {len(high_confidence)}")
    print(f"High-confidence errors: {len(errors)}")
    print(
        f"Error rate among high-confidence predictions: "
        f"{len(errors) / len(high_confidence):.2%}"
    )

    # Find the previous step within each trajectory.
    lookup = {
        (int(row["episode"]), int(row["time"])): row
        for row in rows
    }

    for row in errors[:3]:
        episode = int(row["episode"])
        t = int(row["time"])
        true_state = int(row["true_state"])
        estimate = int(row["estimate"])
        observation = row["observation"]

        if t == 0:
            prior = np.full(world.num_states, 0.25)
        else:
            previous = lookup[(episode, t - 1)]

            previous_belief = np.array([
                float(previous[f"belief_{i}"])
                for i in range(world.num_states)
            ])

            prior = predict_belief(
                previous_belief, int(row["action"])
            )

        obs_index = world.OBSERVATIONS.index(observation)
        likelihoods = world.observation_model[:, obs_index]

        print(f"\nEpisode {episode}, t={t}")
        print(f"True position: {true_state}")
        print(f"Observation: {observation}")
        print(f"Estimated position: {estimate}")
        print(f"Confidence: {float(row['confidence']):.2%}")

        print(
            f"Prior P(true): {prior[true_state]:.4f}, "
            f"P(estimated): {prior[estimate]:.4f}"
        )
        print(
            f"Observation likelihood at true position: "
            f"{likelihoods[true_state]:.2f}"
        )
        print(
            f"Observation likelihood at estimated position: "
            f"{likelihoods[estimate]:.2f}"
        )

    print("\nEpisode 126: steps 4–12")

    for row in rows:
        if int(row["episode"]) == 126 and 4 <= int(row["time"]) <= 12:
            print(
                f"t={row['time']:>2} "
                f"action={row['action']:>2} "
                f"true={row['true_state']} "
                f"obs={row['observation']} "
                f"estimate={row['estimate']} "
                f"confidence={float(row['confidence']):.2%} "
                f"P(true)={float(row['prob_true']):.2%}"
            )


if __name__ == "__main__":
    main()
