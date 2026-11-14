
import numpy as np

from predictive_navigation.environment import RingWorld


def main():
    rng = np.random.default_rng(42)
    world = RingWorld()

    state = 0
    actions = [1, 1, 1, 1, 1, 1]

    print("Time | Action | True state | Observation")
    print("-" * 45)

    observation = world.observe(state, rng)
    print(f"{0:>4} | {'start':>6} | {state:>10} | {observation}")

    for t, action in enumerate(actions, start=1):
        state = world.move(state, action)
        observation = world.observe(state, rng)

        print(
            f"{t:>4} | {action:>6} | "
            f"{state:>10} | {observation}"
        )


if __name__ == "__main__":
    main()
