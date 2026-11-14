
import numpy as np


class RingWorld:
    #Four positions arranged in a circle

    OBSERVATIONS = ("A", "B", "C")

    def __init__(self):
        # Rows: true positions 0, 1, 2, 3
        # Columns: observations A, B, C
        self.observation_model = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.8, 0.1, 0.1],
            [0.1, 0.1, 0.8],
        ])

        self.num_states = 4

    def _check_state(self, state):
        if not 0 <= state < self.num_states:
            raise ValueError(f"Invalid state: {state}")

    def move(self, state, action):
        #Move one step forward, backward, or stay still
        self._check_state(state)

        if action not in (-1, 0, 1):
            raise ValueError("Action must be -1, 0, or 1")

        return (state + action) % self.num_states

    def observation_probs(self, state):
        #Return P(observation | true state)
        self._check_state(state)
        return self.observation_model[state].copy()

    def observe(self, state, rng):
        #Sample a noisy observation at the given position
        probs = self.observation_probs(state)

        return str(rng.choice(self.OBSERVATIONS, p=probs))
