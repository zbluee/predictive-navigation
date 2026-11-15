# Research Log

## Experiment 01: Four-state environment

**Date:** Nov 14, 2026

### Goal

Build a small navigation environment with ambiguous locations and noisy observations. Check that movement, observation sampling, and random seeds behave as expected.

### Setup

- Four positions in a circle
- Actions: -1, 0, +1
- Observations: A, B, C
- Positions 0 and 2 share the same observation probabilities
- Correct observation probability: 0.8
- Random seed: 42

### Checks

- [x] Movement wraps around correctly
- [x] Observation probabilities sum to one
- [x] Positions 0 and 2 have identical observation distributions
- [x] Repeated runs with the same seed produce the same observations
- [x] All tests pass

### Results

Ran pytest -q. All five tests passed.

The tests confirm that the movement rules, observation probabilities, and seeded sampling behave as intended.

This verifies the basic env't implementation. It does not yet tell us whether an agent can infer its location from noisy observations.

### Limitations

The world has only four discrete positions. Movement is exact, and there is no heading, continuous space, or realistic vision. The environment is meant to explain the inference problem, not reproduce the paper's simulator.

### Next step

Implement Bayesian filtering to estimate location from action and observation history.

### Problems and corrections

no implementation problems encountered so far.

## Experiment 02: Bayesian filtering

**Date:** Nov 14, 2026

### Part 1: Belief prediction

Implemented the prediction step of a Bayesian filter. It shifts the probability distribution over positions based on the agent's movement.

Since movement is deterministic in our four-state environment, NumPy's `roll()` function is sufficient.

### Checks

- [x] Forward movement shifts the belief correctly
- [x] Backward movement shifts the belief correctly
- [x] Staying still leaves the belief unchanged
- [x] Probabilities still sum to one
- [x] Invalid movement commands are rejected

### Results

Ran `pytest tests/test_bayes_filter.py -v`. All five tests passed.

The predicted belief shifts correctly with each action while preserving the total probability of 1.

### Limitations

This implementation assumes movement is exact. It cannot yet handle movement uncertainty, such as an agent attempting to move forward but remaining in place.

### Next step

Implement the observation update using Bayes' rule. Then combine prediction and update to track the agent's belief across a trajectory.

Part 2: Observation update

Implemented the Bayesian observation update. It combines the prior belief with the likelihood of a new observation and normalizes the result.

For example, with equal prior probabilities at positions 1 and 3, observing C increases the probability of position 3 to approximately 88.9%.

### Checks

Observing C favors position 3 over position 1

Observing B favors position 1 over position 3

Observing A does not distinguish positions 0 and 2 when their prior probabilities are equal

Invalid observations are rejected

All existing tests still pass

### Results

All nine Bayesian-filter tests passed, including the four new observation-update tests.

The results matched our hand calculations. Observing C produced approximately 11.1% probability for position 1 and 88.9% for position 3. Observing A left the equal belief between positions 0 and 2 unchanged.

### Limitations

The observation probabilities are fixed and known to the Bayesian observer. We haven't tested the filter across full trajectories or compared its estimates with the true positions.

### Next step

Run the Bayesian filter over simulated trajectories and examine how its belief changes with movement and observations.

### Part 3: Bayesian filtering over a trajectory

### Date: Nov 15, 2026

### Goal

Run the Bayesian filter over several movements and observations to see how its belief changes over time.

### Setup

Four-state environment from Experiment 01

Initial belief: uniform across all four positions

Initial true position: 0 (known only to the simulator)

Actions: [1, 1, 1, 0, -1, 1, 1]

Random seed: 42

Exact movement and known observation probabilities

### Results

The filter updated its belief across the trajectory using movement and visual observations.

One interesting case occurred at t = 2:

True position: 2

Observation: B

Most likely position: 2

Posterior probability: approximately 79%

Although B is more likely at position 1, the filter still assigned 79% probability to position 2.

This happened because the previous belief strongly favored position 1. After moving forward, that probability shifted to position 2. The new observation B provided conflicting evidence, but it wasn't enough to outweigh the prior belief.

This shows why considering movement and observation history matters. The filter does not simply choose the position associated with the latest observation.

### Limitations

The Bayesian observer knows the observation probabilities and movement rules. Movement is exact, and the environment contains only four positions.

This is an ideal observer for our simplified environment, not a reproduction of the full Bayesian observer in Kim and Kang's paper.

### Next step

Evaluate the filter across multiple trajectories to examine location accuracy and how its uncertainty changes with noisy or ambiguous observations.

Part 4: Evaluation across multiple trajectories

Date: November 15, 2026

Setup:

200 trajectories, 20 movements each

4,200 evaluated observations

Random initial positions and actions

Seed: 42

Python 3.12.8, NumPy 2.5.4

Results:

Metric

Result

Location accuracy

92.05%

Mean probability of true position

89.67%

Mean confidence

92.27%

Mean entropy

0.185 nats

Mean log loss

0.195

Errors with ≥90% confidence

20

The filter correctly identified the location in 3,866 of 4,200 observations. However, it made 20 errors with confidence of at least 90%.

This shows that even an ideal observer can make confident mistakes when observations are noisy. The aggregate results alone do not establish probability calibration.

Overall findings

The Bayesian filter combines movement and observation history to estimate location. It can resolve ambiguous observations, retain useful prior information, and adjust uncertainty when new evidence conflicts with previous beliefs.

Limitations

The observer knows the correct movement and observation models. The environment is small, movement is exact, and observations within each trajectory are correlated.

No neural network has been trained. These experiments demonstrate Bayesian filtering in a simplified environment, not a reproduction of Kim and Kang's work.

Next steps

Examine location errors and uncertainty over time.

Evaluate variation across trajectories and random seeds.

Build matched predictive and reconstruction networks using the same environment.

### Part 4: Evaluation across multiple trajectories

**Date:** November 15, 2026

**Setup:**
- 200 trajectories, 20 movements each
- 4,200 evaluated observations
- Random initial positions and actions
- Seed: 42
- Python 3.12.8, NumPy 2.5.4

**Results:**

| Metric | Result |
|---|---:|
| Location accuracy | 92.05% |
| Mean probability of true position | 89.67% |
| Mean confidence | 92.27% |
| Mean entropy | 0.185 nats |
| Mean log loss | 0.195 |
| Errors with ≥90% confidence | 20 |

The filter correctly identified the location in 3,866 of 4,200 observations. However, it made 20 errors with confidence of at least 90%.

This shows that even an ideal observer can make confident mistakes when observations are noisy. The aggregate results alone do not establish probability calibration.

### Overall findings

The Bayesian filter combines movement and observation history to estimate location. It can resolve ambiguous observations, retain useful prior information, and adjust uncertainty when new evidence conflicts with previous beliefs.

### Limitations

The observer knows the correct movement and observation models. The environment is small, movement is exact, and observations within each trajectory are correlated.

No neural network has been trained. These experiments demonstrate Bayesian filtering in a simplified environment, not a reproduction of Kim and Kang's work.

### Next steps

1. Examine location errors and uncertainty over time.
2. Evaluate variation across trajectories and random seeds.
3. Build matched predictive and reconstruction networks using the same environment.