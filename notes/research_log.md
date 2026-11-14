# Research Log

## Experiment 01 — Four-state environment

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

as expected, all test passed

### Limitations

The world has only four discrete positions. Movement is exact, and there is no heading, continuous space, or realistic vision. The environment is meant to explain the inference problem, not reproduce the paper's simulator.

### Next step

Implement Bayesian filtering to estimate location from action and observation history.

### Problems and corrections

nothing so far
