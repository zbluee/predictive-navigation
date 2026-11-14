# Predictive Navigation

I'm using this project to better understand how an agent can learn about its location from noisy visual observations and movement.

The starting point is the work of Yeowon Kim and Yul H. R. Kang on predictive learning and cognitive maps. I'm particularly interested in why learning to predict future observations might produce representations of location and uncertainty that differ from those learned through reconstruction.

I'm starting with a small environment where some locations look identical. The plan is to build a Bayesian observer first, then compare simple predictive and reconstruction networks against it.

This is a learning project, not a reproduction of the paper.

## Current experiment

**01 Four-state navigation environment**

The agent moves around a circular world with four positions. It receives observations A, B, or C, but those observations can be noisy. Positions 0 and 2 have identical observation distributions.

The first step is to check that the environment behaves as intended before introducing Bayesian inference or neural networks.

## Setup

Requires Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Run the example:

```bash
python scripts/try_environment.py
```

Run the tests:

```bash
pytest -q
```

## References

- Yeowon Kim and Yul H. R. Kang, [*Predictive Learning Induces Probabilistic Cognitive Maps*](https://openreview.net/pdf?id=O9bbw7BLmP), NeurIPS 2024 NeuroAI Workshop.
- Yeowon Kim and Yul H. R. Kang, [*Predictive learning induces Bayesian cognitive maps in the hippocampus*](https://www.biorxiv.org/content/10.64898/2026.06.03.729991v1), 2026 preprint.

## Progress

See [research log](notes/research_log.md) for experiments, results, and changes.