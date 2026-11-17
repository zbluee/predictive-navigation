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

### Part 5: High-confidence error analysis

**Date:** November 15, 2026

**Goal:** Examine why the Bayesian filter makes confident errors and how those errors persist.

### Results

Of 3,157 predictions with at least 90% confidence, 20 were incorrect (0.63%). These occurred in 7 of 200 trajectories.

Episode 126 had six consecutive high-confidence errors (t=5–10). A misleading observation reinforced an incorrect belief, while subsequent ambiguous observations failed to correct it.

At t=11, a more informative observation increased the probability assigned to the true position from 1.54% to 11.11%. However, the filter still predicted the wrong position at t=12.

### Interpretation

Incorrect beliefs can persist when observations provide insufficient evidence to distinguish locations. New evidence can reduce confidence in an incorrect estimate without immediately correcting it.

### Limitations

Errors within trajectories are correlated. We have not measured recovery times across all trajectories or evaluated calibration.

### Next step

Evaluate performance across multiple random seeds before introducing predictive and reconstruction networks.

## Experiment 03: Predictive and reconstruction learning

**Date:** Nov 17, 2026

### Part 1: Dataset preparation

**Goal:** Prepare fixed trajectories for comparing predictive and reconstruction learning.

**Setup:**
- Training: 600 trajectories (seed 100)
- Validation: 100 trajectories (seed 101)
- Test: 100 trajectories (seed 102)
- 20 movements and 21 observations per trajectory
- Same four-state environment and observation model

**Results:**

*Complete after generating the datasets and running the tests.*

**Limitations:** All trajectories come from the same simplified environment. The splits test performance on new trajectories, not generalization to different environments.

**Next step:** Implement a small recurrent network and train it to predict the next observation.

### Part 2: Recurrent model

**Date:** Nov 17, 2026

**Goal:** Build a small recurrent model for processing observation and movement history.

**Implementation:** GRU with 16 hidden units and a linear output layer predicting probabilities over three observation categories.

**Results:** Complete after testing the forward pass, sequence shapes, gradient computation, and absence of future-observation leakage.

**Limitations:** The model is untrained. We have not evaluated prediction accuracy or whether the hidden states encode location.

**Next step:** Train the predictive network on the prepared trajectories and evaluate next-observation prediction on the validation set.

### Part 3: Predictive network training

**Date:** Nov 17, 2026

**Goal:** Train a small GRU to predict the next observation using observation and movement history.

**Setup:**
- GRU with 16 hidden units
- Adam optimizer, learning rate 0.003
- Batch size 64, maximum 30 epochs
- Model seed 7
- 600 training and 100 validation trajectories
- Best checkpoint selected by validation cross-entropy

**Results:**

*Record the best validation loss, accuracy, baseline comparison, and selected epoch after running.*

**Limitations:** Prediction performance alone does not show whether the hidden states represent position or uncertainty.

**Next step:** Examine the learning curves and compare against a non-recurrent predictor before training the reconstruction control.

### Part 4: Extended training and architecture comparison

**Date:** Nov 17, 2026

**Goal:** Investigate why the predictive GRU underperformed the conditional baseline.

### Results

Extending the original GRU to 120 epochs gave a best validation loss of 1.0262 and 44.75% accuracy, still below the conditional baseline (0.9355, 58.05%).

Replacing the linear prediction head with an MLP improved validation loss to **0.7716** and accuracy to **71.85%**, with the best checkpoint at epoch 48.

Validation loss increased after the best checkpoint while training loss continued decreasing, suggesting overfitting.

### Interpretation

The more expressive prediction head substantially improved next-observation prediction. However, the result does not establish that recurrent memory or Bayesian-like location representations caused the improvement.

### Next step

Compare against a non-recurrent MLP using the same current observation and upcoming action to isolate the benefit of recurrent history.

### Part 5: Feedforward comparison

**Date:** Nov 17, 2026

**Goal:** Test whether access to observation and movement history improves next-observation prediction.

**Setup:** Compared the predictive GRU (1,939 parameters) with a feedforward MLP (2,043 parameters), using the same training and validation data, optimizer, learning rate, batch size, seed, and 120-epoch budget.

### Results

The feedforward MLP achieved **0.9341 validation loss and 57.35% accuracy** at epoch 47, close to the conditional baseline (0.9355 loss, 58.05% accuracy).

The GRU + MLP performed substantially better, achieving **0.7716 loss and 71.85% accuracy**.

### Interpretation

The results suggest that observation and movement history provide useful information for predicting future observations. However, we haven't directly tested what the recurrent hidden states represent.

### Limitations

The models have similar parameter counts but different architectures. Results are from one model seed and one validation split.

### Next step

Test whether disrupting the GRU's history reduces prediction performance, then train the matched reconstruction control.

### Part 6: Recurrent memory ablation

**Date:** Nov 17, 2026

**Goal:** Test whether the trained predictive GRU depends on recurrent memory.

**Results:** Resetting the hidden state at every step reduced validation accuracy from 71.85% to 58.35% and increased loss from 0.7716 to 0.9574.

**Interpretation:** The trained GRU relies on information carried across time for prediction.

**Limitation:** Resetting memory changes the conditions encountered during training. The result does not establish that the hidden state represents Bayesian location beliefs.

**Next step:** Train a reconstruction GRU with the same architecture and inputs.

### Part 7: Reconstruction network

**Date:** Nov 17, 2026

**Goal:** Train a reconstruction GRU as a control for the predictive-learning experiment.

**Setup:** Same GRU + MLP architecture (1,939 parameters), dataset, optimizer, batch size, seed and 120-epoch budget as the predictive model. The target was changed from the next observation to the current observation.

**Results:** Reconstruction accuracy reached 100% by epoch 5. The best checkpoint was at epoch 120, with 100% validation accuracy and near-zero cross-entropy loss.

**Interpretation:** The reconstruction model successfully learned to reproduce its current observation. Because the target was already available as input, this does not establish that the network learned location or uncertainty representations.

**Next step:** Freeze the predictive and reconstruction GRUs, train identical linear location decoders, and compare their performance on held-out trajectories.