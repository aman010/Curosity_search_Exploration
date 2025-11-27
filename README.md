The development of Curiosity-Driven Search has significantly advanced Reinforcement Learning, primarily by helping agents explore environments with sparse rewards. However, integrating safety into curiosity-driven search within Multi-Agent Environments (MARL) remains a critical challenge.This problem is highly relevant to areas like adversarial detection, where one agent attempts to deceive or "fool" another agent (e.g., via strategically misleading actions or "keywords") to gain a positional advantage. A truly robust agent must develop the best strategies not only to protect itself but also to build winning strategies that anticipate and overcome adversarial tactics.



## Key Novelty Mechanisms and Implementation Details

Our safety framework is built on two core risk metrics applied via a weighted penalty ($\mathbf{-\beta \times R_{\text{Total}}}$) to the final reward.

1. Macro-Strategic Risk ($\mathbf{R_{MS}}$) - Value Uncertainty

This component addresses the uncertainty of the game state itself, penalizing 50/50 fights.

Metric: Derived from the Smoothed Advantage (Current Agent Value - Opponent Value).

Smoothing: We use an Exponential Moving Average (EMA) on the instantaneous advantage to reduce noise and prevent overreaction to minor fluctuations. This ensures the risk signal is based on a reliable trend, not a single observation.

Penalty: $\mathbf{R_{MS}}$ is highest when the Smoothed Advantage is near zero (an uncertain state) and lowest when the outcome is clear (decisive win or loss).

2. Policy Divergence Risk ($\mathbf{R_{PD}}$) - Policy Uncertainty

This component addresses the uncertainty of the agent's policy, which is crucial for safety.

Metric: Calculated using the Policy Entropy (Shannon Entropy) of the action probability distribution.

Goal: High entropy means the policy is "confused" and unsure of the best action.

Penalty: $\mathbf{R_{PD}}$ is highest when the policy is uncertain (high entropy) and lowest when the policy is certain (low entropy). This drives the agent to quickly find and commit to confident, decisive actions.

Combined Risk: The total macro-safety penalty is based on the maximum of the two risks, $\mathbf{R_{\text{Total}} = \max(R_{MS}, R_{PD})}$, ensuring the agent is penalized if either the game state or its own policy is unstable.

| Experiment | Mean Agent 1 | Mean Agent 2 | Std Agent 1 | Std Agent 2 |
|------------|--------------|--------------|-------------|-------------|
| Simple_approach_augmented + Curiosity 0.1 | -43.902 | -64.785 | 122.346 | 100.005 |
| Simple_approch_augmented + Curiosity 0.8 | -118.806 | -128.701 | 61.172 | 72.824 |
| Simple_approach_non_augmented  + Curiosity 0.1 | 100.083 | 75.196 | 153.596 | 99.376 |
| Simple_apprach_non_augmented + Curiosity 0.8 | -192.793 | -109.615 | 108.983 | 71.083 |
| Feature_base_augmented + Curiosity 0.1 | 22.023 | 20.851 | 24.807 | 31.701 |
| Feature_based_augmented + Curiosity 0.8 | -95.245 | -95.889 | 52.715 | 53.021 |
| Feature_based_non_augmented + Curiosity 0.1 | 38.071 | 71.000 | 47.126 | 100.060 |
| Feature_baded_non_augmented + Curiosity 0.8 | 114.698 | -258.248 | 396.172 | 141.704 |
| RND_augmented + Curioisty 0.1 | 42.970 | 48.511 | 58.676 | 73.520 |
| RND_augmented + Curosity 0.8 | -109.738 | -109.156 | 59.772 | 59.448 |
| RND_non_augmented + Curosity 0.1 | 53.226 | 52.769 | 82.567 | 65.052 |


