"""Reward-modulated linear readout: the trainable interface on a fixed brain.

The connectome's synapses never change (they are the fly's biology). What
learns is a small linear readout from brain spike counts to a softmax over
discrete choices, updated by a three-factor rule:

    W += lr * (reward - baseline) * (onehot(choice) - probs) outer counts

`(reward - baseline)` is a reward prediction error, i.e. the dopamine
analogue in the "fly plays Doom, dopamine-reinforced" recipe. For a linear
softmax policy this is exactly REINFORCE with a baseline; the eligibility
factor `(onehot - probs) outer counts` is the Hebbian term tying the
update to the neurons that actually fired.

# ponytail: dense numpy, one shared learning rate, running-mean baseline.
"""

import numpy as np


class RewardModulatedReadout:
    def __init__(self, n_brain: int, n_choices: int, lr: float = 5.0,
                 baseline_decay: float = 0.95, seed: int = 0):
        self.W = np.zeros((n_choices, n_brain))
        self.lr = lr
        self.baseline_decay = baseline_decay
        self.baseline = 0.0
        self.rng = np.random.default_rng(seed)

    def choose(self, counts: np.ndarray, epsilon: float = 0.0) -> tuple[int, np.ndarray]:
        """Pick a choice given per-neuron spike counts: softmax over scores,
        but with probability `epsilon` explore uniformly instead. Without
        exploration the policy collapses onto the first above-baseline
        answer and never samples the rest of the palette.

        Returns (choice_index, choice_probabilities).
        """
        scores = self.W @ counts
        probs = np.exp(scores - scores.max())
        probs /= probs.sum()
        if epsilon > 0.0:
            probs = (1 - epsilon) * probs + epsilon / len(probs)
        return int(self.rng.choice(len(probs), p=probs)), probs

    def update(self, counts: np.ndarray, choice: int, probs: np.ndarray, reward: float) -> float:
        """Apply the three-factor update; returns the reward prediction error."""
        rpe = reward - self.baseline
        self.baseline = self.baseline_decay * self.baseline + (1 - self.baseline_decay) * reward
        elig = -probs
        elig[choice] += 1.0
        self.W *= 1 - 1e-3  # weight decay: keeps softmax from saturating and freezing the policy
        self.W += self.lr * rpe * np.outer(elig, counts)
        return rpe


def _selfcheck():
    """A readout should learn to pick choice 1 when neuron A dominates,
    choice 0 when neuron B dominates, within a few hundred episodes."""
    ro = RewardModulatedReadout(n_brain=2, n_choices=2, lr=2.0, seed=1)
    wins = 0
    for ep in range(400):
        a_dominant = ep % 2 == 0
        counts = np.array([3.0, 0.5]) if a_dominant else np.array([0.5, 3.0])
        choice, probs = ro.choose(counts)
        correct = 1 if a_dominant else 0
        reward = 1.0 if choice == correct else 0.0
        ro.update(counts, choice, probs, reward)
        if ep >= 300 and reward == 1.0:
            wins += 1
    assert wins > 90, f"readout did not learn: {wins}/100 late-episode wins"
    print(f"OK: readout learned the mapping ({wins}/100 late-episode wins)")


if __name__ == "__main__":
    _selfcheck()
