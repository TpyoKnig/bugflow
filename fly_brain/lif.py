"""A minimal leaky integrate-and-fire network simulator.

# ponytail: dense numpy, one-step synaptic delay, no refractory period.
# Fine up to a few thousand neurons (a curated connectome subgraph). For the
# full 166k-neuron MaleCNS graph, swap `weights` for a scipy.sparse matrix
# and this still works unchanged (numpy @ sparse dispatches to sparse matmul).
"""
from __future__ import annotations

import numpy as np


class LIFNetwork:
    def __init__(self, weights: np.ndarray, tau: float = 8.0, v_thresh: float = 1.0,
                 v_reset: float = 0.0, dt: float = 1.0):
        n = weights.shape[0]
        assert weights.shape == (n, n), "weights must be square (n, n)"
        self.weights = weights
        self.decay = np.exp(-dt / tau)
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.v = np.zeros(n)
        self.spikes = np.zeros(n, dtype=bool)

    @property
    def n(self) -> int:
        return len(self.v)

    def step(self, ext_input: np.ndarray) -> np.ndarray:
        """Advance one timestep given external drive; returns this step's spikes."""
        syn_input = self.weights @ self.spikes.astype(float)
        self.v = self.decay * self.v + ext_input + syn_input
        self.spikes = self.v >= self.v_thresh
        self.v = np.where(self.spikes, self.v_reset, self.v)
        return self.spikes

    def run(self, steps: int, stim_fn) -> np.ndarray:
        """Run `steps` timesteps. `stim_fn(t) -> (n,) external input array`.

        Returns a (steps, n) bool array of spikes.
        """
        out = np.zeros((steps, self.n), dtype=bool)
        for t in range(steps):
            out[t] = self.step(stim_fn(t))
        return out
