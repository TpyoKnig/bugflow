"""Assert-based smoke test: run with `python3 -m fly_brain.test_lif`.

Proves excitation propagates and inhibition suppresses, on a 3-neuron chain:
neuron0 --excite--> neuron1 --inhibit--> neuron2
"""
import numpy as np

from fly_brain.lif import LIFNetwork


def test_excitation_propagates_and_inhibition_suppresses():
    weights = np.array([
        [0.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
        [0.0, -5.0, 0.0],
    ])
    net = LIFNetwork(weights, tau=8.0, v_thresh=1.0)

    def stim(t):
        ext = np.zeros(3)
        ext[0] = 0.5  # constant strong drive to neuron 0 only
        ext[2] = 0.3  # neuron 2 would spike on its own without inhibition
        return ext

    spikes = net.run(steps=50, stim_fn=stim)

    assert spikes[:, 0].any(), "driven neuron 0 never spiked"
    assert spikes[:, 1].any(), "excitatory input from neuron 0 never made neuron 1 spike"

    # Without inhibition neuron 2 (driven by ext[2]=0.3 alone) spikes repeatedly.
    baseline = LIFNetwork(np.zeros((3, 3)), tau=8.0, v_thresh=1.0)
    baseline_spikes = baseline.run(steps=50, stim_fn=lambda t: np.array([0.0, 0.0, 0.3]))
    assert baseline_spikes[:, 2].sum() > 0, "test setup broken: neuron 2 should spike without inhibition"

    assert spikes[:, 2].sum() < baseline_spikes[:, 2].sum(), \
        "inhibitory input from neuron 1 failed to suppress neuron 2's spiking"

    print("OK: excitation propagated (n1 spikes)=%d, inhibition suppressed n2 (%d < %d)" % (
        spikes[:, 1].sum(), spikes[:, 2].sum(), baseline_spikes[:, 2].sum()))


if __name__ == "__main__":
    test_excitation_propagates_and_inhibition_suppresses()
