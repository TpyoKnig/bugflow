"""Assert-based test for load_real()'s data wrangling, with neuprint mocked.

Run: python3 -m fly_brain.test_connectome   (needs pandas; no network, no token)

Checks the things that were wrong in the first draft and that a live run
would only reveal after a slow neuPrint fetch:
- fetch_adjacencies return order is (neurons_df, conn_df)
- weights[post, pre] orientation
- inhibitory presyn neurons get negative outgoing weights, using the
  consensusNt > celltypePredictedNt > predictedNt precedence
- neurons that only appear as targets are kept
- group regexes match on instance and produce correct indices
"""
import sys
import types

import numpy as np


def _install_fake_neuprint(neurons_df, conn_df, calls):
    fake = types.ModuleType("neuprint")

    class Client:
        def __init__(self, server, dataset, token):
            calls.append(("Client", server, dataset, token))

    class NeuronCriteria:
        def __init__(self, **kw):
            calls.append(("NC", kw))

    def fetch_adjacencies(sources, targets, **kw):
        calls.append(("fetch_adjacencies", kw))
        return neurons_df, conn_df

    fake.Client = Client
    fake.NeuronCriteria = NeuronCriteria
    fake.fetch_adjacencies = fetch_adjacencies
    sys.modules["neuprint"] = fake


def test_load_real_wrangling():
    import pandas as pd
    from fly_brain.connectome import load_real

    # 4 neurons: two sensory (one GABAergic), one hidden, one motor that only receives.
    neurons_df = pd.DataFrame({
        "bodyId": [101, 102, 200, 300],
        "type": ["LPLC2", "LPLC2", "GF", "DNp01"],
        "instance": ["LPLC2_L", "LPLC2_R", "GF_L", "DNp01_L"],
        "consensusNt": [None, "gaba", None, None],
        "celltypePredictedNt": ["acetylcholine", "acetylcholine", None, None],
        "predictedNt": ["acetylcholine", "acetylcholine", "acetylcholine", "acetylcholine"],
    })
    conn_df = pd.DataFrame({
        "bodyId_pre": [101, 102, 200],
        "bodyId_post": [200, 200, 300],
        "weight": [40, 20, 60],
    })
    calls = []
    _install_fake_neuprint(neurons_df, conn_df, calls)

    g = load_real(
        token="t",
        sensory_groups={"eyes": "LPLC2_.*"},
        motor_groups={"escape": "DNp01_L"},
        weight_scale=20.0,
    )

    assert g.n == 4, "target-only neuron (300) must be kept"
    assert list(g.ids) == [101, 102, 200, 300]

    # weights[post, pre]
    assert g.weights[2, 0] == 2.0, "101 -> 200 should be +40/20"
    assert g.weights[2, 1] == -1.0, "102 (consensusNt=gaba beats celltypePredictedNt=ach) -> 200 should be -20/20"
    assert g.weights[3, 2] == 3.0, "200 -> 300 should be +60/20"
    assert g.weights[0, 2] == 0.0, "no reverse edge"

    assert list(g.sensory["eyes"]) == [0, 1]
    assert list(g.motor["escape"]) == [3]

    fa_kwargs = next(c[1] for c in calls if c[0] == "fetch_adjacencies")
    assert fa_kwargs["omit_rois"] is True
    for col in ("consensusNt", "celltypePredictedNt", "predictedNt"):
        assert col in fa_kwargs["properties"], f"{col} must be requested or sign inference silently no-ops"
    nc_kwargs = [c[1] for c in calls if c[0] == "NC"]
    assert all(kw["regex"] is True and "instance" in kw for kw in nc_kwargs)

    print("OK: load_real wrangling (orientation, sign precedence, target-only neurons, groups)")


if __name__ == "__main__":
    test_load_real_wrangling()
