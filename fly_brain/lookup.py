"""Look up MaleCNS neurons by instance regex, to verify group patterns before a real run.

Usage:
    NEUPRINT_TOKEN=... python3 -m fly_brain.lookup 'LPLC2_.*' 'DNp01\(GF\)_.*'

Prints one row per matching neuron: bodyId, instance, type, transmitter
columns, and synapse counts (pre = outputs, post = inputs). If a pattern
prints nothing, it won't match anything in load_real() either.
"""
from __future__ import annotations

import os
import sys


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print(__doc__)
        return 2
    token = os.environ.get("NEUPRINT_TOKEN")
    if not token:
        print("error: set NEUPRINT_TOKEN", file=sys.stderr)
        return 2

    import pandas as pd
    from neuprint import Client, NeuronCriteria as NC, fetch_neurons

    client = Client(
        os.environ.get("NEUPRINT_SERVER", "https://neuprint.janelia.org"),
        dataset=os.environ.get("NEUPRINT_DATASET", "male-cns:v1.0"),
        token=token,
    )
    cols = ["bodyId", "instance", "type", "consensusNt", "celltypePredictedNt", "predictedNt", "pre", "post"]
    df = fetch_neurons(NC(instance=argv, regex=True, client=client), omit_rois=True,
                       returned_columns=cols, client=client)
    if df.empty:
        print(f"no neurons match {argv}")
        return 1
    with pd.option_context("display.max_rows", 500, "display.width", 200):
        print(df[[c for c in cols if c in df.columns]].to_string(index=False))
    print(f"\n{len(df)} neurons")
    return 0


if __name__ == "__main__":
    sys.exit(main())
