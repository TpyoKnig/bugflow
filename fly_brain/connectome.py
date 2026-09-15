"""Build a signed neuron-to-neuron weight matrix to drive an LIF simulation.

Two sources:

- ``load_synthetic()``: a small random weighted graph, no external data or
  credentials required. This is what runs by default and what the bundled
  smoke test exercises.
- ``load_real()``: pulls an induced subgraph of the actual MaleCNS v1.0
  Drosophila connectome (166,700 neurons, released by Janelia FlyEM et al.,
  https://male-cns.janelia.org/) via the neuPrint API, for a chosen set of
  sensory/motor cell types. Requires ``pip install neuprint-python`` and a
  free neuPrint API token (https://neuprint.janelia.org -> account -> token).
  This is the same public dataset the "fly plays Beat Saber/Doom/Minecraft"
  posts used as their fixed structural connectivity.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ConnectomeGraph:
    ids: np.ndarray                 # (n,) neuron identifiers (bodyId or synthetic index)
    labels: list                    # (n,) human-readable cell type per neuron
    weights: np.ndarray             # (n, n) signed weight matrix; weights[i, j] = j -> i
    sensory: dict = field(default_factory=dict)  # group name -> index array
    motor: dict = field(default_factory=dict)    # group name -> index array
    positions: np.ndarray | None = None          # (n, 3) soma xyz for rendering; None = unknown

    @property
    def n(self) -> int:
        return len(self.ids)


def load_synthetic(
    n_hidden: int = 120,
    seed: int = 0,
    sensory_groups: tuple = ("left_eye", "right_eye"),
    motor_groups: tuple = ("escape_dn", "approach_dn", "turn_left_dn", "turn_right_dn"),
    sensory_size: int = 20,
    motor_size: int = 5,
    ff_density: float = 0.25,
    ff_weight: tuple = (0.5, 0.9),
    out_density: float = 0.3,
    out_weight: tuple = (0.4, 0.9),
    p_exc_out: float = 0.65,
) -> ConnectomeGraph:
    """A small random signed weighted graph, standing in for a real connectome.

    # ponytail: two-layer feedforward (sensory -> hidden -> motor), no
    # recurrent hidden/hidden or motor/motor connections. A randomly wired
    # *recurrent* toy network almost always lands on a runaway all-firing
    # attractor (row sums of a dense random signed matrix exceed v_thresh
    # once activity is nonzero) or on total silence; there's no cheap way to
    # keep a random recurrent graph in the "transient, differentiated
    # bursts" regime a demo needs. Feedforward-only can't self-sustain, so
    # bursts always decay between beats. Swap in load_real() for the actual
    # (recurrent) MaleCNS connectome, where real synaptic statistics (not
    # uniform random weights) are what keeps activity bounded.
    """
    rng = np.random.default_rng(seed)

    n_sensory = len(sensory_groups) * sensory_size
    n_motor = len(motor_groups) * motor_size
    n = n_sensory + n_hidden + n_motor

    labels = (
        [f"{g}_{i}" for g in sensory_groups for i in range(sensory_size)]
        + [f"hidden_{i}" for i in range(n_hidden)]
        + [f"{g}_{i}" for g in motor_groups for i in range(motor_size)]
    )
    ids = np.arange(n)

    sensory_idx = np.arange(0, n_sensory)
    hidden_idx = np.arange(n_sensory, n_sensory + n_hidden)
    motor_idx = np.arange(n_sensory + n_hidden, n)

    weights = np.zeros((n, n))
    ff_mask = rng.random((n_hidden, n_sensory)) < ff_density
    weights[np.ix_(hidden_idx, sensory_idx)] = ff_mask * rng.uniform(*ff_weight, (n_hidden, n_sensory))

    out_mask = rng.random((n_motor, n_hidden)) < out_density
    out_sign = rng.choice([1.0, -1.0], size=(n_motor, n_hidden), p=[p_exc_out, 1 - p_exc_out])
    weights[np.ix_(motor_idx, hidden_idx)] = out_mask * out_sign * rng.uniform(*out_weight, (n_motor, n_hidden))

    sensory = {
        g: np.arange(i * sensory_size, (i + 1) * sensory_size)
        for i, g in enumerate(sensory_groups)
    }
    motor = {
        g: n_sensory + n_hidden + np.arange(i * motor_size, (i + 1) * motor_size)
        for i, g in enumerate(motor_groups)
    }
    return ConnectomeGraph(ids=ids, labels=labels, weights=weights, sensory=sensory, motor=motor)



# Neurotransmitters treated as inhibitory. In Drosophila, GABA and glutamate
# are the main inhibitory transmitters (glutamate acts via GluCl); histamine
# is the photoreceptor transmitter and is inhibitory on its targets.
# Acetylcholine (the main excitatory transmitter), dopamine, serotonin,
# octopamine, and unknown/None are all treated as excitatory here.
_INHIBITORY_NT = {"gaba", "glutamate", "histamine"}

# Neuron properties carrying a transmitter prediction in the MaleCNS neuPrint
# dataset, in order of preference (curated -> per-cell-type -> per-neuron).
_NT_COLUMNS = ("consensusNt", "celltypePredictedNt", "predictedNt")


def _patterns(spec) -> list:
    return list(spec) if isinstance(spec, (list, tuple)) else [spec]


def load_real(
    token: str,
    sensory_groups: dict,
    motor_groups: dict,
    server: str = "https://neuprint.janelia.org",
    dataset: str = "male-cns:v1.0",
    min_total_weight: int = 3,
    weight_scale: float = 20.0,
) -> ConnectomeGraph:
    """Fetch a real induced subgraph of the MaleCNS v1.0 connectome via neuPrint.

    Args:
        token: neuPrint API token (https://neuprint.janelia.org -> account -> Auth Token).
        sensory_groups: ``{group_name: regex or [regex, ...]}``. Each regex is
            matched (full match, neo4j ``=~``) against the neuron's ``instance``
            string. In MaleCNS an instance is ``<type>_<side>``, e.g.
            ``"LPLC2_L"``; ``"LPLC2_.*"`` matches both sides.
            Verify names first with ``python3 -m fly_brain.lookup <regex>``.
        motor_groups: same shape, for descending/motor neurons, e.g.
            ``{"escape_dn": "DNp01\\(GF\\)_.*"}`` (giant fiber).
        min_total_weight: drop connections weaker than this synapse count.
        weight_scale: divide raw synapse counts by this. Raw counts run into
            the hundreds; the LIF threshold is 1.0, so this sets how many
            simultaneous presynaptic spikes it takes to fire a neuron.
            Tune it (see README "Calibrating real mode").

    Returns:
        ConnectomeGraph with ``weights[post, pre] = signed synapse count / weight_scale``.
        ``labels`` are instance strings. Only neurons participating in at least
        one kept connection are included (that's what fetch_adjacencies returns).

    Requires: ``pip install neuprint-python`` (pulls in pandas).
    """
    from neuprint import Client, NeuronCriteria as NC, fetch_adjacencies

    client = Client(server, dataset=dataset, token=token)

    all_patterns = [p for spec in {**sensory_groups, **motor_groups}.values() for p in _patterns(spec)]
    # Two separate criteria objects: fetch_adjacencies assigns each its own cypher match variable.
    sources = NC(instance=all_patterns, regex=True, client=client)
    targets = NC(instance=all_patterns, regex=True, client=client)

    # Returns (neurons_df, conn_df). omit_rois=True gives one row per (pre, post)
    # pair with the total weight, which is all we need. Missing properties come
    # back as null columns rather than erroring.
    neuron_df, conn_df = fetch_adjacencies(
        sources, targets,
        min_total_weight=min_total_weight,
        omit_rois=True,
        properties=["type", "instance", "somaLocation", *_NT_COLUMNS],
        client=client,
    )
    if conn_df.empty:
        raise ValueError(
            f"No connections with weight >= {min_total_weight} among instances matching {all_patterns}. "
            "Check the patterns with `python3 -m fly_brain.lookup <regex>`."
        )

    ids = neuron_df["bodyId"].to_numpy()
    pos = {int(b): i for i, b in enumerate(ids)}
    n = len(ids)

    weights = np.zeros((n, n))
    pre_idx = conn_df["bodyId_pre"].map(pos).to_numpy()
    post_idx = conn_df["bodyId_post"].map(pos).to_numpy()
    weights[post_idx, pre_idx] = conn_df["weight"].to_numpy(dtype=float) / weight_scale

    # Sign each presynaptic neuron's outgoing column by its transmitter.
    nt = None
    for col in _NT_COLUMNS:
        if col in neuron_df.columns:
            nt = neuron_df[col] if nt is None else nt.fillna(neuron_df[col])
    if nt is None:
        print("warning: no neurotransmitter column found; treating every connection as excitatory")
        nt = neuron_df["bodyId"] * 0  # all non-inhibitory
    inhibitory = nt.astype(str).str.lower().isin(_INHIBITORY_NT).to_numpy()
    weights[:, inhibitory] *= -1

    labels = neuron_df["instance"].astype(str).tolist()

    def _group_indices(spec_map):
        import re
        out = {}
        for group, spec in spec_map.items():
            combined = re.compile("|".join(f"(?:{p})" for p in _patterns(spec)))
            out[group] = np.array([i for i, lbl in enumerate(labels) if combined.fullmatch(lbl)], dtype=int)
            if out[group].size == 0:
                print(f"warning: group {group!r} matched no neurons in the fetched subgraph")
        return out

    positions = None
    if "somaLocation" in neuron_df.columns:
        loc = neuron_df["somaLocation"]
        if loc.notna().all():
            # fetch_neurons gives [x,y,z]; fetch_adjacencies gives {"coordinates": [x,y,z]}
            positions = np.array([list(map(float, p["coordinates"] if isinstance(p, dict) else p))
                                  for p in loc], dtype=float)

    return ConnectomeGraph(
        ids=ids, labels=labels, weights=weights,
        sensory=_group_indices(sensory_groups), motor=_group_indices(motor_groups),
        positions=positions,
    )
