"""Train the fly brain to design n8n workflows.

Each episode presents one workflow brief as a sustained stimulus to that
brief's sensory group. The fixed connectome (synthetic by default, real
MaleCNS with --mode real) spikes for a while; per-neuron spike counts feed
two reward-modulated readouts that vote for a trigger node and an action
node from a palette of real n8n node types. Reward: 1.0 both correct for
the brief, 0.5 one correct, 0.0 otherwise. The readouts' synapses update
by the dopamine-style rule in rl.py; the connectome itself never changes.

After training, the fly's greedy (argmax) design for each brief is exported
as an importable n8n workflow JSON.

    .venv/bin/python -m fly_brain.designer --episodes 800
"""

import argparse
import csv
import json
import os
import uuid

import numpy as np

from .connectome import load_synthetic, load_real
from .lif import LIFNetwork
from .rl import RewardModulatedReadout

# Real n8n node types (n8n-nodes-base.*). The briefs' correct answers are
# the pairing a human would pick.
TRIGGERS = ["webhook", "scheduleTrigger", "emailReadImap", "rssFeedTrigger"]
ACTIONS = ["httpRequest", "googleSheets", "readWriteFile", "slack"]

BRIEFS = [
    {"slug": "form-to-endpoint", "brief": "When a form is submitted, post it to the endpoint",
     "trigger": "webhook", "action": "httpRequest"},
    {"slug": "morning-metrics", "brief": "Every morning, append the metrics to a sheet",
     "trigger": "scheduleTrigger", "action": "googleSheets"},
    {"slug": "email-archiver", "brief": "When an email arrives, save it to disk",
     "trigger": "emailReadImap", "action": "readWriteFile"},
    {"slug": "rss-alert", "brief": "When a new RSS item appears, alert the team channel",
     "trigger": "rssFeedTrigger", "action": "slack"},
]

_NODE_TYPE = lambda name: f"n8n-nodes-base.{name}"


def episode(brain, net, brief_idx: int, steps: int, rng) -> np.ndarray:
    """Run the brain on one brief; return per-neuron spike counts (rate)."""
    stim_group = brain.sensory[f"brief_{brief_idx}"]
    net.v[:] = 0.0
    net.spikes[:] = False

    def stim(_t):
        ext = np.zeros(brain.n)
        ext[stim_group] = rng.uniform(0.6, 1.2, len(stim_group))
        return ext

    spikes = net.run(steps, stim)
    return spikes.sum(axis=0) / steps


def build_workflow(brief: dict, trigger: str, action: str) -> dict:
    """Assemble an importable n8n workflow from the fly's two choices."""
    wid = f"flyDesigned-{brief['slug']}"
    nodes = [
        {"parameters": _trigger_params(trigger, brief), "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, wid + ".t")),
         "name": "Trigger", "type": _NODE_TYPE(trigger), "typeVersion": 2 if trigger == "webhook" else 1,
         "position": [240, 300], **({"webhookId": wid} if trigger == "webhook" else {})},
        {"parameters": _action_params(action), "id": str(uuid.uuid5(uuid.NAMESPACE_DNS, wid + ".a")),
         "name": "Action", "type": _NODE_TYPE(action), "typeVersion": _ACTION_VERSIONS[action],
         "position": [460, 300]},
    ]
    return {
        "id": wid,
        "name": f"Fly-designed: {brief['brief']}",
        "nodes": nodes,
        "connections": {"Trigger": {"main": [[{"node": "Action", "type": "main", "index": 0}]]}},
        "settings": {"executionOrder": "v1"},
        "active": False,
    }


_ACTION_VERSIONS = {"httpRequest": 4.2, "googleSheets": 4.5, "readWriteFile": 1, "slack": 2.3}


def _trigger_params(trigger: str, brief: dict) -> dict:
    if trigger == "webhook":
        return {"httpMethod": "POST", "path": f"fly-{brief['slug']}", "responseMode": "lastNode", "options": {}}
    if trigger == "scheduleTrigger":
        return {"rule": {"interval": [{"field": "cronExpression", "expression": "0 8 * * *"}]}}
    return {}  # emailReadImap / rssFeedTrigger need credentials/URLs filled by a human


def _action_params(action: str) -> dict:
    if action == "httpRequest":
        return {"method": "POST", "url": "http://localhost:5680/webhook/fly-brain", "options": {}}
    if action == "readWriteFile":
        return {"operation": "write", "fileName": "fly-email.txt", "options": {}}
    return {}


def make_brain(mode: str, seed: int = 0):
    """The fixed brain the readouts sit on: one sensory group per brief."""
    brief_names = [f"brief_{i}" for i in range(len(BRIEFS))]
    if mode == "real":
        # Four distinguishable stimuli: left/right instances of the two
        # looming-detector types (instance strings are <type>_<side>).
        return load_real(token=os.environ["NEUPRINT_TOKEN"],
                         sensory_groups=dict(zip(brief_names,
                             [["LPLC2_L"], ["LPLC2_R"], ["LC4_L"], ["LC4_R"]])),
                         motor_groups={"escape_dn": "DNp01\\(GF\\)_.*"},
                         weight_scale=1000.0)
    return load_synthetic(sensory_groups=tuple(brief_names), motor_groups=("readout",), seed=seed)


def main():
    ap = argparse.ArgumentParser(description=__doc__)

    ap.add_argument("--episodes", type=int, default=20000)
    ap.add_argument("--steps", type=int, default=100, help="simulation steps per episode")
    ap.add_argument("--lr", type=float, default=1.0, help="constant; hot or decaying lr stalls this bandit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["synthetic", "real"], default=os.environ.get("FLY_MODE", "synthetic"))
    ap.add_argument("--out", default="n8n/fly-designed", help="directory for exported workflows")
    ap.add_argument("--csv", default=None, help="training log; default results/fly-training-log-<mode>.csv")
    ap.add_argument("--save", default="results/fly-readout.npz", help="trained readout weights, used by demo.py")
    args = ap.parse_args()
    args.csv = args.csv or f"results/fly-training-log-{args.mode}.csv"

    rng = np.random.default_rng(args.seed)
    brain = make_brain(args.mode, args.seed)
    net = LIFNetwork(brain.weights, tau=4.0)
    trig_ro = RewardModulatedReadout(brain.n, len(TRIGGERS), lr=args.lr, seed=args.seed)
    act_ro = RewardModulatedReadout(brain.n, len(ACTIONS), lr=args.lr, seed=args.seed + 1)

    print(f"brain: mode={args.mode} neurons={brain.n}; training {args.episodes} episodes "
          f"({len(BRIEFS)} briefs x {len(TRIGGERS)} triggers x {len(ACTIONS)} actions)")

    history = []
    with open(args.csv, "w", newline="") as fh:
        log = csv.writer(fh)
        log.writerow(["episode", "brief", "trigger", "action", "reward"])
        for ep in range(args.episodes):
            bi = ep % len(BRIEFS)
            brief = BRIEFS[bi]
            counts = episode(brain, net, bi, args.steps, rng)
            counts = counts / (np.linalg.norm(counts) + 1e-9)  # unit-norm: stabilizes lr
            eps = 0.1  # constant exploration; decaying it lets the policy freeze early

            t_choice, t_probs = trig_ro.choose(counts, eps)
            a_choice, a_probs = act_ro.choose(counts, eps)
            t_ok = TRIGGERS[t_choice] == brief["trigger"]
            a_ok = ACTIONS[a_choice] == brief["action"]
            reward = 1.0 if (t_ok and a_ok) else 0.5 if (t_ok or a_ok) else 0.0

            # separate dopamine channels: each readout is scored on its own
            # decision, not blamed for the other readout's mistake
            trig_ro.update(counts, t_choice, t_probs, 1.0 if t_ok else 0.0)
            act_ro.update(counts, a_choice, a_probs, 1.0 if a_ok else 0.0)
            history.append(reward)
            log.writerow([ep, brief["slug"], TRIGGERS[t_choice], ACTIONS[a_choice], reward])

            if (ep + 1) % 100 == 0:
                print(f"  episode {ep + 1:5d}: rolling reward (last 100) = {np.mean(history[-100:]):.2f}")

    print("\nFinal greedy designs (what the fly would build):")
    os.makedirs(args.out, exist_ok=True)
    for bi, brief in enumerate(BRIEFS):
        counts = episode(brain, net, bi, args.steps, rng)
        counts = counts / (np.linalg.norm(counts) + 1e-9)
        t = TRIGGERS[int(np.argmax(trig_ro.W @ counts))]
        a = ACTIONS[int(np.argmax(act_ro.W @ counts))]
        ok = "OK " if (t == brief["trigger"] and a == brief["action"]) else "BAD"
        print(f"  [{ok}] {brief['brief']}\n        -> {t} + {a}")
        wf = build_workflow(brief, t, a)
        path = os.path.join(args.out, f"{brief['slug']}.json")
        with open(path, "w") as fh:
            json.dump(wf, fh, indent=2)
    np.savez(args.save, trigger=trig_ro.W, action=act_ro.W, mode=args.mode, seed=args.seed)
    print(f"\nworkflows written to {args.out}/; training log at {args.csv}; readout saved to {args.save}")


if __name__ == "__main__":
    main()
