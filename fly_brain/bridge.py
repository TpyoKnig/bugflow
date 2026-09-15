"""Simulate a connectome-driven LIF network and POST motor bursts to n8n.

Usage (synthetic network, no credentials):
    N8N_WEBHOOK_URL=http://localhost:5678/webhook/fly-brain \\
        python3 -m fly_brain.bridge --steps 400

Usage (real MaleCNS v1.0 subgraph via neuPrint):
    NEUPRINT_TOKEN=... N8N_WEBHOOK_URL=... python3 -m fly_brain.bridge --mode real \\
        --sensory-groups '{"looming": "LPLC2_.*"}' \\
        --motor-groups   '{"escape_dn": "DNp01\\(GF\\)_.*"}'

Every `--beat-every` steps one sensory group gets a pulse (standing in for a
beat in Beat Saber / a frame in Doom / a block in Minecraft). Whenever a motor
group's spike rate over the trailing `--window` steps reaches
`--rate-threshold` (and its `--cooldown` has elapsed), an event is POSTed to
the webhook as JSON:

    {"event": "motor_burst", "neuron_group": "escape_dn",
     "spike_rate": 0.2, "step": 122, "mode": "synthetic"}

`--steps 0` runs forever (for a long-running container); `--step-delay`
paces the loop in seconds per step. Every environment variable has a flag
of the same meaning; flags win.

Exit codes: 0 finished, 2 bad arguments. Webhook failures are logged to
stderr and never abort the run.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

import numpy as np

from fly_brain.connectome import load_real, load_synthetic
from fly_brain.lif import LIFNetwork


def log(msg: str, *, err: bool = False) -> None:
    print(msg, file=sys.stderr if err else sys.stdout, flush=True)


def post_event(url: str, payload: dict, timeout: float = 5.0) -> bool:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            log(f"  -> n8n {resp.status}: {resp.read().decode('utf-8', 'replace')[:200]}")
        return True
    except urllib.error.HTTPError as exc:
        # 404 here almost always means the workflow isn't published/active.
        log(f"  -> webhook HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:200]}", err=True)
    except OSError as exc:  # URLError, ConnectionRefused, TimeoutError, ...
        log(f"  -> webhook POST failed: {exc}", err=True)
    return False


def _env_float(name, default):
    return float(os.environ.get(name, default))


def _env_int(name, default):
    return int(os.environ.get(name, default))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["synthetic", "real"], default=os.environ.get("FLY_MODE", "synthetic"))
    ap.add_argument("--steps", type=int, default=_env_int("FLY_STEPS", 400), help="0 = run forever")
    ap.add_argument("--step-delay", type=float, default=_env_float("FLY_STEP_DELAY", 0.0),
                    help="seconds to sleep per step (pace a long-running container)")
    ap.add_argument("--beat-every", type=int, default=_env_int("FLY_BEAT_EVERY", 12),
                    help="inject a sensory pulse every N steps")
    ap.add_argument("--window", type=int, default=_env_int("FLY_WINDOW", 5),
                    help="trailing steps used for each motor group's spike rate")
    ap.add_argument("--rate-threshold", type=float, default=_env_float("FLY_RATE_THRESHOLD", 0.15))
    ap.add_argument("--cooldown", type=int, default=_env_int("FLY_COOLDOWN", 10),
                    help="min steps between events for the same motor group")
    ap.add_argument("--tau", type=float, default=_env_float("FLY_TAU", 4.0), help="LIF membrane time constant (steps)")
    ap.add_argument("--v-thresh", type=float, default=_env_float("FLY_V_THRESH", 1.0), help="LIF spike threshold")
    ap.add_argument("--webhook-url", default=os.environ.get("N8N_WEBHOOK_URL"))
    ap.add_argument("--seed", type=int, default=_env_int("FLY_SEED", 0))
    # real mode only
    ap.add_argument("--token", default=os.environ.get("NEUPRINT_TOKEN"))
    ap.add_argument("--server", default=os.environ.get("NEUPRINT_SERVER", "https://neuprint.janelia.org"))
    ap.add_argument("--dataset", default=os.environ.get("NEUPRINT_DATASET", "male-cns:v1.0"))
    ap.add_argument("--sensory-groups", default=os.environ.get("FLY_SENSORY_GROUPS"),
                    help='JSON dict of group -> instance regex, e.g. \'{"looming": "LPLC2_.*"}\'')
    ap.add_argument("--motor-groups", default=os.environ.get("FLY_MOTOR_GROUPS"),
                    help='JSON dict of group -> instance regex, e.g. \'{"escape_dn": "DNp01\\(GF\\)_.*"}\'')
    ap.add_argument("--min-weight", type=int, default=_env_int("FLY_MIN_WEIGHT", 3),
                    help="drop connections with fewer synapses than this")
    ap.add_argument("--weight-scale", type=float, default=_env_float("FLY_WEIGHT_SCALE", 20.0),
                    help="synapse count divisor; smaller = stronger coupling")
    return ap


def main(argv=None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)

    if not args.webhook_url:
        log("warning: no --webhook-url / N8N_WEBHOOK_URL set, events will only be printed", err=True)

    if args.mode == "synthetic":
        graph = load_synthetic(seed=args.seed)
    else:
        if not args.token:
            ap.error("--mode real requires --token or NEUPRINT_TOKEN")
        if not (args.sensory_groups and args.motor_groups):
            ap.error("--mode real requires --sensory-groups and --motor-groups (JSON dicts)")
        graph = load_real(
            token=args.token,
            sensory_groups=json.loads(args.sensory_groups),
            motor_groups=json.loads(args.motor_groups),
            server=args.server,
            dataset=args.dataset,
            min_total_weight=args.min_weight,
            weight_scale=args.weight_scale,
        )

    net = LIFNetwork(graph.weights, tau=args.tau, v_thresh=args.v_thresh)
    rng = np.random.default_rng(args.seed)
    sensory_groups = [(g, idx) for g, idx in graph.sensory.items() if idx.size]
    motor_groups = {g: idx for g, idx in graph.motor.items() if idx.size}
    if not sensory_groups or not motor_groups:
        log("error: need at least one non-empty sensory group and one non-empty motor group", err=True)
        return 2

    history = np.zeros((args.window, net.n), dtype=bool)
    last_fired = {g: -10**9 for g in motor_groups}
    total_spikes = {g: 0 for g in motor_groups}
    events_sent = {g: 0 for g in motor_groups}

    def stim(t):
        ext = np.zeros(net.n)
        if t % args.beat_every == 0:
            _, idx = sensory_groups[(t // args.beat_every) % len(sensory_groups)]
            ext[idx] = rng.uniform(0.6, 1.2, size=len(idx))
        return ext

    log(f"mode={args.mode} neurons={net.n} sensory_groups={[g for g, _ in sensory_groups]} "
        f"motor_groups={list(motor_groups)} steps={'inf' if args.steps == 0 else args.steps} "
        f"webhook={args.webhook_url or '(none)'}")

    t = 0
    try:
        while args.steps == 0 or t < args.steps:
            spikes = net.step(stim(t))
            history[t % args.window] = spikes

            for group_name, idx in motor_groups.items():
                total_spikes[group_name] += int(spikes[idx].sum())
                rate = history[:, idx].mean()
                if rate >= args.rate_threshold and (t - last_fired[group_name]) >= args.cooldown:
                    last_fired[group_name] = t
                    events_sent[group_name] += 1
                    payload = {
                        "event": "motor_burst",
                        "neuron_group": group_name,
                        "spike_rate": round(float(rate), 3),
                        "step": t,
                        "mode": args.mode,
                    }
                    log(f"[step {t:5d}] {group_name} spike_rate={rate:.2f} -> firing webhook")
                    if args.webhook_url:
                        post_event(args.webhook_url, payload)

            t += 1
            if args.step_delay:
                time.sleep(args.step_delay)
    except KeyboardInterrupt:
        log("interrupted")

    log(f"done after {t} steps. per motor group: " + ", ".join(
        f"{g}: spikes={total_spikes[g]} events={events_sent[g]}" for g in motor_groups))
    if sum(events_sent.values()) == 0:
        log("no events fired: network is silent for these parameters. See README 'Calibrating real mode'.", err=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
