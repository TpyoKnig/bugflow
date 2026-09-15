"""Live visual demo: watch the fly brain design n8n workflows by itself.

Serves a single page that renders the brain (real 3D soma positions in
real mode) flashing as neurons spike, next to an n8n "canvas" where the
trained readouts' votes fill in, the workflow assembles, and n8n runs it.

    .venv/bin/python -m fly_brain.designer --mode real      # once: trains + saves results/fly-readout.npz
    .venv/bin/python -m fly_brain.demo                       # then open http://localhost:8765

# ponytail: stdlib http.server + Server-Sent Events, one simulation per
# connected browser tab. Fine for a screen recording; not a multi-user app.
"""

import argparse
import json
import os
import queue
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from .designer import ACTIONS, BRIEFS, TRIGGERS, build_workflow, make_brain
from .lif import LIFNetwork

HTML = os.path.join(os.path.dirname(__file__), "demo.html")


def layout(brain) -> np.ndarray:
    """(n, 3) positions: real soma coordinates, or a layered layout for synthetic."""
    if brain.positions is not None:
        p = brain.positions - brain.positions.mean(axis=0)
        return p / np.abs(p).max()
    rng = np.random.default_rng(0)
    p = rng.normal(size=(brain.n, 3)) * 0.25
    layer = np.zeros(brain.n)
    for idx in brain.sensory.values():
        layer[idx] = -1.0
    for idx in brain.motor.values():
        layer[idx] = 1.0
    p[:, 0] += layer
    return p / np.abs(p).max()


def post_json(url: str, payload: dict, timeout: float = 5.0):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode()[:200]
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:200]
    except Exception as e:  # connection refused etc.
        return 0, str(e)[:200]


class Demo:
    def __init__(self, readout_path: str, n8n_url: str, steps: int, fps: float):
        w = np.load(readout_path, allow_pickle=False)
        self.mode = str(w["mode"])
        self.W_trig, self.W_act = w["trigger"], w["action"]
        self.brain = make_brain(self.mode, int(w["seed"]))
        self.positions = layout(self.brain)
        self.n8n_url, self.steps, self.dt = n8n_url.rstrip("/"), steps, 1.0 / fps
        print(f"demo: mode={self.mode} neurons={self.brain.n} readout={readout_path}")

    def init_event(self) -> dict:
        return {
            "type": "init", "mode": self.mode, "n": self.brain.n,
            "positions": np.round(self.positions, 3).tolist(),
            "sensory": {g: idx.tolist() for g, idx in self.brain.sensory.items()},
            "motor": {g: idx.tolist() for g, idx in self.brain.motor.items()},
            "triggers": TRIGGERS, "actions": ACTIONS,
            "briefs": [b["brief"] for b in BRIEFS],
        }

    def episodes(self):
        """Generator of events, forever."""
        net = LIFNetwork(self.brain.weights, tau=4.0)
        rng = np.random.default_rng()
        yield self.init_event()
        ep = 0
        while True:
            bi = ep % len(BRIEFS)
            brief = BRIEFS[bi]
            yield {"type": "brief", "index": bi, "text": brief["brief"], "episode": ep}
            stim_idx = self.brain.sensory[f"brief_{bi}"]
            net.v[:] = 0.0
            net.spikes[:] = False
            counts = np.zeros(self.brain.n)
            for t in range(self.steps):
                ext = np.zeros(self.brain.n)
                ext[stim_idx] = rng.uniform(0.6, 1.2, len(stim_idx))
                spikes = net.step(ext)
                counts += spikes
                yield {"type": "spikes", "t": t, "idx": np.flatnonzero(spikes).tolist()}
                if t % 10 == 9:
                    yield {"type": "votes", **self._votes(counts / (t + 1))}
                time.sleep(self.dt)
            votes = self._votes(counts / self.steps)
            t_i, a_i = int(np.argmax(votes["trigger"])), int(np.argmax(votes["action"]))
            trig, act = TRIGGERS[t_i], ACTIONS[a_i]
            ok = trig == brief["trigger"] and act == brief["action"]
            yield {"type": "choice", "trigger": trig, "action": act, "correct": ok,
                   "workflow": build_workflow(brief, trig, act)}
            time.sleep(0.8)
            yield {"type": "n8n", **self._execute(brief, trig)}
            if self.puppet_connected:            # let the fly finish building it in the editor
                if not self.ack.wait(timeout=120):
                    self.puppet_connected = False   # puppet went away; stop waiting for it
                self.ack.clear()
            time.sleep(2.5)
            ep += 1

    # --- broadcast: one simulation, every connected page/puppet sees the same events ---
    puppet_connected = False
    ack = threading.Event()
    clients: list = []
    lock = threading.Lock()

    def start(self):
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        for ev in self.episodes():
            if ev["type"] == "init":
                continue
            self.broadcast(ev)

    def broadcast(self, ev: dict):
        with self.lock:
            for q in self.clients:
                q.put(ev)

    def subscribe(self) -> "queue.Queue":
        q = queue.Queue()
        q.put(self.init_event())
        with self.lock:
            self.clients.append(q)
        return q

    def unsubscribe(self, q):
        with self.lock:
            self.clients.remove(q)

    def _votes(self, rates: np.ndarray) -> dict:
        c = rates / (np.linalg.norm(rates) + 1e-9)

        def softmax(s):
            e = np.exp(s - s.max())
            return (e / e.sum()).round(3).tolist()
        return {"trigger": softmax(self.W_trig @ c), "action": softmax(self.W_act @ c)}

    def _execute(self, brief: dict, trigger: str) -> dict:
        if trigger != "webhook":
            return {"status": "designed", "detail": f"{trigger} needs credentials; workflow JSON exported"}
        url = f"{self.n8n_url}/webhook/fly-{brief['slug']}"
        code, body = post_json(url, {"source": "fly-brain-demo", "brief": brief["slug"]})
        return {"status": "executed" if code == 200 else "failed", "http": code, "detail": body, "url": url}


class Handler(BaseHTTPRequestHandler):
    demo: "Demo"  # set in main

    def log_message(self, *_):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if self.path == "/ack":                    # puppet finished building in the editor
            self.demo.ack.set()
        elif self.path == "/puppet":               # puppet attached: server now waits for acks
            self.demo.puppet_connected = True
        elif self.path == "/status":               # puppet narrates what it is doing
            self.demo.broadcast({"type": "puppet", "text": body.decode()})
        self.send_response(204)
        self.end_headers()

    def do_GET(self):
        if self.path == "/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            q = self.demo.subscribe()
            try:
                while True:
                    ev = q.get()
                    self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
            finally:
                self.demo.unsubscribe(q)
        else:
            with open(HTML, "rb") as fh:
                body = fh.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--readout", default="results/fly-readout.npz")
    ap.add_argument("--n8n-url", default=os.environ.get("N8N_URL", "http://localhost:5680"))
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--fps", type=float, default=30.0, help="simulation steps per second")
    args = ap.parse_args()
    Handler.demo = Demo(args.readout, args.n8n_url, args.steps, args.fps)
    Handler.demo.start()
    print(f"open http://localhost:{args.port}")
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
