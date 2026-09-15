"""A fly builds workflows inside the real n8n editor.

Opens a visible Chromium on the n8n editor. For each brief the trained
fly brain (same readouts as demo.py) picks a trigger and an action; an
animated fly then flies to the "+" button, opens the node picker, types
the node name, clicks it, wires the action node, renames the workflow to
the brief, and saves. Every click is a real click on the real n8n UI.

    .venv/bin/pip install playwright && .venv/bin/python -m playwright install chromium
    NEUPRINT_TOKEN=... .venv/bin/python -m fly_brain.puppet

Requires: n8n on --n8n-url with an owner account (--email/--password),
and a trained results/fly-readout.npz (fly_brain.designer).

# ponytail: relies on n8n's data-test-id hooks (verified on n8n 2.13.4);
# a major editor redesign will need the selectors below updated.
"""

import argparse
import json
import os
import time
import urllib.request

from playwright.sync_api import sync_playwright

from .designer import BRIEFS

DISPLAY = {  # n8n node-creator search text per node type
    "webhook": "Webhook", "scheduleTrigger": "Schedule Trigger",
    "emailReadImap": "Email Trigger (IMAP)", "rssFeedTrigger": "RSS Feed Trigger",
    "httpRequest": "HTTP Request", "googleSheets": "Google Sheets",
    "readWriteFile": "Read/Write Files from Disk", "slack": "Slack",
}

FLY_OVERLAY = r"""
(() => {
  if (window.__fly) return;
  const st = document.createElement('style');
  st.textContent = `
    #__fly { position:fixed; left:60px; top:60px; z-index:999999; pointer-events:none;
             font-size:54px; line-height:1; filter:drop-shadow(0 6px 6px rgba(0,0,0,.35));
             transition:left .9s cubic-bezier(.3,.8,.3,1), top .9s cubic-bezier(.3,.8,.3,1);
             animation:__buzz .12s infinite alternate; transform-origin:50% 60%; }
    @keyframes __buzz { from { transform:rotate(-6deg) translateY(0) } to { transform:rotate(6deg) translateY(-2px) } }
    #__say { position:fixed; z-index:999999; pointer-events:none; max-width:360px; background:#111; color:#fff;
             font:15px/1.35 ui-sans-serif,system-ui; padding:10px 14px; border-radius:14px; opacity:0;
             transition:opacity .3s, left .9s, top .9s; box-shadow:0 8px 24px rgba(0,0,0,.3); }
    #__say::after { content:''; position:absolute; left:18px; bottom:-8px; border:8px solid transparent; border-top-color:#111; border-bottom:0; }
    #__ping { position:fixed; z-index:999998; pointer-events:none; width:18px; height:18px; border-radius:50%;
              border:3px solid #ff6d5a; opacity:0; transform:translate(-50%,-50%) scale(.4); }
    @keyframes __pingA { from { opacity:1; transform:translate(-50%,-50%) scale(.4) } to { opacity:0; transform:translate(-50%,-50%) scale(2.6) } }
    #__banner { position:fixed; left:50%; top:14px; transform:translateX(-50%); z-index:999999; pointer-events:none;
                background:#111; color:#fff; font:800 30px/1.1 ui-sans-serif,system-ui; padding:14px 26px; border-radius:16px;
                box-shadow:0 10px 30px rgba(0,0,0,.35); white-space:nowrap; max-width:90vw; overflow:hidden; text-overflow:ellipsis; }
  `;
  document.documentElement.appendChild(st);
  const fly = document.createElement('div'); fly.id = '__fly'; fly.textContent = '\u{1FAB0}';
  const say = document.createElement('div'); say.id = '__say';
    const ping = document.createElement('div'); ping.id = '__ping';
    const banner = document.createElement('div'); banner.id = '__banner'; banner.textContent = '\u{1FAB0} waiting for the brain\u2026';
    document.documentElement.append(fly, say, ping, banner);
  window.__fly = {
    moveTo(x, y) { fly.style.left = (x - 27) + 'px'; fly.style.top = (y - 44) + 'px';
                   say.style.left = (x - 10) + 'px'; say.style.top = (y - 60 - say.offsetHeight - 20) + 'px'; },
    say(t) { say.textContent = t; say.style.opacity = t ? 1 : 0; if (t) banner.textContent = '\u{1FAB0} ' + t; },
    click(x, y) { ping.style.left = x + 'px'; ping.style.top = y + 'px';
                  ping.style.animation = 'none'; void ping.offsetWidth; ping.style.animation = '__pingA .6s ease-out'; },
  };
})();
"""


class FlyHand:
    """Moves the fly to a target, then really clicks it."""

    def __init__(self, page, pace: float):
        self.page, self.pace = page, pace

    def _ensure(self):
        self.page.evaluate(FLY_OVERLAY)

    def say(self, text: str):
        self._ensure()
        self.page.evaluate("t => window.__fly.say(t)", text)

    def click(self, selector: str, force: bool = False):
        self._ensure()
        loc = self.page.locator(selector).first
        loc.wait_for(state="visible", timeout=15000)
        box = loc.bounding_box()
        x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
        self.page.evaluate("([x, y]) => window.__fly.moveTo(x, y)", [x, y])
        time.sleep(0.95 * self.pace)                       # fly in
        self.page.evaluate("([x, y]) => window.__fly.click(x, y)", [x, y])
        loc.click(force=force)
        time.sleep(0.5 * self.pace)

    def type(self, text: str):
        self.page.keyboard.type(text, delay=int(60 * self.pace))
        time.sleep(0.6 * self.pace)


def add_node(hand: FlyHand, page, opener: str, node_type: str):
    hand.click(opener, force=True)
    hand.type(DISPLAY[node_type])
    hand.click('[data-test-id="node-creator-item-name"]')
    for _ in range(2):  # app nodes (Sheets, Slack) open an actions sub-list: take the first
        time.sleep(0.6)
        if page.locator('[data-test-id="node-creator-search-bar"]').count():
            hand.click('[data-test-id="node-creator-item-name"]')
        else:
            break
    time.sleep(0.8)
    if page.locator('[data-test-id="ndv-close-button"]').count():   # close the node's parameter panel
        hand.click('[data-test-id="ndv-close-button"]')
    page.wait_for_selector('[data-test-id="node-parameters"]', state="detached", timeout=10000)
    time.sleep(0.4 * hand.pace)


def build_in_editor(hand: FlyHand, page, base: str, brief: dict, trigger: str, action: str, narrate):
    if not page.url.endswith("/workflow/new"):      # first build starts on the canvas we opened at startup
        page.goto(f"{base}/workflow/new")
    page.wait_for_selector('[data-test-id="canvas-add-button"]')
    hand._ensure()
    page.evaluate("() => window.__fly.moveTo(120, 120)")
    hand.say(f"brief: {brief['brief']}")
    narrate("fly is in the n8n editor, opening a new workflow…")
    time.sleep(1.5 * hand.pace)

    hand.say(f"trigger: {DISPLAY[trigger]}")
    narrate(f"fly adds trigger node: {DISPLAY[trigger]}")
    add_node(hand, page, '[data-test-id="canvas-add-button"]', trigger)

    hand.say(f"action: {DISPLAY[action]}")
    narrate(f"fly adds action node: {DISPLAY[action]}")
    add_node(hand, page, '[data-test-id="canvas-handle-plus"]', action)

    hand.say("naming it…")
    narrate("fly names the workflow after the brief")
    hand.click('[data-test-id="workflow-name-input"]')
    page.keyboard.press("Meta+a")
    hand.type(f"Fly: {brief['brief']}")
    page.keyboard.press("Enter")
    time.sleep(0.5)

    hand.say("saving")
    page.keyboard.press("Meta+s")
    time.sleep(1.2)
    ok = trigger == brief["trigger"] and action == brief["action"]
    hand.say("✓ saved. " + ("this matches the brief" if ok else "hmm, not what the brief asked"))
    narrate("✓ fly saved the workflow in the n8n editor")
    time.sleep(3.0 * hand.pace)


def sse_events(resp):
    """Yield parsed events from an open Server-Sent Events response."""
    with resp:
        for raw in resp:
            line = raw.decode().rstrip("\n")
            if line.startswith("data: "):
                yield json.loads(line[6:])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demo-url", default=os.environ.get("DEMO_URL", "http://localhost:8765"),
                    help="the running fly_brain.demo server; this is the brain we follow")
    ap.add_argument("--n8n-url", default=os.environ.get("N8N_URL", "http://localhost:5680"))
    ap.add_argument("--email", default=os.environ.get("N8N_EMAIL"), help="n8n owner login (or N8N_EMAIL)")
    ap.add_argument("--password", default=os.environ.get("N8N_PASSWORD"), help="n8n owner password (or N8N_PASSWORD)")
    ap.add_argument("--pace", type=float, default=1.0, help="1.0 = watchable; 0.3 = fast")
    ap.add_argument("--loops", type=int, default=0, help="briefs to build; 0 = forever")
    ap.add_argument("--record", default=None, help="directory to save a .webm screen recording of the editor")
    ap.add_argument("--headless", action="store_true", help="no visible window (use with --record)")
    ap.add_argument("--seconds", type=float, default=0, help="stop after this many seconds (for recording); 0 = never")
    args = ap.parse_args()
    if not (args.email and args.password):
        ap.error("n8n login required: --email/--password or N8N_EMAIL/N8N_PASSWORD")

    base = args.n8n_url.rstrip("/")
    demo_url = args.demo_url.rstrip("/")

    def post(path, text=""):
        req = urllib.request.Request(f"{demo_url}{path}", data=text.encode(), method="POST")
        urllib.request.urlopen(req, timeout=5).close()

    stream = urllib.request.urlopen(f"{demo_url}/events")   # subscribe first ...
    post("/puppet")                                          # ... then tell the brain to wait for our acks

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=args.headless, args=["--window-size=1500,950"])
        ctx = browser.new_context(viewport={"width": 1500, "height": 900},
                                  record_video_dir=args.record,
                                  record_video_size={"width": 1500, "height": 900} if args.record else None)
        r = ctx.request.post(f"{base}/rest/login", data={"emailOrLdapLoginId": args.email, "password": args.password})
        assert r.status == 200, f"n8n login failed: HTTP {r.status} {r.text()[:200]}"
        page = ctx.new_page()
        t0 = time.time()                             # recording starts with the page; t=+... in the log is relative to it
        page.add_init_script(FLY_OVERLAY)
        hand = FlyHand(page, args.pace)
        page.goto(f"{base}/workflow/new")            # be on screen before the brain's first decision
        page.wait_for_selector('[data-test-id="canvas-add-button"]')
        hand._ensure()
        page.evaluate("() => window.__fly.moveTo(120, 120)")
        hand.say("waiting for the brain…")

        built, current_brief, last_choice = 0, None, None
        try:
            for ev in sse_events(stream):
                if args.seconds and time.time() - t0 > args.seconds:
                    break
                if ev["type"] == "brief":
                    current_brief, last_choice = BRIEFS[ev["index"]], None
                elif ev["type"] == "choice":
                    last_choice = ev
                elif ev["type"] == "n8n":        # brain decided and n8n executed: the fly's turn
                    if current_brief is None or last_choice is None:
                        post("/ack")             # joined mid-episode: release the brain, wait for the next brief
                        continue
                    c = last_choice
                    print(f"[{built + 1}] t=+{time.time() - t0:.1f}s {current_brief['brief']!r} -> {c['trigger']} + {c['action']} "
                          f"({'correct' if c['correct'] else 'wrong'})", flush=True)
                    build_in_editor(hand, page, base, current_brief, c["trigger"], c["action"],
                                    lambda t: post("/status", t))
                    post("/ack")
                    built += 1
                    if args.loops and built >= args.loops:
                        break
        except KeyboardInterrupt:
            pass
        finally:
            ctx.close()   # flushes the recording
            browser.close()


if __name__ == "__main__":
    main()
