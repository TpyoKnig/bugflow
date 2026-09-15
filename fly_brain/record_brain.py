"""Record the brain visualization (demo.py page) to a .webm for --seconds.

    .venv/bin/python -m fly_brain.record_brain --seconds 70 --out recordings/brain
"""

import argparse
import time

from playwright.sync_api import sync_playwright


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://localhost:8765")
    ap.add_argument("--seconds", type=float, default=70)
    ap.add_argument("--out", default="recordings/brain")
    args = ap.parse_args()
    size = {"width": 1500, "height": 900}
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        ctx = b.new_context(viewport=size, record_video_dir=args.out, record_video_size=size)
        page = ctx.new_page()
        page.goto(args.url)
        time.sleep(args.seconds)
        ctx.close()
        b.close()
    print(f"saved to {args.out}/")


if __name__ == "__main__":
    main()
