#!/usr/bin/env bash
# Records the brain viz and the fly-in-the-editor in parallel (both follow the ONE brain
# simulation served by fly_brain.demo), then stitches a side-by-side MP4.
# Requires: demo.py serving on :8765, n8n on :5680 with owner account, ffmpeg, playwright chromium.
#   scripts/make_video.sh [seconds] [out.mp4]
set -euo pipefail
cd "$(dirname "$0")/.."
SECONDS_WANTED="${1:-60}"
OUT="${2:-fly-demo.mp4}"
REC=recordings
rm -rf "$REC"; mkdir -p "$REC/brain" "$REC/editor"

# Start both recorders together so the shared brain events land at the same offset in each file.
.venv/bin/python -m fly_brain.record_brain --seconds $((SECONDS_WANTED + 15)) --out "$REC/brain" >"$REC/brain.log" 2>&1 &
BRAINREC=$!
.venv/bin/python -m fly_brain.puppet --pace 0.6 --headless --seconds $((SECONDS_WANTED + 15)) --record "$REC/editor" >"$REC/puppet.log" 2>&1 &
PUPPET=$!
wait $BRAINREC
wait $PUPPET || true

BRAIN=$(ls "$REC"/brain/*.webm | head -1)
EDITOR=$(ls "$REC"/editor/*.webm | head -1)

# Skip the first ~3s (page load / login), take SECONDS_WANTED, stack side by side.
ffmpeg -y -loglevel error \
  -ss 3 -t "$SECONDS_WANTED" -i "$BRAIN" \
  -ss 3 -t "$SECONDS_WANTED" -i "$EDITOR" \
  -filter_complex "\
    [0:v]scale=960:576:force_original_aspect_ratio=decrease,pad=960:576:(ow-iw)/2:(oh-ih)/2:color=#0b0d12[l];\
    [1:v]scale=960:576:force_original_aspect_ratio=decrease,pad=960:576:(ow-iw)/2:(oh-ih)/2:color=#0b0d12[r];\
    [l][r]hstack=inputs=2[out]" \
  -map "[out]" -r 30 -c:v libx264 -pix_fmt yuv420p -preset medium -crf 20 -movflags +faststart "$OUT"
echo "wrote $OUT"
