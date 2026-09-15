#!/usr/bin/env bash
# Records the brain viz and the fly-in-the-editor in parallel (both follow the ONE brain
# simulation served by fly_brain.demo), then cuts a 1920x1080 MP4:
#   phase 1: the brain viz full-frame while the fly takes its first brief (big captions)
#   phase 2: crossfade to side-by-side the moment the fly starts building in the editor
# Requires: demo.py serving on :8765, n8n on :5680 with N8N_EMAIL/N8N_PASSWORD, ffmpeg, playwright chromium.
#   scripts/make_video.sh [seconds] [out.mp4]
set -euo pipefail
cd "$(dirname "$0")/.."
SECONDS_WANTED="${1:-60}"
OUT="${2:-fly-demo.mp4}"
REC=recordings
rm -rf "$REC"; mkdir -p "$REC/brain" "$REC/editor"

# Start both recorders together so the shared brain events land at the same offset in each file.
.venv/bin/python -m fly_brain.record_brain --seconds $((SECONDS_WANTED + 20)) --out "$REC/brain" >"$REC/brain.log" 2>&1 &
BRAINREC=$!
.venv/bin/python -m fly_brain.puppet --pace 0.6 --headless --seconds $((SECONDS_WANTED + 20)) --record "$REC/editor" >"$REC/puppet.log" 2>&1 &
PUPPET=$!
wait $BRAINREC
wait $PUPPET || true

BRAIN=$(ls "$REC"/brain/*.webm | head -1)
EDITOR=$(ls "$REC"/editor/*.webm | head -1)
# When did the fly start its first build? (puppet.log: "[1] t=+12.3s ...")
CUT=$(sed -n 's/^\[1\] t=+\([0-9.]*\)s.*/\1/p' "$REC/puppet.log" | head -1)
CUT=${CUT:-15}
SKIP=3                                   # page load / login
FADE=0.8
PH1=$(python3 -c "print(max(1.0, $CUT - $SKIP + 1.5))")   # phase 1 length: cut as the fly starts moving

ffmpeg -y -loglevel error \
  -ss $SKIP -t "$SECONDS_WANTED" -i "$BRAIN" \
  -ss $SKIP -t "$SECONDS_WANTED" -i "$EDITOR" \
  -filter_complex "\
    [0:v]split[b1][b2];\
    [b1]scale=1800:1080,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=#0b0d12,setsar=1[full];\
    [b2]scale=960:576,setsar=1[l];\
    [1:v]scale=960:576,setsar=1[r];\
    [l][r]hstack=inputs=2,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=#0b0d12,trim=start=$PH1,setpts=PTS-STARTPTS[side];\
    [full][side]xfade=transition=fade:duration=$FADE:offset=$PH1[out]" \
  -map "[out]" -r 30 -c:v libx264 -pix_fmt yuv420p -preset medium -crf 20 -movflags +faststart "$OUT"
echo "wrote $OUT (cut to side-by-side at ${PH1}s)"
