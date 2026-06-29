#!/bin/bash
cd /f/Hackathon/code/webots
WB="/e/Webots/msys64/mingw64/bin/webots.exe"; WORLD="F:/Hackathon/code/webots/worlds/rescue_city_mesh.wbt"
kw(){ powershell -Command "Stop-Process -Name webots,webots-bin -Force -EA SilentlyContinue" >/dev/null 2>&1; }
for inc in "$@"; do
  echo "== $inc =="; kw; sleep 2
  echo "{\"incident\":\"$inc\",\"live_view\":false}" > mission.json
  rm -f mission_log.json mission_error.log "frames/frame_$inc.png"
  GEMMA_MOCK=1 "$WB" --batch --mode=fast --minimize --stdout --stderr "$WORLD" >/dev/null 2>&1 &
  for i in $(seq 1 30); do sleep 4; [ -f "frames/frame_$inc.png" ] && [ -s "frames/frame_$inc.png" ] && { echo "$inc OK ($(stat -c%s frames/frame_$inc.png)b)"; sleep 2; break; }; [ -f mission_error.log ] && { cat mission_error.log; break; }; done
  kw; sleep 2
done
