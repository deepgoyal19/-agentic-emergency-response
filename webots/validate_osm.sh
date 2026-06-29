#!/bin/bash
# Validate the OSM (Morges) hero world for the given incidents (default: fire stalker).
cd /f/Hackathon/code/webots
WEBOTS="/e/Webots/msys64/mingw64/bin/webots.exe"
WORLD="F:/Hackathon/code/webots/worlds/rescue_city_osm.wbt"
INC="${@:-fire stalker}"
kw(){ powershell -Command "Stop-Process -Name webots,webots-bin -Force -ErrorAction SilentlyContinue" >/dev/null 2>&1; }
done_check(){ python -c "import json,sys;d=json.load(open('mission_log.json'));sys.exit(0 if any('COMPLETE' in p['msg'] for p in d.get('phases',[])) else 1)" 2>/dev/null; }
for inc in $INC; do
  echo "===== $inc ====="; kw; sleep 2
  echo "{\"incident\": \"$inc\", \"live_view\": false}" > mission.json
  rm -f mission_log.json mission_error.log "frames/frame_$inc.png"
  "$WEBOTS" --batch --mode=fast --minimize --stdout --stderr "$WORLD" >/dev/null 2>&1 &
  for i in $(seq 1 55); do
    sleep 4
    [ -f mission_error.log ] && { echo "$inc ERROR"; cat mission_error.log; break; }
    if [ -f mission_log.json ] && done_check; then echo "$inc COMPLETE ~$((i*4))s"; break; fi
  done
  [ -f "frames/frame_$inc.png" ] && echo "$inc frame OK ($(stat -c%s frames/frame_$inc.png) b)" || echo "$inc NO FRAME"
  kw; sleep 2
done
echo "DONE"