#!/bin/bash
cd /f/Hackathon/code/webots
WEBOTS="/e/Webots/msys64/mingw64/bin/webots.exe"
WORLD="F:/Hackathon/code/webots/worlds/rescue_city.wbt"
kill_webots(){ powershell -Command "Stop-Process -Name webots,webots-bin -Force -ErrorAction SilentlyContinue" >/dev/null 2>&1; }
check_complete(){ python -c "import json,sys; d=json.load(open('mission_log.json')); sys.exit(0 if any('COMPLETE' in p['msg'] for p in d.get('phases',[])) else 1)" 2>/dev/null; }

for inc in accident fire stalker; do
  echo "===== $inc ====="
  kill_webots; sleep 2
  echo "{\"incident\": \"$inc\", \"live_view\": false}" > mission.json
  rm -f mission_log.json mission_error.log "frames/frame_$inc.png"
  "$WEBOTS" --batch --mode=fast --minimize --stdout --stderr "$WORLD" > "run_$inc.log" 2>&1 &
  done=0
  for i in $(seq 1 35); do
    sleep 4
    if [ -f mission_error.log ]; then echo "$inc: CONTROLLER ERROR"; cat mission_error.log; done=1; break; fi
    if [ -f mission_log.json ] && check_complete; then echo "$inc: COMPLETE after ~$((i*4))s"; done=1; break; fi
  done
  if [ $done -eq 0 ]; then echo "$inc: TIMEOUT — webots/world log tail:"; tail -25 "run_$inc.log"; fi
  sleep 1
  if [ -f "frames/frame_$inc.png" ]; then echo "$inc: frame OK ($(stat -c%s frames/frame_$inc.png) bytes)"; else echo "$inc: NO FRAME"; fi
  kill_webots; sleep 2
done
echo "===== DONE ====="
ls -la frames/*.png 2>/dev/null
