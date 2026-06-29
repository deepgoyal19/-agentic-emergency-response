@echo off
REM Launch Rescue City in STABLE mode. A normal double-click open of the .wbt crashes
REM this machine's Webots on load (some GUI element); --batch suppresses that and the
REM simulation + 3D view run fine for recording. Press Ctrl+C in this window to stop.
cd /d %~dp0
echo Launching Rescue City (stable --batch mode)...
echo Leave this window open. Close the Webots window when done.
"E:\Webots\msys64\mingw64\bin\webots.exe" --batch --mode=realtime worlds\rescue_city_mesh.wbt
