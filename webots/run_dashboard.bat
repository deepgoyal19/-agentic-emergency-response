@echo off
REM Live mission dashboard for Rescue City (Gemma 4 on Cerebras).
REM Uses serve.py (dual-stack: BOTH localhost and 127.0.0.1 work, threaded, no-cache).
cd /d %~dp0
python serve.py
