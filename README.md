# Agentic Emergency Response System

> A smart-city emergency response simulation where **7 autonomous drones** are driven by **8 specialized Gemma 4 agents running on Cerebras** — reasoning, in real time, over live camera feeds to resolve four emergencies at once. Nothing is hard-coded.

![Gemma 4](https://img.shields.io/badge/Gemma_4-gemma--4--31b-orange)
![Cerebras](https://img.shields.io/badge/Inference-Cerebras-red)
![Webots](https://img.shields.io/badge/Webots-R2025a-blue)
![Python](https://img.shields.io/badge/Python-3.10%2B-green)

*Built for the Cerebras × Google DeepMind **Gemma 4 Hackathon** — Track 1.*

---

## What it is

Emergency dispatch today runs on rigid, pre-written playbooks. We asked: **what if a city's first responders could *reason* about each situation as it unfolds** — like a human commander, but in seconds?

So we built a 3D smart-city in [Webots](https://cyberbotics.com/) where 7 drones rest on charging towers across the skyline. When emergencies are reported, a team of **8 Gemma 4 agents** reasons over the drones' real camera images to decide *who to send, how low to fly, when to escalate, and which responder fits the threat* — all live, all from what the drones actually see.

A live web dashboard visualizes every agent firing, its Cerebras latency, the drone POV, the frames Gemma analyzed, a 2D city map, and the mission log.

---

## The agent team

Eight specialized agents, each a separate `gemma-4-31b` call on Cerebras with a strict structured-output schema:

| Agent | Responsibility |
|-------|----------------|
| **Dispatcher** | Classifies the 911 call (type & severity) and decides the initial response |
| **Path Planner** | Routes the nearest drone from its tower using live GPS |
| **Perception** | Reads the live drone camera — decides *how low* to descend and what it sees |
| **Surveillance** | Tracks moving subjects and confirms scene state (e.g. suspect handed off to police) |
| **Analyst** | Assesses scene severity from the aerial vision frames |
| **Coordinator** | Decides the **right responder type** and whether to escalate / call backup |
| **Executor** | Carries out the action — deliver aid, broadcast a warning, suppress fire |
| **Fleet Manager** | Battery-aware return-to-tower and recharge management |

```
911 call ─▶ Dispatcher ─▶ [launch nearest tower] ─▶ Path Planner (live GPS)
        ─▶ fly ─▶ Perception ─▶ Analyst (vision) ─▶ Coordinator ─▶ Executor
        ─▶ Surveillance ─▶ Fleet Manager (battery) ─▶ return & recharge
```

---

## Four emergencies, handled at once

| | Scenario | What Gemma decides, live |
|--|----------|--------------------------|
| 🚑 | **Traffic accident** | Perception chooses *how low* to descend (~3.5 m, safely above the wreck); Executor drops the first-aid kit right beside the casualty |
| 🚶‍♀️ | **Stalker** | The drone tracks the *follower* (not the caller) with two cameras, broadcasts a warning, and confirms he retreats before the scene is cleared |
| 🔥 | **High-rise fire** | The lead drone sees the blaze is too big and the Coordinator calls in **2 more fire units on demand**; the squad splits the tower by floor |
| 🔫 | **Armed suspect** | A patrol drone spots the rifle and the Coordinator dispatches a **police car** (not more drones), feeding live coordinates until the handoff is confirmed |

**Why it's different:** *nothing is hard-coded.* Which unit to send, how low to fly, when to escalate — every call is made live by Gemma 4 from the camera feed. Roughly **40+ agent decisions per cycle**, resolved in **~6 seconds of Cerebras compute**.

---

## Tech stack

- **Gemma 4 (`gemma-4-31b`)** — multimodal reasoning over drone camera frames, structured JSON outputs
- **Cerebras Inference** — the speed that makes 40+ live agent calls per cycle practical
- **Webots R2025a** — 3D robotics simulation (Mavic 2 Pro drones, real offscreen cameras, kinematic flight)
- **Python** — orchestrator, the 8 agents, and the embodied Webots controllers
- **Web dashboard** — a dual-stack HTTP server with live POV, the 4 Gemma views, a 2D map, the agent pipeline, and Run / Reset / Stop controls

---

## Setup & run

**Prerequisites:** Python 3.10+, [Webots R2025a](https://cyberbotics.com/), and a [Cerebras API key](https://cloud.cerebras.ai/).

```bash
# 1. install dependencies
pip install -r requirements.txt

# 2. add your Cerebras key
cp .env.example .env        # then edit .env:  CEREBRAS_API_KEY=csk-...

# 3. (if Webots isn't on the default path) point the server at it
#    PowerShell:  $env:WEBOTS_EXE = "C:\Program Files\Webots\msys64\mingw64\bin\webots.exe"

# 4. launch the dashboard + simulation server
cd webots
python serve.py             # or double-click webots/run_dashboard.bat
```

Then open **http://localhost:{Port Number}/dashboard/** and click **Run**. Webots launches headless, the drones execute the four emergencies, and the dashboard streams everything live. Use **Reset** to start a clean run, **Stop** to halt.

> Tip: use `127.0.0.1`, not `localhost` — some browsers resolve `localhost` to IPv6 first and show a blank page. (The server is dual-stack, so both *should* work.)

---

## Project structure

```
agents/            8 Gemma 4 agents (one file each) + cerebras_client + schemas
incidents/         911 incident presets (call text + coordinates)
orchestrator.py    response pipeline, fleet, and battery model
webots/
  worlds/          rescue_city_mesh.wbt  ← main world (+ alternate OSM worlds)
  controllers/     drone_agent  ← embodied runtime that flies drones & calls agents
  assets/          3D models — city mesh, towers, vehicles, props
  dashboard/       index.html  ← live dashboard   ·   picker.html ← placement tool
  serve.py         dashboard server + Webots launcher (Run / Reset / Stop)
requirements.txt
.env.example       copy to .env and add your CEREBRAS_API_KEY (never committed)
```

---

## Credits & attributions

- **3D City** — *"City pack 2"* by **[Pasha](https://sketchfab.com/Pasha.)** on Sketchfab ([model](https://skfb.ly/pzVr9)), licensed under **[CC Attribution 4.0](http://creativecommons.org/licenses/by/4.0/)**. The city skyline our entire simulation is built on comes from this pack.
- **Drone & pedestrians** — Mavic 2 Pro and Pedestrian PROTOs from **Webots** (Cyberbotics).
- **Inference** — **Gemma 4** by Google DeepMind, served on **Cerebras**.

*Additional vehicle/prop models are placeholders from free sources; replace `webots/assets/*` with your own licensed models for distribution.*

---

*Built with Gemma 4 on Cerebras.*
