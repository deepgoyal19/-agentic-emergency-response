# Rescue City — Gemma 4 on Cerebras (Track 1: Multiverse Agents)

An emergency-response drone simulation where **6 Gemma 4 agents on Cerebras** coordinate
to dispatch a drone from a charging tower, fly it to a 911 incident, analyze the scene
from the drone camera with **Gemma 4 multimodal vision**, act, and return — battery-aware.

## Architecture

```
911 call ─▶ Dispatcher ─▶ [launch from nearest tower] ─▶ Path Planner (from live GPS)
        ─▶ fly ─▶ Perception ─▶ Analyst (vision) ─▶ Executor ─▶ Fleet Manager (battery)
        ─▶ return to home/nearest tower ─▶ recharge
```

All six agents are separate `gemma-4-31b` calls on Cerebras (`agents/*.py`), each with a
strict structured-output schema. The Webots controller (`webots/controllers/drone_agent/`)
is the embodied runtime: it flies the Mavic 2 Pro, captures real camera frames, and calls
the agents at the right moments. The orchestrator (`orchestrator.py`) holds the pipeline,
fleet, and battery model.

## Run the demo

1. **Pick an incident** — edit `webots/mission.json`:
   `{"incident": "accident"}`  (or `fire`, or `stalker`)

2. **Start the live dashboard** — double-click `webots/run_dashboard.bat`, then open
   <http://localhost:8000/dashboard/>. It shows each agent firing with its Cerebras
   latency, the battery gauge, the aerial frame Gemma analyzed, and the mission log.

3. **Open a world in Webots**, press ▶. Two worlds are available:
   - `webots/worlds/rescue_city.wbt` — custom stylized city (light, fast, moving traffic).
   - `webots/worlds/rescue_city_osm.wbt` — the **real town of Morges, Switzerland**
     (OpenStreetMap import: 1101 buildings, real roads). Heavier; best for cinematic
     wide shots. Our scenario auto-drops into an open square (origin handled by the
     controller from the world filename).

   The drone launches from the dispatched tower, flies to the incident, analyzes, acts,
   and returns. The dashboard updates live for either world.

4. **Headless (no GUI)** — render all three incidents and their camera frames:
   `bash webots/validate_all.sh`  → frames in `webots/frames/`.

## Mock vs live Gemma

Runs in **mock mode** with no API key (full pipeline, for development). To go live on
Cerebras, put your key in `code/.env`:

```
CEREBRAS_API_KEY=csk-...
```

The client auto-detects the key and the same pipeline runs on real `gemma-4-31b`.
The dashboard provider badge flips `MOCK → CEREBRAS · LIVE` and the latencies become real.

## Tuning (all via env or code)

- Battery: `BATTERY_DECAY_PER_M`, `BATTERY_HOVER_DECAY_PER_S`, `BATTERY_RESERVE`,
  `BATTERY_LOW_THRESHOLD` (see `orchestrator.BATTERY`).
- Fleet/towers/drone batteries: `orchestrator.DEFAULT_FLEET`.
- Incidents (911 text, coordinates): `incidents/presets.py`.
- Reasoning: the Analyst uses `reasoning_effort="low"`; set to `"none"` for max speed.
