"""Command Center orchestrator — runs the 5-agent emergency mission pipeline.

  911 call ─▶ Dispatcher ─▶ Path Planner ─▶ [drone flies] ─▶ Perception(frame)
            ─▶ Analyst(vision) ─▶ Executor ─▶ report to Command Center

Every Gemma 4 call's latency is captured so the demo can show the end-to-end mission
"think time" on Cerebras. Run this file directly to exercise the whole pipeline in
mock mode (before API access) or live (once CEREBRAS_API_KEY is set).

  python orchestrator.py            # runs all 3 incidents
  python orchestrator.py fire       # runs one incident
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

from agents import dispatcher, path_planner, perception, analyst, executor, fleet_manager, surveillance, coordinator
from incidents.presets import PRESETS

# A small default fleet/city state. Replace tower/drone coords with the real ones
# from the Webots world once it's built.
# Central Emergency Drone Base: the towers are clustered at the city centre; incidents
# happen far across the city (see presets), so drones launch and fly fast + direct over
# the traffic — the whole point of aerial response. Tower (x,y) must match rescue_city.wbt.
DEFAULT_FLEET = {
    # x,y = stylized-city coords; osm_xy = real charging-station sites across Morges town.
    "towers": [
        {"id": 1, "x": -14, "y": 10, "osm_xy": (-828.9, 363.2), "mesh_xy": (-168, 63),
         "capacity": 2, "charged_drones": [3]},
        {"id": 2, "x": 14, "y": 10, "osm_xy": (-231.6, 413.3), "mesh_xy": (134, -103),
         "capacity": 2, "charged_drones": [1]},
        {"id": 3, "x": 0, "y": -16, "osm_xy": (-584.1, -168.0), "mesh_xy": (39, 160),
         "capacity": 1, "charged_drones": [2]},
    ],
    "drones": [
        {"id": 1, "payload": "first_aid_kit", "battery": 92, "home_tower": 2},
        {"id": 2, "payload": "extinguisher", "battery": 78, "home_tower": 3},
        {"id": 3, "payload": "camera_siren", "battery": 100, "home_tower": 1},
    ],
}

# ---- Battery model (all rates tunable here or via env BATTERY_*) -----------
import math as _math

BATTERY = {
    # % battery consumed per metre of horizontal flight
    "decay_per_m": float(os.environ.get("BATTERY_DECAY_PER_M", 0.12)),
    # % consumed per second while hovering / performing an action
    "hover_decay_per_s": float(os.environ.get("BATTERY_HOVER_DECAY_PER_S", 0.5)),
    # safety reserve % that must remain after reaching a tower
    "reserve": float(os.environ.get("BATTERY_RESERVE", 8.0)),
    # at/below this %, the drone asks the Fleet Manager agent: continue or return?
    "low_threshold": float(os.environ.get("BATTERY_LOW_THRESHOLD", 35.0)),
}

# World-placement transform. Fleet/incident coordinates below are relative to a
# scenario origin; the controller sets these per world so the SAME layout drops into
# either map. Standard city: scale 1, origin (0,0). OSM city: scaled down to fit an
# inland plaza, origin = that plaza's centre.
CITY_ORIGIN = [0.0, 0.0]
CITY_SCALE = [1.0]
# Absolute-placement key per world: "" = relative (stylized city); "osm" uses tower
# osm_xy / incident osm_location; "mesh" uses mesh_xy / mesh_location (imported city).
SITE_KEY = [""]


def flight_cost(dist_m: float) -> float:
    """Battery %% needed to fly `dist_m` metres."""
    return dist_m * BATTERY["decay_per_m"]


def can_reach(battery: float, dist_m: float, reserve: float | None = None) -> bool:
    reserve = BATTERY["reserve"] if reserve is None else reserve
    return battery - flight_cost(dist_m) >= reserve


def _dist(a, b) -> float:
    return _math.hypot(a[0] - b[0], a[1] - b[1])


def nearest_tower(pos, fleet=DEFAULT_FLEET, reachable_with: float | None = None):
    """Closest tower to `pos` (origin-aware). If reachable_with (battery%) is given,
    prefer the closest tower the drone can actually still reach on that charge."""
    towers = [(t, _dist(pos, _tower_xy(fleet, t["id"]))) for t in fleet["towers"]]
    towers.sort(key=lambda td: td[1])
    if reachable_with is not None:
        for t, d in towers:
            if can_reach(reachable_with, d):
                return t, d
    return towers[0]


def choose_return_tower(battery: float, pos, home_tower_id: int, fleet=DEFAULT_FLEET) -> dict:
    """After a mission: go to the assigned/home tower if the charge allows, else
    divert to the nearest reachable tower. Returns a decision dict for logging."""
    home = _tower_xy(fleet, home_tower_id)
    d_home = _dist(pos, home)
    if can_reach(battery, d_home):
        return {"tower_id": home_tower_id, "xy": (home[0], home[1]), "dist": d_home,
                "reason": f"battery {battery:.0f}% — enough to return to assigned tower {home_tower_id}"}
    t, d = nearest_tower(pos, fleet, reachable_with=battery)
    txy = _tower_xy(fleet, t["id"])
    return {"tower_id": t["id"], "xy": (txy[0], txy[1]), "dist": d,
            "reason": f"battery {battery:.0f}% — low, diverting to nearest tower {t['id']}"}


@dataclass
class Step:
    agent: str
    latency_s: float
    provider: str
    output: dict | str


@dataclass
class MissionTrace:
    incident_id: str
    steps: list[Step] = field(default_factory=list)

    @property
    def total_latency(self) -> float:
        return sum(s.latency_s for s in self.steps)

    def add(self, agent, result):
        out = result.parsed if result.parsed is not None else result.content
        self.steps.append(Step(agent, result.latency_s, result.provider, out))


def _tower_xy(fleet, tower_id):
    ox, oy = CITY_ORIGIN
    s = CITY_SCALE[0]
    key = SITE_KEY[0]
    for t in fleet["towers"]:
        if t["id"] == tower_id:
            if key and t.get(key + "_xy"):
                return (t[key + "_xy"][0], t[key + "_xy"][1], 0.0)
            return (t["x"] * s + ox, t["y"] * s + oy, 0.0)
    return (ox, oy, 0.0)


def incident_xy(inc):
    """Incident location in world coordinates. For absolute-placement worlds (osm/mesh)
    the incident sits on a real street; otherwise scale+origin from the base."""
    key = SITE_KEY[0]
    if key:
        loc = getattr(inc, key + "_location", None)
        if loc:
            return (loc[0], loc[1], 0.0)
    ox, oy = CITY_ORIGIN
    s = CITY_SCALE[0]
    return (inc.location[0] * s + ox, inc.location[1] * s + oy, 0.0)


def _emitter(trace, on_event):
    def emit(agent, result):
        trace.add(agent, result)
        if on_event:
            on_event(agent, result)
    return emit


def coordination_phase(threat_desc: str, threat_level: str, gps, trace=None,
                       on_event=None, mock_response=None) -> dict:
    """Gemma 4 Coordinator decides whether the patrol needs backup, and how many units."""
    trace = trace if trace is not None else MissionTrace("patrol")
    emit = _emitter(trace, on_event)
    r = coordinator.decide(threat_desc, threat_level, gps, mock_response=mock_response)
    emit("Coordinator", r)
    return r.parsed or {}


def perception_phase(image: str, trace=None, on_event=None, mock_response=None) -> dict:
    """The drone's eyes: Gemma looks at the live frame and decides whether the subject
    is well framed or the drone should reposition (e.g. descend for a closer look).
    The controller acts on `suggest_move` — so the IMAGE controls the drone."""
    trace = trace if trace is not None else MissionTrace("perception")
    emit = _emitter(trace, on_event)
    fr = perception.check_frame(image, mock_response=mock_response)
    emit("Perception", fr)
    out = fr.parsed or {}
    return {"framed": bool(out.get("framed", True)),
            "move_direction": out.get("move_direction", "centered"),
            "target_altitude_m": out.get("target_altitude_m", 6.0),
            "reason": out.get("reason", "")}


def surveillance_phase(image: str, gps, trace=None, on_event=None,
                       mock_response=None) -> dict:
    """Patrol overwatch: a single geo-tagged aerial frame -> Gemma threat scan.
    `gps` is the drone's (x, y) when the frame was taken, so a detection carries the
    exact coordinate to return to. Returns {scan, gps}."""
    trace = trace if trace is not None else MissionTrace("patrol")
    emit = _emitter(trace, on_event)
    s = surveillance.scan(image, mock_response=mock_response)
    emit("Surveillance", s)
    return {"scan": s.parsed or {}, "gps": [round(gps[0], 1), round(gps[1], 1)]}


def dispatch_phase(incident_id: str, fleet=DEFAULT_FLEET, trace=None, on_event=None) -> dict:
    """STEP 1 — Dispatcher classifies the 911 call and assigns the nearest tower/drone.
    Returns {dispatch, tower_xyz}. The controller teleports the drone to tower_xyz
    (the nearest tower's drone launching), THEN routes from the drone's live GPS."""
    inc = PRESETS[incident_id]
    trace = trace if trace is not None else MissionTrace(incident_id)
    emit = _emitter(trace, on_event)
    d = dispatcher.run(inc.call_text, fleet, mock_response=inc.mock_dispatch)
    emit("Dispatcher", d)
    dispatch = d.parsed or inc.mock_dispatch
    return {"dispatch": dispatch, "tower_xyz": _tower_xy(fleet, dispatch["from_tower_id"]),
            "trace": trace}


def route_phase(incident_id: str, start, trace=None, on_event=None) -> dict:
    """STEP 2 — Path Planner routes from the drone's LIVE position `start` to the incident."""
    inc = PRESETS[incident_id]
    target = incident_xy(inc)
    trace = trace if trace is not None else MissionTrace(incident_id)
    emit = _emitter(trace, on_event)
    p = path_planner.run(start, target, mock_response={
        "waypoints": [
            {"x": start[0], "y": start[1], "z": 28, "label": "ascend from current position"},
            {"x": target[0], "y": target[1], "z": 28, "label": "direct transit above the city"},
            {"x": target[0], "y": target[1], "z": 7, "label": "descend over scene"},
        ],
        "cruise_altitude": 28, "eta_seconds": 12, "notes": "Direct overflight from live position.",
    })
    emit("PathPlanner", p)
    return {"flight_plan": p.parsed, "start_xyz": tuple(start), "target_xyz": target,
            "trace": trace}


def plan_phase(incident_id: str, fleet=DEFAULT_FLEET, drone_location=None,
               trace: MissionTrace | None = None, on_event=None) -> dict:
    """Compose dispatch + route (used by the standalone runner). The Webots controller
    calls dispatch_phase and route_phase separately so it can teleport between them."""
    trace = trace if trace is not None else MissionTrace(incident_id)
    dp = dispatch_phase(incident_id, fleet, trace=trace, on_event=on_event)
    start = tuple(drone_location) if drone_location is not None else dp["tower_xyz"]
    rp = route_phase(incident_id, start, trace=trace, on_event=on_event)
    return {**dp, **rp, "trace": trace}


def analyze_phase(incident_id: str, image: str, dispatch: dict,
                  trace: MissionTrace | None = None, on_event=None) -> dict:
    """ON ARRIVAL (best framed view): Analyst does the multimodal scene analysis, Executor
    decides drone commands + the command-center report. Perception/framing now runs as its
    own controller-driven loop (perception_phase in scan_sweep), so it is NOT repeated here.
    `image` = path/URL to the live Webots camera frame.
    """
    inc = PRESETS[incident_id]
    trace = trace if trace is not None else MissionTrace(incident_id)

    def emit(agent, result):
        trace.add(agent, result)
        if on_event:
            on_event(agent, result)

    # 4) Analyst — multimodal scene analysis (the core Gemma 4 vision step)
    a = analyst.run(image, inc.title, mock_response=inc.mock_analysis)
    emit("Analyst", a)
    analysis = a.parsed or inc.mock_analysis

    # 5) Executor — act + report to command center
    e = executor.run(analysis, dispatch["incident_type"], dispatch["required_payload"],
                     mock_response=_mock_exec(dispatch, analysis))
    emit("Executor", e)

    return {"analysis": analysis, "execution": e.parsed, "trace": trace}


def battery_decision_phase(battery, pos, home_tower_id, dist_remaining=0.0,
                           mission_done=True, fleet=DEFAULT_FLEET, trace=None, on_event=None) -> dict:
    """Low-battery trigger: ask the Fleet Manager (Gemma) whether to CONTINUE or RETURN,
    then resolve the return tower (home if the charge allows, else nearest reachable).
    Returns {decision, return}. `return` is None when continuing."""
    trace = trace if trace is not None else MissionTrace("battery")
    emit = _emitter(trace, on_event)
    home = _tower_xy(fleet, home_tower_id)
    d_home = _dist(pos, home)
    _, d_near = nearest_tower(pos, fleet, reachable_with=battery)
    reserve = BATTERY["reserve"]
    can_home = can_reach(battery, d_home)

    # mock decision mirrors the deterministic safety logic (used until live Gemma is on)
    if not mission_done and can_reach(battery, dist_remaining + d_home):
        mock = {"decision": "continue_mission", "return_target": "none",
                "rationale": f"battery {battery:.0f}% covers finishing the task plus the return."}
    else:
        target = "home_tower" if can_home else "nearest_tower"
        mock = {"decision": "return_to_base", "return_target": target,
                "rationale": (f"battery {battery:.0f}% — "
                              + ("enough to reach home tower." if can_home
                                 else "too low for home tower; diverting to nearest."))}

    r = fleet_manager.run(battery, dist_remaining, d_home, d_near, reserve, mission_done,
                          mock_response=mock)
    emit("FleetManager", r)
    decision = r.parsed or mock

    ret = choose_return_tower(battery, pos, home_tower_id, fleet) \
        if decision["decision"] == "return_to_base" else None
    return {"decision": decision, "return": ret, "trace": trace}


def run_mission(incident_id: str, fleet=DEFAULT_FLEET, image_override: str | None = None,
                on_event=None) -> MissionTrace:
    """Full standalone pipeline (no Webots). plan -> [drone would fly] -> analyze."""
    inc = PRESETS[incident_id]
    trace = MissionTrace(incident_id)
    image = image_override or inc.test_image
    planned = plan_phase(incident_id, fleet, trace=trace, on_event=on_event)
    analyze_phase(incident_id, image, planned["dispatch"], trace=trace, on_event=on_event)
    return trace


def _mock_exec(dispatch, analysis):
    action = analysis.get("recommended_action", "relay_to_responders")
    cmd_map = {
        "deliver_first_aid": ["descend_to_subject", "release_first_aid", "ascend_and_relay"],
        "hold_position_and_warn": ["circle_and_record", "activate_siren", "broadcast_warning"],
        "deploy_extinguisher": ["descend_to_subject", "discharge_extinguisher", "ascend_and_relay"],
        "relay_to_responders": ["ascend_and_relay"],
        "continue_observation": ["circle_and_record"],
    }
    return {
        "drone_commands": cmd_map.get(action, ["ascend_and_relay"]),
        "command_center_report": f"Executed {action}. {analysis.get('action_detail','')}",
        "status": "responders_needed" if analysis.get("injuries_suspected") else "ongoing",
    }


def print_trace(trace: MissionTrace):
    inc = PRESETS[trace.incident_id]
    print(f"\n{'='*70}\n  MISSION: {inc.title}\n{'='*70}")
    for s in trace.steps:
        tag = "MOCK" if s.provider == "mock" else s.provider.upper()
        head = s.output if isinstance(s.output, str) else _one_line(s.output)
        print(f"  [{s.latency_s*1000:6.0f} ms | {tag:8}] {s.agent:12} -> {head}")
    print(f"  {'-'*66}")
    print(f"  END-TO-END Gemma 4 think time: {trace.total_latency*1000:.0f} ms "
          f"across {len(trace.steps)} agents\n")


def _one_line(d: dict) -> str:
    for k in ("summary", "recommended_action", "command_center_report", "notes", "framed"):
        if k in d:
            return f"{k}={d[k]}"
    return str(d)[:80]


if __name__ == "__main__":
    which = sys.argv[1:] or list(PRESETS.keys())
    print(f"\n  Provider: {'MOCK (no API key)' if __import__('agents').gemma.mock else 'CEREBRAS (live Gemma 4 31B)'}")
    for inc_id in which:
        if inc_id not in PRESETS:
            print(f"  unknown incident '{inc_id}', choices: {list(PRESETS)}")
            continue
        print_trace(run_mission(inc_id))
