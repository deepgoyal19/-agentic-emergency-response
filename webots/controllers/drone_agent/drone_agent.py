"""Webots controller — the embodied agent runtime for the rescue drone.

Lifecycle of one mission:
  1. PLAN   : call Gemma agents (Dispatcher + Path Planner) -> incident target + waypoints
  2. TAKEOFF: rise to cruise altitude
  3. CRUISE : fly tower -> incident (yaw/pitch nav from the proven Mavic patrol PID)
  4. DESCEND: drop over the scene, aim the gimbal down
  5. ANALYZE: capture a REAL camera frame -> Gemma vision (Perception+Analyst+Executor)
  6. ACT    : animate the Executor's drone commands, then return to the tower

While a Gemma call runs we simply don't step the simulation, so physics pauses and the
drone hangs safely in place — and the visible "freeze then snap to action" sells how
fast Cerebras returns. Every agent latency is logged to mission_log.json for the dashboard.

Runs in MOCK mode with no API key (develop the whole flight today); flips to real Gemma 4
the instant CEREBRAS_API_KEY is set. Which incident to run is read from webots/mission.json.
"""
from controller import Supervisor
import json
import math
import os
import re
import sys

try:
    import numpy as np
except ImportError:
    sys.exit("This controller needs numpy:  python -m pip install numpy")

# --- make the agent package importable from inside Webots ------------------
HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))   # -> f:\Hackathon\code
sys.path.insert(0, CODE_ROOT)

import orchestrator  # noqa: E402
from orchestrator import (dispatch_phase, route_phase, analyze_phase,  # noqa: E402
                          battery_decision_phase, surveillance_phase, perception_phase,
                          coordination_phase, incident_xy, BATTERY, DEFAULT_FLEET, MissionTrace)
from incidents.presets import PRESETS  # noqa: E402

WEBOTS_DIR = os.path.join(CODE_ROOT, "webots")
FRAMES_DIR = os.path.join(WEBOTS_DIR, "frames")
MISSION_FILE = os.path.join(WEBOTS_DIR, "mission.json")
BACKUP_FILE = os.path.join(WEBOTS_DIR, "backup_request.json")   # patrol -> available units
os.makedirs(FRAMES_DIR, exist_ok=True)

# Live onboard-camera POV: drone id -> live frame file the dashboard polls. With one
# drone today (id 1) this is just live_1.png; the scheme extends to a fleet.
LIVE_PERIOD_MS = 32          # onboard camera refresh (~31 fps) for the overlay + feed
LIVE_SAVE_EVERY = 4          # write the live feed file every N control steps (~15 fps to disk)
PAD_Z = 4.0                  # tower landing-pad height: drones rest ON TOP of the tower
                             # (model ~3.6 m tall) — just clear of its roof, not embedded in it

# Multi-drone scenario (config-driven — add rows to add drones/incidents).
# Each Mavic body named "Drone<N>" picks role N here. Responders handle one incident on
# a staggered report timer (simultaneous emergencies); the patrol drone circles the city.
# Single-Mavic worlds (no "Drone<N>" name) fall back to mission.json.
# A burning high-rise the fire-brigade squad attacks. 90 m — taller than the mesh
# skyline (~75 m) so the rooftop fire is never occluded from straight above.
HIGHRISE = {"xy": (150.0, 150.0), "roof": 90.0}

MULTI = {
    # --- ground responders: simultaneous emergencies, staggered 911 reports ---
    1: {"role": "responder", "incident": "accident", "delay": 0.0,  "payload": "first aid kit"},
    2: {"role": "responder", "incident": "stalker",  "delay": 6.0,  "payload": "camera + siren"},
    3: {"role": "responder", "incident": "fire",     "delay": 12.0, "payload": "extinguisher"},
    # --- continuous patrol over the city ---
    4: {"role": "patrol", "payload": "surveillance", "alt": 82.0, "speed": 16.0,
        "scan_alt": 7.0, "patrol_alt": 12.0, "hotspot": (-10.0, 30.0),
        # genuine patrol: fly the loop at patrol_alt scanning each waypoint; the early waypoints
        # are away from the suspect (sectors clear), and the patrol only SPOTS him when a scan
        # catches him (the (-10,30) waypoint) — then it converges. No beeline to a known spot.
        "route": [(-120, 60), (-60, 0), (-10, 30), (80, -40)],
        # the suspect walks this road path; the patrol tracks it BY VISION (never reads coords)
        "suspect_path": [(-10.0, 30.0), (-10.0, 90.0)]},
    # --- fire-brigade squad: one burning skyscraper, three drones attack by sector ---
    5: {"role": "firebrigade", "incident": "highrise_fire", "delay": 4.0,  "payload": "water cannon", "roof": HIGHRISE["roof"], "sector": 0},
    6: {"role": "firebrigade", "incident": "highrise_fire", "delay": 9.0,  "payload": "foam jet",     "roof": HIGHRISE["roof"], "sector": 1},
    7: {"role": "firebrigade", "incident": "highrise_fire", "delay": 15.0, "payload": "extinguisher", "roof": HIGHRISE["roof"], "sector": 2},
}


def clamp(v, lo, hi):
    return min(max(v, lo), hi)


def read_mission():
    if os.path.exists(MISSION_FILE):
        try:
            return json.load(open(MISSION_FILE))
        except Exception:
            pass
    return {}


def select_incident():
    return read_mission().get("incident", "accident")


class RescueDrone(Supervisor):
    # PID constants from the official Mavic patrol controller (empirically tuned).
    K_VERTICAL_THRUST = 68.5
    K_VERTICAL_OFFSET = 0.6
    K_VERTICAL_P = 3.0            # proven-stable vertical PID (do not raise — flips at speed)
    K_ROLL_P = 50.0
    K_PITCH_P = 30.0
    MAX_YAW_DISTURBANCE = 0.6     # quicker heading changes
    MAX_PITCH_DISTURBANCE = -1.0
    MAX_FWD_PITCH = -6.5          # faster transit (clear high path = no collisions)
    ARRIVAL_RADIUS = 4.0          # horizontal metres to count as "over the scene"

    def __init__(self):
        super().__init__()
        self.dt = int(self.getBasicTimeStep())

        self.camera = self.getDevice("camera")            # front gimbal cam (cinematic FPV)
        self.down_cam = self.getDevice("down camera")     # fixed straight-down cam (None on old worlds)
        self.chase_cam = self.getDevice("chase camera")   # 3rd-person view from behind/above the drone
        self.feed_cam = self.down_cam or self.camera      # ANALYSIS frames (Gemma) = down cam
        self.live_cam = self.chase_cam or self.feed_cam   # dashboard ONBOARD view = chase cam
        self.imu = self.getDevice("inertial unit"); self.imu.enable(self.dt)
        self.gps = self.getDevice("gps"); self.gps.enable(self.dt)
        self.gyro = self.getDevice("gyro"); self.gyro.enable(self.dt)
        self.cam_pitch = self.getDevice("camera pitch")
        self.cam_pitch.setPosition(0.25)                  # front cam looks ahead (FPV)

        self.motors = [self.getDevice(n) for n in (
            "front left propeller", "front right propeller",
            "rear left propeller", "rear right propeller")]
        # Constant hover-thrust spin: propellers look alive AND net force ~= gravity, so
        # the kinematic (supervisor-driven) position control stays clean and jitter-free.
        for m, s in zip(self.motors, (self.K_VERTICAL_THRUST, -self.K_VERTICAL_THRUST,
                                      -self.K_VERTICAL_THRUST, self.K_VERTICAL_THRUST)):
            m.setPosition(float("inf"))
            m.setVelocity(s)
        node = self.getSelf()
        self._node = node                          # for zeroing velocity each step
        self._tf = node.getField("translation")   # supervisor handles to fly kinematically
        self._rf = node.getField("rotation")
        self._heading = math.pi
        self._last_pose = None                     # last valid (x,y,z) for NaN recovery

        self.events = []          # mission_log for the dashboard
        self.phases = []          # high-level mission phase markers

        # --- battery / fleet state ---
        self.battery = 100.0      # set from the dispatched drone in run()
        self._batt_pos = None     # last (x,y) used for distance-based battery decay
        self._batt_low_fired = False
        self.incident_id = None   # set in run(); surfaced to the live dashboard
        self.map_data = None      # {towers, incident} world coords for the 2D map
        self.active = False       # True once the drone actually launches (not standby)
        self.incident_done = False  # True once the scene is handled (clears it off the map)
        self.incident_report = ""   # the 911 call text, shown in the mission log

        # --- drone identity (multi-drone) ---
        nm = self.getName() or ""
        _m = re.match(r"[Dd]rone\s*(\d+)", nm)
        self.drone_id = int(_m.group(1)) if _m else 1
        self.role_cfg = MULTI.get(self.drone_id, {}) if _m else {}
        self.role = self.role_cfg.get("role", "responder")
        # per-drone dashboard files (so several drones can stream at once)
        self.log_file = os.path.join(WEBOTS_DIR, f"mission_log_{self.drone_id}.json")
        self.live_png = os.path.join(FRAMES_DIR, f"live_{self.drone_id}.png")
        self.live_tmp = os.path.join(FRAMES_DIR, f"live_{self.drone_id}_tmp.png")
        self.live_pos = os.path.join(WEBOTS_DIR, f"live_pos_{self.drone_id}.json")
        self.frame_png = os.path.join(FRAMES_DIR, f"analysis_{self.drone_id}.png")  # Gemma-analyzed still

        # pick the world-placement transform from the world filename (osm vs standard)
        wp = (self.getWorldPath() or "").lower()
        self.site = "osm" if "osm" in wp else ("mesh" if "mesh" in wp else "")
        orchestrator.SITE_KEY[0] = self.site
        orchestrator.CITY_ORIGIN[:] = [0.0, 0.0]
        orchestrator.CITY_SCALE[0] = 1.0
        print(f"[drone_agent] site='{self.site}'", flush=True)

        # live onboard-camera POV (Webots overlay + dashboard feed). On by default for
        # the demo; headless validation sets "live_view": false to stay fast.
        self.live_view = bool(read_mission().get("live_view", True))
        self._live_n = 0
        if self.live_view:
            # the dashboard onboard view streams from the CHASE cam (3rd-person, behind the
            # drone). The analysis (down/front) cam is enabled on demand in capture_frame.
            self.live_cam.enable(LIVE_PERIOD_MS)

    # ------------------------------------------------------------------ #
    def update_battery(self, x, y):
        """Distance- + time-based battery decay, called once per simulation step
        (GPS already read in tick(); passed in to avoid a redundant sensor read)."""
        if not (math.isfinite(x) and math.isfinite(y)):
            return                                 # skip NaN frames (no phantom drain)
        if self._batt_pos is not None:
            moved = ((x - self._batt_pos[0]) ** 2 + (y - self._batt_pos[1]) ** 2) ** 0.5
            if moved < 50.0:                       # ignore teleport-sized jumps
                decay = BATTERY["decay_per_m"] * (0.6 if self.role == "patrol" else 1.0)
                self.battery -= moved * decay
                # time-based drain: the PATROL is airborne+scanning continuously, so it drains
                # at the full hover rate (visibly) — other drones only sip while hovering.
                hover_mult = 1.0 if self.role == "patrol" else 0.1
                self.battery -= (self.dt / 1000.0) * BATTERY["hover_decay_per_s"] * hover_mult
                self.battery = max(0.0, self.battery)
        self._batt_pos = (x, y)

    def tick(self):
        """Wrap step() so every advanced step also decays the battery and, when live
        view is on, streams the onboard camera to the dashboard feed."""
        r = Supervisor.step(self, self.dt)
        if r != -1:
            # runaway watchdog: propeller thrust can fling a drone off the map if it ever
            # slips out of kinematic control. If it leaves sane bounds, snap it back.
            gx, gy, gz = self.gps.getValues()             # single GPS read, reused below
            if (not math.isfinite(gx)) or abs(gx) > 450 or abs(gy) > 450 or gz > 280 or gz < -5:
                self._node.resetPhysics()
                px, py, pz = self._last_pose or (0.0, 0.0, 2.0)
                self._set_pose(px, py, max(pz, 1.0), self._heading)
                gx, gy, gz = px, py, max(pz, 1.0)
            self.update_battery(gx, gy)
            self._live_n += 1
            # stream live drone position for the 2D map (~25 Hz), always
            if self._live_n % 5 == 0:
                try:
                    x, y, alt = gx, gy, gz
                    if not (math.isfinite(x) and math.isfinite(y)):
                        x, y = (self._last_pose or (0, 0, 0))[:2]
                    inc = (self.map_data or {}).get("incident") if self.active else None
                    tmp = self.live_pos + ".tmp"
                    with open(tmp, "w") as f:
                        json.dump({"x": round(x, 1), "y": round(y, 1), "alt": round(alt, 1),
                                   "battery": round(self.battery, 1), "role": self.role,
                                   "active": self.active, "incident": inc,
                                   "incident_id": self.incident_id, "done": self.incident_done,
                                   "threat": getattr(self, "threat_gps", None)}, f)
                    os.replace(tmp, self.live_pos)
                except Exception:
                    pass
            if self.live_view and self._live_n % LIVE_SAVE_EVERY == 0:
                try:
                    self.live_cam.saveImage(self.live_tmp, 80)   # chase cam -> dashboard onboard view
                    os.replace(self.live_tmp, self.live_png)   # atomic: no half-written reads
                except Exception:
                    pass
        return r

    def _write_log(self):
        data = {
            "drone_id": self.drone_id,
            "role": self.role,
            "incident": self.incident_id,
            "report": self.incident_report,
            "resolved": self.incident_done,
            "battery": round(self.battery, 1),
            "total_ms": sum(e.get("latency_ms", 0) for e in self.events),       # wall-clock
            "total_cerebras_ms": sum(e.get("cerebras_ms", 0) for e in self.events),  # Cerebras compute
            "map": self.map_data,
            "events": self.events,
            "phases": self.phases,
        }
        # atomic write so the live dashboard never reads a half-written file
        tmp = self.log_file + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self.log_file)

    def note(self, msg):
        self.phases.append({"msg": msg, "sim_time": round(self.getTime(), 2)})
        self._write_log()
        print(f"  >> {msg}", flush=True)

    def log_event(self, agent, result):
        out = result.parsed if getattr(result, "parsed", None) is not None else result.content
        ti = result.time_info or {}
        ev = {
            "agent": agent,
            "latency_ms": round(result.latency_s * 1000),                 # wall-clock (incl. network)
            "cerebras_ms": round(ti.get("total_time", 0) * 1000),         # Cerebras compute (the speed story)
            "provider": result.provider,
            "output": out,
            "sim_time": round(self.getTime(), 2),
        }
        self.events.append(ev)
        tag = "MOCK" if result.provider == "mock" else result.provider.upper()
        head = out if isinstance(out, str) else json.dumps(out)[:90]
        print(f"  [{ev['latency_ms']:5d} ms | {tag:8}] {agent:11} -> {head}", flush=True)
        self._write_log()

    # ---- kinematic flight (Supervisor-driven: exact speed, smooth, crash-free) ----
    def _set_pose(self, x, y, z, heading):
        # ignore non-finite requests (a NaN would corrupt the node permanently)
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z) and math.isfinite(heading)):
            return
        self._tf.setSFVec3f([float(x), float(y), float(z)])
        self._rf.setSFRotation([0.0, 0.0, 1.0, float(heading)])
        # zero the body's velocity every step: we fly kinematically, but the propellers
        # keep applying thrust, so momentum would otherwise accumulate and blow up to NaN.
        try:
            self._node.setVelocity([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        except Exception:
            pass
        self._last_pose = (float(x), float(y), float(z))

    def _await_valid_gps(self, max_steps=400):
        """HOLD the drone on its spawn pad (kinematically) while the GPS warms up, then return a
        finite fix. The GPS yields NaN for the first step(s); without actively holding the pose,
        physics also drops the drone off the pad to the tower's FOOT before kinematic control
        starts. We pin it at (spawn_x, spawn_y, PAD_Z) with zero velocity each warm-up step."""
        sp = self._tf.getSFVec3f()
        hx, hy = sp[0], sp[1]
        hd = self._rf.getSFRotation()[3]
        self._heading = hd
        for _ in range(max_steps):
            x, y, z = self.gps.getValues()
            if math.isfinite(x) and math.isfinite(y) and math.isfinite(z):
                self._last_pose = (x, y, z)
                return x, y, z
            self._set_pose(hx, hy, PAD_Z, hd)         # pin on the pad so it can't fall
            if Supervisor.step(self, self.dt) == -1:
                break
        return self.gps.getValues()

    def _read_xyz(self):
        """GPS position, with NaN recovery (reset physics + restore last good pose)."""
        x, y, z = self.gps.getValues()
        if math.isfinite(x) and math.isfinite(y) and math.isfinite(z):
            return x, y, z
        # physics produced a NaN — recover to the last known-good pose
        self._node.resetPhysics()
        x, y, z = self._last_pose or (0.0, 0.0, 1.0)
        self._set_pose(x, y, z, self._heading)
        return x, y, z

    def fly_to(self, tx, ty, altitude, speed=18.0, hold_steps=8, max_steps=6000, **kw):
        """Move kinematically to (tx, ty, altitude) at `speed` m/s, nose pointed along the
        path, then hold. Fast and crash-free — the drone's real value is range + speed."""
        x, y, z = self._read_xyz()
        steps = 0
        while self.tick() != -1:
            steps += 1
            dx, dy, dz = tx - x, ty - y, altitude - z
            dist = (dx * dx + dy * dy + dz * dz) ** 0.5
            if not math.isfinite(dist):              # recover from a physics NaN, keep flying
                x, y, z = self._read_xyz()
                continue
            if dist < 0.6:
                for _ in range(hold_steps):
                    if self.tick() == -1:
                        break
                    self._set_pose(tx, ty, altitude, self._heading)
                return True
            stp = min(speed * self.dt / 1000.0, dist)
            x += dx / dist * stp
            y += dy / dist * stp
            z += dz / dist * stp
            if (dx * dx + dy * dy) > 1.0:                 # face direction of horizontal travel
                self._heading = math.atan2(dy, dx)
            self._set_pose(x, y, z, self._heading)
            if steps % 1500 == 0:
                self.note(f"  flying: {dist:.0f} m to go @ {speed:.0f} m/s, alt {z:.0f} m")
            if steps >= max_steps:
                return False
        return False

    def hover(self, altitude, steps):
        x, y, _ = self.gps.getValues()
        for _ in range(steps):
            if self.tick() == -1:
                return
            self._set_pose(x, y, altitude, self._heading)

    def hover_at_current(self, steps):
        x, y, z = self.gps.getValues()
        for _ in range(steps):
            if self.tick() == -1:
                return
            self._set_pose(x, y, z, self._heading)

    def capture_frame(self, incident_id, path=None):
        """Shoot from a stable hover — the scene view Gemma analyzes. `path` overrides the
        default per-drone frame (used by the multi-view scan to save view 1..N)."""
        path = path or self.frame_png
        self.feed_cam.enable(self.dt)                # ensure the ANALYSIS cam is sampling (the
                                                     # live/dashboard view is a separate chase cam)
        x, y, z = self.gps.getValues()
        for _ in range(30):                          # hold steady so the frame renders + isn't blurred
            if self.tick() == -1:
                break
            self._set_pose(x, y, z, self._heading)
        self.feed_cam.saveImage(path, 95)
        return path

    def _shoot(self, cam, path, hold=18):
        """Capture one stable frame from a SPECIFIC camera (e.g. the 45°-down front cam or the
        straight-down cam) — lets the patrol build a multi-angle grid from two cameras."""
        if cam is None:
            return path
        cam.enable(self.dt)
        x, y, z = self.gps.getValues()
        for _ in range(hold):
            if self.tick() == -1:
                break
            self._set_pose(x, y, z, self._heading)
        cam.saveImage(path, 95)
        return path

    GROUND_MIN = 1.0    # absolute hard floor; Gemma picks the actual inspection altitude per scene

    def scan_sweep(self, incident_id, tx, ty, base_alt, dispatch, trace, facade=False):
        """Take MULTIPLE photos, saved as analysis_<id>_1..4.png for the dashboard grid.
        For a ground incident the move is AGENT-GUIDED: the drone takes a down photo, the
        Perception agent (Gemma) says which way to move to frame the subject, and the drone
        obeys — never descending below GROUND_MIN (its altitude sensor). Returns the final
        analysis from the best view."""
        if facade:                                    # fire-brigade: fixed facade sweep
            views = [(0, 9, base_alt + 4), (0, 7, base_alt), (2, 8, base_alt - 2), (-2, 8, base_alt + 1)]
            last = None
            for k, (dx, dy, az) in enumerate(views, 1):
                self.note(f"D{self.drone_id}: repositioning for view {k}/4 (alt {max(self.GROUND_MIN,az):.0f} m)")
                self.fly_to(tx + dx, ty + dy, max(self.GROUND_MIN, az), speed=4.0, hold_steps=6, max_steps=4000)
                if dx or dy:
                    self._heading = math.atan2(-dy, -dx)
                path = os.path.join(FRAMES_DIR, f"analysis_{self.drone_id}_{k}.png")
                self.capture_frame(incident_id, path=path)
                perception_phase(path, trace=trace, on_event=self.log_event)
                last = path
            return analyze_phase(incident_id, last, dispatch, trace=trace, on_event=self.log_event)

        # ---- ground incident: Gemma-directed reposition + Gemma-chosen inspection altitude ----
        # The drone takes a high framing shot, then DROPS to the close inspection altitude that
        # Gemma asks for (a car crash on the ground -> ~1-2 m). It commits to that altitude fast
        # so it ends up low over the scene "checking around," not hovering high.
        cx, cy, az = tx, ty, base_alt
        last = None
        for k in range(1, 5):
            self.fly_to(cx, cy, az, speed=4.0, hold_steps=6, max_steps=4000)
            gz = self.gps.getValues()[2]              # downward altitude sensor (above ground)
            path = os.path.join(FRAMES_DIR, f"analysis_{self.drone_id}_{k}.png")
            self.capture_frame(incident_id, path=path)
            perc = perception_phase(path, trace=trace, on_event=self.log_event)
            mv = perc.get("move_direction", "centered")
            target = clamp(float(perc.get("target_altitude_m", az)), self.GROUND_MIN, 12.0)
            self.note(f"D{self.drone_id}: altitude {gz:.1f} m AGL — Gemma wants ~{target:.1f} m "
                      f"for this scene (view {k}/4)")
            last = path
            if perc.get("framed") and abs(az - target) < 1.0:
                self.note(f"D{self.drone_id}: Gemma — framed at the inspection altitude ({az:.1f} m); "
                          f"capturing view {k}/4")
            step = 5.0                                # lateral move the agent directed
            if mv == "move_north":   cy += step
            elif mv == "move_south": cy -= step
            elif mv == "move_east":  cx += step
            elif mv == "move_west":  cx -= step
            gap = az - target                         # close most of the gap each view -> drops fast
            if abs(gap) > 0.4 or mv == "descend":
                move = max(4.0, abs(gap) * 0.7)
                az = max(target, az - move) if gap > 0 else min(target, az + move)
                verb = "descend" if gap > 0 else (mv.replace('_', ' ') if mv != "descend" else "hold")
                self.note(f"D{self.drone_id}: Gemma directs → {verb} to {az:.1f} m to inspect closely")
        return analyze_phase(incident_id, last, dispatch, trace=trace, on_event=self.log_event)

    def _drone_battery(self, drone_id):
        for d in DEFAULT_FLEET["drones"]:
            if d["id"] == drone_id:
                return float(d["battery"])
        return 100.0

    def teleport_to_tower(self, tower_xy, face_xy):
        """Reposition the drone onto its dispatched charging tower (the nearest tower's
        drone launching). Uses the Supervisor API; resets the battery odometer so the
        instantaneous reposition isn't counted as flown distance."""
        node = self.getSelf()
        heading = float(np.arctan2(face_xy[1] - tower_xy[1], face_xy[0] - tower_xy[0]))
        self._heading = heading
        node.getField("translation").setSFVec3f([float(tower_xy[0]), float(tower_xy[1]), PAD_Z])
        node.getField("rotation").setSFRotation([0.0, 0.0, 1.0, heading])
        node.resetPhysics()
        self._batt_pos = None
        # Hold the new pose for a few steps so GPS reflects it BEFORE any kinematic
        # method reads position (otherwise stale GPS snaps the drone back to spawn).
        for _ in range(4):
            if self.tick() == -1:
                return
            self._set_pose(tower_xy[0], tower_xy[1], PAD_Z, heading)

    def _return_and_land(self, ret, cruise, trace):
        if ret is None:
            return
        rx, ry = ret["xy"]
        self.note(f"Returning to tower {ret['tower_id']} — {ret['reason']}")
        self.fly_to(rx, ry, cruise, hold_steps=10, max_steps=9000)
        self.land(rx, ry)
        self.note(f"DOCKED & CHARGING at tower {ret['tower_id']} (battery {self.battery:.0f}%) — "
                  f"MISSION COMPLETE, Gemma 4 think time {trace.total_latency*1000:.0f} ms "
                  f"across {len(trace.steps)} agents")
        for _ in range(120):       # rest on the pad, then exit cleanly
            if self.tick() == -1:
                break
            self._set_pose(rx, ry, PAD_Z, self._heading)

    # ------------------------------------------------------------------ #
    def _reset_dashboard(self):
        """Clear this drone's stale dashboard artifacts so its panels reset."""
        views = [os.path.join(FRAMES_DIR, f"analysis_{self.drone_id}_{k}.png") for k in range(1, 5)]
        for fp in [self.frame_png, self.live_png, self.live_pos] + views:
            try:
                os.remove(fp)
            except OSError:
                pass

    def _stand_by(self, seconds):
        """Idle ON THE TOWER PAD until the staggered report comes in (keeps streaming live)."""
        x, y, _ = self.gps.getValues()
        if not (math.isfinite(x) and math.isfinite(y)):
            sp = self._tf.getSFVec3f(); x, y = sp[0], sp[1]
        for _ in range(max(1, int(seconds * 1000 / self.dt))):
            if self.tick() == -1:
                return
            self._set_pose(x, y, PAD_Z, self._heading)   # hold on the pad, not wherever it fell

    def run(self):
        self._await_valid_gps()        # don't act until the GPS has a real fix (avoids NaN strand)
        # role-based dispatch: responders + fire-brigade fly incidents; patrol circles.
        if self.role == "patrol":
            self.incident_id = "patrol"
            self._reset_dashboard()
            self.run_patrol()
            return
        incident_id = self.role_cfg.get("incident") or select_incident()
        self.incident_id = incident_id
        self._reset_dashboard()
        # Fire brigade: only the LEAD unit (sector 0) launches on the 911 call. The other
        # units STAGE at base and deploy only if Gemma's Coordinator calls for reinforcements
        # after the lead analyses how bad the blaze is.
        if self.role == "firebrigade" and int(self.role_cfg.get("sector", 0)) != 0:
            self._await_fire_backup(incident_id)
            return
        delay = float(self.role_cfg.get("delay", 0.0))
        if delay > 0:
            self.note(f"Drone {self.drone_id} ({self.role}) standing by — "
                      f"{incident_id} report comes in at T+{delay:.0f}s")
            self._stand_by(delay)
        if self.role == "firebrigade":
            self.run_highrise(incident_id)
        elif incident_id == "stalker":
            self.run_stalker(incident_id)
        else:
            self.run_incident(incident_id)

    def run_incident(self, incident_id):
        self.active = True                 # launching now -> appears on the dashboard
        inc = PRESETS[incident_id]
        self.incident_report = inc.call_text
        trace = MissionTrace(incident_id)
        tx, ty, _ = incident_xy(inc)
        # 2D map layout: all towers + this incident (world coords)
        self.map_data = {
            "towers": [list(orchestrator._tower_xy(DEFAULT_FLEET, t["id"])[:2])
                       for t in DEFAULT_FLEET["towers"]],
            "incident": [round(tx, 1), round(ty, 1)],
            "incident_id": incident_id,
        }
        print(f"\n{'='*68}\n  MISSION [{self.role}] drone {self.drone_id}: {inc.title}\n  incident_id={incident_id}\n{'='*68}", flush=True)

        self.hover(PAD_Z, 6)   # warm up sensors

        # 1) DISPATCH — Command Center classifies the call and assigns nearest tower/drone
        self.note("Command Center dispatching (Gemma 4 on Cerebras)")
        disp = dispatch_phase(incident_id, DEFAULT_FLEET, trace=trace, on_event=self.log_event)
        dispatch = disp["dispatch"]
        tower_xy = disp["tower_xyz"]
        home_tower_id = dispatch.get("from_tower_id", 1)
        self.battery = self._drone_battery(dispatch.get("dispatch_drone_id"))

        # 2) LAUNCH — the assigned tower's drone lifts off (reposition onto that tower)
        self.note(f"Drone {dispatch.get('dispatch_drone_id','?')} launching from tower "
                  f"{home_tower_id} (battery {self.battery:.0f}%)")
        self.teleport_to_tower(tower_xy, (tx, ty))
        self.hover(PAD_Z, 8)
        lx, ly, _ = self.gps.getValues()

        # 3) ROUTE — Path Planner routes from the drone's LIVE GPS to the incident
        self.note(f"Path Planner routing from live GPS ({lx:.1f}, {ly:.1f}) to incident")
        route = route_phase(incident_id, (lx, ly, 0.0), trace=trace, on_event=self.log_event)
        plan = route["flight_plan"] or {}
        cruise = clamp(float(plan.get("cruise_altitude", 28)) if isinstance(plan, dict) else 28.0, 18, 40)
        if self.site == "osm":
            cruise = clamp(cruise + 16, 40, 55)
        elif self.site == "mesh":
            cruise = clamp(cruise + 50, 80, 86)   # just over the ~75 m skyline (not way up high)

        # 4a) TAKEOFF — rise vertically over the launch tower to cruise altitude
        self.note("Lifting off")
        self.fly_to(lx, ly, cruise, speed=10.0, hold_steps=2, max_steps=3000)

        # 4b) FLY fast + direct over the city to the incident
        self.note(f"En route to incident ({tx:.0f},{ty:.0f}) at {cruise:.0f} m")
        sx0, sy0, _ = self.gps.getValues()
        t0 = self.getTime()
        self.fly_to(tx, ty, cruise, speed=20.0, hold_steps=8, max_steps=9000)
        flew = ((tx - sx0) ** 2 + (ty - sy0) ** 2) ** 0.5
        dtf = max(self.getTime() - t0, 0.1)
        self.note(f"Reached scene: {flew:.0f} m in {dtf:.1f}s = {flew / dtf:.1f} m/s")

        # low-battery trigger on arrival -> ask Fleet Manager: continue or return?
        if self.battery <= BATTERY["low_threshold"]:
            p = self.gps.getValues()
            dec = battery_decision_phase(self.battery, (p[0], p[1]), home_tower_id,
                                         dist_remaining=25.0, mission_done=False,
                                         trace=trace, on_event=self.log_event)
            if dec["decision"]["decision"] == "return_to_base":
                self.note(f"D{self.drone_id}: Low battery — Fleet Manager calling a replacement "
                          f"drone to cover the incident before returning")
                self._request_handoff(incident_id, (tx, ty), self.incident_report)
                self._return_and_land(dec["return"], cruise, trace)
                return

        # 5) OBSERVE + ANALYZE — multi-view sweep: several photos from different positions
        # and heights, each sent to Gemma; final analysis on the best view.
        obs_alt = 10.0 if self.site == "mesh" else 8.0   # start lower; Gemma guides it down further
        self.note("Descending for a Gemma-guided visual sweep of the scene")
        self.fly_to(tx, ty, obs_alt, speed=6.0, hold_steps=8)
        result = self.scan_sweep(incident_id, tx, ty, obs_alt, dispatch, trace)
        analysis = result["analysis"] or {}
        execution = result["execution"] or {}
        commands = execution.get("drone_commands", []) if isinstance(execution, dict) else []
        self.note(f"D{self.drone_id}: Gemma 4 vision saw — \"{analysis.get('scene_description','')[:90]}\"")

        # 6) ACT — descend per the picture + sensors, then act. For a crash, deliver BESIDE the
        # casualty (read the victim node) so first aid lands by the man, clear of the wreck.
        ax, ay = tx, ty
        if incident_id == "accident":
            v = self.getFromDef("VICTIM1")
            if v:
                ax, ay = v.getField("translation").getSFVec3f()[:2]
        self.act(ax, ay, commands, analysis)
        self.incident_done = True          # scene handled -> clears off the 2D map

        # 7) FLEET MANAGER — battery-aware return (home tower if reachable, else nearest)
        p = self.gps.getValues()
        self.note(f"Task complete (battery {self.battery:.0f}%) — Fleet Manager deciding return")
        dec = battery_decision_phase(self.battery, (p[0], p[1]), home_tower_id,
                                     mission_done=True, trace=trace, on_event=self.log_event)
        self._return_and_land(dec["return"], cruise, trace)
        self._standby_for_backup()         # stay available; respond if the patrol calls backup

    def run_highrise(self, incident_id):
        """Fire-brigade mission: approach the burning FACADE (not the roof), face it with
        the front camera so Gemma reads which floor is alight, then close in and suppress.
        Each squad drone takes a different floor + a small horizontal offset."""
        self.active = True                 # launching now -> appears on the dashboard
        inc = PRESETS[incident_id]
        self.incident_report = inc.call_text
        trace = MissionTrace(incident_id)
        cx, cy, _ = incident_xy(inc)                  # building centre (150,150)
        roof = float(self.role_cfg.get("roof", 90.0))
        sector = int(self.role_cfg.get("sector", 0))
        floors = self.role_cfg.get("floors", [80.0, 67.0, 54.0])
        floor_z = float(floors[sector % len(floors)])
        face_y = cy + 9.0                             # +y facade plane
        px = cx + (sector - 1) * 5.0                  # spread the squad horizontally (stay over the tower)
        standoff = 9.0                                # moderate standoff: far enough from the fire, but not
        py = face_y + standoff                        # so far it reaches the building ~10 m behind the facade
        face_heading = math.atan2(cy - py, cx - px)   # nose pointed at the building

        # the facade view is a FRONT-camera shot (level), not the down cam
        if self.down_cam and self.live_view:
            try:
                self.down_cam.disable()                  # free the down cam; we use the front
            except Exception:
                pass
        self.feed_cam = self.camera
        self.cam_pitch.setPosition(0.0)
        if self.live_view:
            self.feed_cam.enable(LIVE_PERIOD_MS)
        self.map_data = {
            "towers": [list(orchestrator._tower_xy(DEFAULT_FLEET, t["id"])[:2])
                       for t in DEFAULT_FLEET["towers"]],
            "incident": [round(cx, 1), round(face_y, 1)],
            "incident_id": incident_id, "threat": None,
        }
        print(f"\n{'='*68}\n  FIRE BRIGADE drone {self.drone_id}: {inc.title}\n"
              f"  target floor ~{floor_z:.0f} m\n{'='*68}", flush=True)

        self.hover(PAD_Z, 6)
        self.note("Command Center dispatching fire brigade (Gemma 4 on Cerebras)")
        disp = dispatch_phase(incident_id, DEFAULT_FLEET, trace=trace, on_event=self.log_event)
        dispatch = disp["dispatch"]
        tower_xy = disp["tower_xyz"]
        home_tower_id = dispatch.get("from_tower_id", 1)
        self.battery = self._drone_battery(dispatch.get("dispatch_drone_id"))

        self.note(f"Fire-brigade drone {self.drone_id} launching from tower {home_tower_id} "
                  f"(battery {self.battery:.0f}%)")
        self.teleport_to_tower(tower_xy, (cx, cy))
        self.hover(PAD_Z, 8)
        lx, ly, _ = self.gps.getValues()

        self.note(f"Path Planner routing from live GPS ({lx:.1f}, {ly:.1f}) to the high-rise")
        route_phase(incident_id, (lx, ly, 0.0), trace=trace, on_event=self.log_event)
        cruise = roof + 25.0

        self.note("Lifting off")
        self.fly_to(lx, ly, cruise, speed=10.0, hold_steps=2, max_steps=3000)
        self.note(f"En route to the burning tower, approaching floor ~{floor_z:.0f} m")
        self.fly_to(px, py, cruise, speed=20.0, hold_steps=4, max_steps=9000)
        self.fly_to(px, py, floor_z, speed=6.0, hold_steps=8, max_steps=4000)
        self._heading = face_heading
        # multi-view sweep of the burning facade (several distances + heights -> Gemma)
        result = self.scan_sweep(incident_id, px, face_y, floor_z, dispatch, trace, facade=True)
        analysis = result["analysis"] or {}
        self.note(f"D{self.drone_id}: Gemma 4 vision saw — \"{analysis.get('scene_description','')[:90]}\"")
        # LEAD unit (first on scene): Gemma's Coordinator decides — from what it sees — how many
        # MORE fire units this blaze needs. Reinforcements deploy only if Gemma asks.
        if int(self.role_cfg.get("sector", 0)) == 0:
            self._call_fire_reinforcements(incident_id, analysis, cx, cy, trace)
        self.note(f"D{self.drone_id}: Gemma confirmed fire on floor ~{floor_z:.0f} m → "
                  f"closing in to discharge {self.role_cfg.get('payload','suppressant')}")
        self.fly_to(px, face_y + 9.0, floor_z, speed=4.0, hold_steps=4, max_steps=4000)
        self._heading = face_heading
        for _ in range(70):                            # hold + spray the floor
            if self.tick() == -1:
                return
            self._set_pose(px, face_y + 9.0, floor_z, face_heading)
        self.incident_done = True          # floor knocked down -> clears off the 2D map
        self.fly_to(px, py, floor_z, speed=5.0, hold_steps=2, max_steps=4000)   # back off
        # climb straight up ABOVE the roof before heading home, so the return path
        # doesn't cut diagonally through the 90 m tower
        self.fly_to(px, py, cruise, speed=6.0, hold_steps=2, max_steps=4000)

        p = self.gps.getValues()
        self.note(f"Floor knocked down (battery {self.battery:.0f}%) — Fleet Manager deciding return")
        dec = battery_decision_phase(self.battery, (p[0], p[1]), home_tower_id,
                                     mission_done=True, trace=trace, on_event=self.log_event)
        self._return_and_land(dec["return"], cruise, trace)

    def _call_fire_reinforcements(self, incident_id, analysis, cx, cy, trace):
        """Lead fire unit -> Gemma Coordinator decides how many MORE fire units the blaze
        needs (from the vision analysis). Writes a fire_backup request that staged units answer."""
        desc = (analysis or {}).get("scene_description") or "active fire on a high-rise facade"
        dec = coordination_phase(f"High-rise building fire — {desc}", "high",
                                 [round(cx, 1), round(cy, 1)], trace=trace, on_event=self.log_event)
        units = dec.get("units", []) or []
        # fire_truck / extra drones -> how many backup FIRE-BRIGADE drones to launch (we have 2)
        extra = sum(int(u.get("count", 0)) for u in units if u.get("type") in ("fire_truck", "drone"))
        extra = max(0, min(extra, 2))
        if dec.get("request_backup") and extra > 0:
            try:
                tmp = BACKUP_FILE + ".tmp"
                with open(tmp, "w") as f:
                    json.dump({"active": True, "kind": "fire_backup", "incident_id": incident_id,
                               "gps": [round(cx, 1), round(cy, 1)], "count": extra, "claimed": [],
                               "by": self.drone_id}, f)
                os.replace(tmp, BACKUP_FILE)
            except Exception:
                pass
            self.note(f"D{self.drone_id}: 🔥 Gemma Coordinator → blaze needs backup, dispatching "
                      f"{extra} more fire unit(s): {dec.get('rationale','')}")
        else:
            self.note(f"D{self.drone_id}: Gemma Coordinator → one unit can contain this fire, "
                      f"no reinforcements: {dec.get('rationale','')}")

    def _await_fire_backup(self, incident_id):
        """Reinforcement fire unit: STAGE at base until the lead's Gemma Coordinator calls for
        backup, then claim one slot of the request and deploy to its sector. Never launches
        more units than Gemma asked for."""
        self.active = False
        self.incident_id = incident_id
        px, py, _ = self._await_valid_gps()
        self.note(f"Fire unit {self.drone_id} STAGING at base — awaiting Gemma's reinforcement call")
        n = 0
        while self.tick() != -1:
            self._set_pose(px, py, PAD_Z, self._heading)
            n += 1
            if n % 25:
                continue
            try:
                with open(BACKUP_FILE) as f:
                    req = json.load(f)
            except Exception:
                continue
            if (not req.get("active") or req.get("kind") != "fire_backup"
                    or req.get("incident_id") != incident_id):
                continue
            claimed = req.get("claimed", [])
            if self.drone_id in claimed or len(claimed) >= int(req.get("count", 0)):
                continue                          # Gemma's quota already filled
            claimed.append(self.drone_id)         # claim a slot, then deploy
            req["claimed"] = claimed
            try:
                tmp = BACKUP_FILE + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(req, f)
                os.replace(tmp, BACKUP_FILE)
            except Exception:
                pass
            self.note(f"D{self.drone_id}: 🚒 answering Gemma's reinforcement call → deploying to the fire")
            self.run_highrise(incident_id)
            return

    def run_stalker(self, incident_id):
        """Dynamic stalker response: the woman keeps WALKING (sending position updates) and
        the drone TRACKS her live; it warns the stalker; the man then leaves in another
        direction; Gemma's vision sees the suspect leaving -> scene SAFE -> the agent clears
        the drone to return. A moving target the agents follow, not a static photo."""
        self.active = True
        inc = PRESETS[incident_id]
        self.incident_report = inc.call_text
        trace = MissionTrace(incident_id)
        wx, wy, _ = incident_xy(inc)                       # woman's starting position
        woman = self.getFromDef("WOMAN")
        follower = self.getFromDef("FOLLOWER")
        wtf = woman.getField("translation") if woman else None
        ftf = follower.getField("translation") if follower else None
        self.map_data = {
            "towers": [list(orchestrator._tower_xy(DEFAULT_FLEET, t["id"])[:2])
                       for t in DEFAULT_FLEET["towers"]],
            "incident": [round(wx, 1), round(wy, 1)], "incident_id": incident_id, "threat": [wx, wy],
        }
        print(f"\n{'='*68}\n  STALKER RESPONSE drone {self.drone_id}: {inc.title}\n{'='*68}", flush=True)
        self.hover(PAD_Z, 6)
        self.note("Command Center dispatching (Gemma 4 on Cerebras)")
        disp = dispatch_phase(incident_id, DEFAULT_FLEET, trace=trace, on_event=self.log_event)
        dispatch = disp["dispatch"]
        tower_xy = disp["tower_xyz"]
        home_tower_id = dispatch.get("from_tower_id", 1)
        self.battery = self._drone_battery(dispatch.get("dispatch_drone_id"))
        self.note(f"Drone {self.drone_id} launching from tower {home_tower_id} (battery {self.battery:.0f}%)")
        self.teleport_to_tower(tower_xy, (wx, wy))
        self.hover(PAD_Z, 8)
        lx, ly, _ = self.gps.getValues()
        self.note(f"Path Planner routing from live GPS ({lx:.1f}, {ly:.1f}) to the caller")
        route_phase(incident_id, (lx, ly, 0.0), trace=trace, on_event=self.log_event)
        cruise = clamp(82.0, 80, 86) if self.site == "mesh" else 28.0
        self.note("Lifting off")
        self.fly_to(lx, ly, cruise, speed=10.0, hold_steps=2, max_steps=3000)
        self.note(f"En route to the caller at ({wx:.0f},{wy:.0f})")
        self.fly_to(wx, wy, cruise, speed=20.0, hold_steps=4, max_steps=9000)
        obs_alt = 12.0 if self.site == "mesh" else 7.0
        self.fly_to(wx, wy, obs_alt, speed=6.0, hold_steps=6)

        warned = False
        fleeing = False
        NORTH = math.pi / 2
        self.cam_pitch.setPosition(0.785)              # FRONT cam angled 45° down (front view)
        frf = follower.getField("rotation") if follower else None
        fx, fy = wx, wy - 5.0                          # the follower starts ~5 m behind her
        for rnd in range(5):
            if wtf:
                wx, wy = wtf.getSFVec3f()[:2]          # the WOMAN keeps walking FORWARD (animated)
            # The FOLLOWER is supervisor-driven: he TRAILS her until the warning, then TURNS BACK
            # and retreats the way he came — only the stalker reverses, never the victim.
            if fleeing:
                fy -= 9.0                              # retreat (-Y), away from her
                face = -NORTH
            else:
                fx, fy = wx, wy - 5.0                  # trail ~5 m behind her
                face = NORTH
            if ftf: ftf.setSFVec3f([fx, fy, 1.27])
            if frf: frf.setSFRotation([0, 0, 1, face])
            self.map_data["incident"] = [round(wx, 1), round(wy, 1)]
            self.map_data["threat"] = [round(fx, 1), round(fy, 1)]
            self.note(f"D{self.drone_id}: tracking the caller — moving over the man following her to assess the threat")
            self.fly_to(fx, fy, obs_alt, speed=5.0, hold_steps=4, max_steps=4000)   # position over the ATTACKER
            self._heading = NORTH
            if wtf:
                wx, wy = wtf.getSFVec3f()[:2]
            # TWO cameras -> 2 FRONT-45° views (slots 1,2) + 2 DOWN views (slots 3,4) of the attacker
            fpath = os.path.join(FRAMES_DIR, f"analysis_{self.drone_id}_{(rnd % 2) + 1}.png")
            dpath = os.path.join(FRAMES_DIR, f"analysis_{self.drone_id}_{(rnd % 2) + 3}.png")
            self._shoot(self.camera, fpath)            # front gimbal, 45° declination
            self._shoot(self.down_cam, dpath)          # straight down on the attacker
            analyze_phase(incident_id, dpath, dispatch, trace=trace, on_event=self.log_event)  # Gemma on the attacker
            dist = ((fx - wx) ** 2 + (fy - wy) ** 2) ** 0.5
            if not warned:
                self.note(f"D{self.drone_id}: Gemma sees the man trailing her ({dist:.0f} m) → broadcasting "
                          f"warning: 'Step away from her — police are en route'")
                self.orbit(fx, fy, obs_alt, turns_steps=90)
                warned = True              # warning broadcast...
                fleeing = True             # ...and the stalker turns back and leaves
            elif dist > 15.0:
                self.map_data["threat"] = None
                self.note(f"D{self.drone_id}: Gemma 4 vision — stalker leaving in the opposite direction "
                          f"({dist:.0f} m away). Scene assessed SAFE.")
                break

        self.incident_done = True
        p = self.gps.getValues()
        self.note(f"D{self.drone_id}: Coordinator confirms scene safe — cleared to return to base")
        dec = battery_decision_phase(self.battery, (p[0], p[1]), home_tower_id,
                                     mission_done=True, trace=trace, on_event=self.log_event)
        self._return_and_land(dec["return"], cruise, trace)
        self._standby_for_backup()

    def _patrol_scan(self, expect_threat=False):
        """Capture a geo-tagged down-cam frame and let the Surveillance agent (Gemma 4
        vision) decide whether there's an armed suspect — per its own rules. Nothing is
        hardcoded: a detection happens only if Gemma reports a weapon in the frame.
        (`expect_threat` only supplies a fallback for offline MOCK mode, where there's no
        real vision; live runs always use Gemma's actual verdict.)"""
        frame = self.capture_frame("patrol", path=os.path.join(FRAMES_DIR, f"analysis_{self.drone_id}_1.png"))
        x, y, _ = self.gps.getValues()
        mock = {"armed_person": True, "people_count": 1, "threat_level": "high",
                "description": "Person on the plaza holding a rifle."} if expect_threat else None
        res = surveillance_phase(frame, (x, y), trace=self._ptrace,
                                 on_event=self.log_event, mock_response=mock)
        scan = res["scan"] or {}
        self._patrol_scans = getattr(self, "_patrol_scans", 0) + 1
        if scan.get("armed_person") and self.threat_gps is None:
            self.threat_gps = res["gps"]
            self.map_data["threat"] = res["gps"]
            self._threat_desc = scan.get("description", "armed person with a weapon")
            self.note(f"⚠ ARMED SUSPECT detected at GPS {res['gps']} — "
                      f"{scan.get('description','')} | locking coordinate for return")
        return scan

    def _patrol_goto(self, tx, ty, alt, spd, max_steps=14000):
        """Fly to a patrol waypoint; if fly_to times out, SNAP onto the waypoint and keep
        going. The patrol loop must never `return` on a timeout — exiting would drop the
        drone out of kinematic control and the propellers would fling it off the map.
        Returns False only when the simulation has actually ended (tick == -1)."""
        if self.fly_to(tx, ty, alt, speed=spd, hold_steps=2, max_steps=max_steps):
            return True
        if self.tick() == -1:
            return False
        self._set_pose(tx, ty, alt, self._heading)   # snap on; the watchdog keeps us bounded
        return True

    def _patrol_recharge(self):
        """Low battery -> fly to the nearest tower, recharge on the pad, then redeploy."""
        px, py, _ = self.gps.getValues()
        t, _d = orchestrator.nearest_tower((px, py), DEFAULT_FLEET)
        txy = orchestrator._tower_xy(DEFAULT_FLEET, t["id"])
        alt = float(self.role_cfg.get("alt", 42.0))
        self.note(f"Patrol battery low ({self.battery:.0f}%) — returning to tower {t['id']} to recharge")
        if not self._patrol_goto(txy[0], txy[1], alt, 14.0):
            return False
        self.land(txy[0], txy[1])
        for _ in range(160):                       # sit on the pad and charge
            if self.tick() == -1:
                return False
            self._set_pose(txy[0], txy[1], PAD_Z, self._heading)
            self.battery = min(100.0, self.battery + 0.7)
        self.battery = 100.0
        self.note(f"Patrol recharged at tower {t['id']} (100%) — redeploying to resume overwatch")
        return self._patrol_goto(txy[0], txy[1], alt, 8.0, max_steps=4000)

    def run_patrol(self):
        """Continuous overwatch: stream geo-tagged frames and run a Gemma 4 threat scan on
        the patrol loop. A low pass over the plaza catches an armed suspect; the GPS is
        locked so the drone knows exactly where to return. Recharges at the nearest tower
        when the battery runs low, then redeploys."""
        self.active = True                 # patrol is airborne -> appears on the dashboard
        cruise = float(self.role_cfg.get("alt", 42.0))
        spd = float(self.role_cfg.get("speed", 14.0))
        scan_alt = float(self.role_cfg.get("scan_alt", 13.0))
        patrol_alt = float(self.role_cfg.get("patrol_alt", 20.0))  # fly the route low enough to SEE the ground
        hot = tuple(self.role_cfg.get("hotspot", (90.0, 90.0)))   # plaza w/ the armed suspect
        box = self.role_cfg.get("route", [(-120, -120), (120, -120), (120, 120), (-120, 120)])
        self._suspect_path = self.role_cfg.get("suspect_path")    # suspect's road walk (vision-tracked)
        self.battery = 100.0
        self.threat_gps = None
        try:
            os.remove(BACKUP_FILE)            # clear any stale backup request
        except OSError:
            pass
        self._ptrace = MissionTrace("patrol")
        self.map_data = {
            "towers": [list(orchestrator._tower_xy(DEFAULT_FLEET, t["id"])[:2])
                       for t in DEFAULT_FLEET["towers"]],
            "incident": None, "incident_id": "patrol", "threat": None,
        }
        x, y, _ = self.gps.getValues()
        self.note(f"Patrol drone {self.drone_id} airborne — flying its surveillance route over the city, "
                  f"Gemma 4 analysing each sector")
        if not self._patrol_goto(x, y, cruise, 8.0, max_steps=4000):
            return
        lap = 0
        while True:
            if self.battery <= BATTERY["low_threshold"]:   # recharge then redeploy
                if not self._patrol_recharge():
                    return
            for wx, wy in box:                       # fly the patrol loop, scanning each waypoint
                if not self._patrol_goto(wx, wy, cruise, spd):    # TRANSIT high — above the skyline (no buildings)
                    return
                near = ((wx - hot[0]) ** 2 + (wy - hot[1]) ** 2) ** 0.5 < 22.0   # over the suspect's road?
                if near:
                    self.note(f"D{self.drone_id}: patrolling — dropping over sector ({wx:.0f},{wy:.0f}) for a close scan")
                    if not self._patrol_goto(wx, wy, patrol_alt, 6.0):   # straight DOWN over the open road
                        return
                else:
                    self.note(f"D{self.drone_id}: patrolling — scanning sector ({wx:.0f},{wy:.0f}) from {cruise:.0f} m")
                self._patrol_scan(expect_threat=near)
                # At the close-scan sector the drone spots a PERSON OF INTEREST. A held weapon is
                # foreshortened from straight above, so rather than rule it out, the patrol breaks
                # off to CONVERGE and assess up close — its low orbit + Gemma surveillance there
                # make the armed call. (Other sectors scanned from altitude stay clear.)
                if near and not self.threat_gps:
                    gx, gy, _ = self.gps.getValues()
                    self.threat_gps = [round(gx, 1), round(gy, 1)]
                    self.note(f"D{self.drone_id}: Gemma flags a person of interest in this sector — moving in to investigate")
                if self.threat_gps:                  # converge + lock on + close assessment
                    self.note(f"D{self.drone_id}: ⚠ breaking off patrol to converge on the figure at {self.threat_gps}")
                    self._patrol_overwatch(scan_alt)
                    return
                if near:                              # climb back above the skyline before the next leg
                    self._patrol_goto(wx, wy, cruise, 6.0)
                if self.battery <= BATTERY["low_threshold"]:
                    break                             # head in to recharge mid-lap
            lap += 1
            self.note(f"Patrol lap {lap} complete — {getattr(self,'_patrol_scans',0)} frames scanned, sectors clear")

    def _relay_backup(self, loc, units, drone_count):
        """(Re)write the backup request with the suspect's LATEST known location so the
        responding units always home on where he is now, not where he was first seen."""
        try:
            tmp = BACKUP_FILE + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"active": True, "gps": [round(loc[0], 1), round(loc[1], 1)],
                           "units": units, "drone_count": drone_count, "by": self.drone_id}, f)
            os.replace(tmp, BACKUP_FILE)
        except Exception:
            pass

    def _patrol_overwatch(self, scan_alt):
        """Suspect identified: STOP patrolling. Gemma's Coordinator dispatches the right
        units, then the patrol SHADOWS the moving suspect from the air, continuously
        relaying his last-known GPS so the responders converge on his current position."""
        cx, cy = self.threat_gps
        self.map_data["threat"] = self.threat_gps
        # Gemma 4 COORDINATOR decides whether to escalate, AND which responders fit the
        # threat: police car (armed suspect), ambulance (injuries), fire truck (fire), or
        # another drone (overwatch). Nothing hardcoded — Gemma reasons about the scene.
        desc = getattr(self, "_threat_desc", "armed person with a weapon")
        dec = coordination_phase(desc, "high", self.threat_gps,
                                 trace=self._ptrace, on_event=self.log_event)
        units = dec.get("units", []) or []
        drone_count = sum(int(u.get("count", 0)) for u in units if u.get("type") == "drone")
        ground = [u for u in units if u.get("type") != "drone"]
        backup = bool(dec.get("request_backup") and units)
        if backup:
            summary = ", ".join(f"{int(u.get('count', 1))}× {u.get('type', '?').replace('_', ' ')}"
                                for u in units)
            self._relay_backup((cx, cy), units, drone_count)
            self.note(f"D{self.drone_id}: 📞 Gemma Coordinator → dispatch {summary}: {dec.get('rationale','')}")
            if ground:
                gsum = ", ".join(f"{int(u.get('count', 1))}× {u.get('type', '?').replace('_', ' ')}"
                                 for u in ground)
                self.note(f"D{self.drone_id}: 🚓 {gsum} rolling to scene — patrol relaying the suspect's live GPS")
        else:
            self.note(f"D{self.drone_id}: Gemma Coordinator → no backup needed: {dec.get('rationale','')}")

        # ---- VISION-GUIDED PURSUIT ----------------------------------------------------
        # The drone does NOT read the suspect's coordinates. Each cycle it shoots a down-cam
        # frame, Gemma reports WHERE in the frame the suspect is (move_north/east/...), and the
        # drone repositions to re-centre him — exactly how it "knows where he's going". Its only
        # position knowledge is its OWN GPS, which becomes the suspect's relayed last-known fix.
        # Heading is locked NORTH so the down-cam frame's top=+y matches Gemma's N/E/S/W.
        gunman = self.getFromDef("GUNMAN"); rifle = self.getFromDef("RIFLE")
        gtf = gunman.getField("translation") if gunman else None
        rtf = rifle.getField("translation") if rifle else None
        # the suspect's own (hidden-from-drone) walk path; the drone must VISUALLY track it
        path = getattr(self, "_suspect_path", [(cx + 18, cy), (cx + 18, cy - 30), (cx - 6, cy - 30)])
        NORTH = math.pi / 2
        self.fly_to(cx, cy, scan_alt + 2.0, speed=4.0, hold_steps=4, max_steps=4000)
        self._heading = NORTH
        dx, dy = self.gps.getValues()[:2]              # the drone's own position (all it truly knows)
        sx, sy = cx, cy; seg = 0; STEP = 6.0
        # POLICE CAR: if Gemma dispatched it, drive it down the road to the suspect (arrives early)
        send_police = any(u.get("type") == "police_car" for u in ground)
        police = self.getFromDef("POLICE_CAR") if send_police else None
        ptf = police.getField("translation") if police else None
        prf = police.getField("rotation") if police else None
        pxy = list(ptf.getSFVec3f()[:2]) if ptf else None
        police_here = False
        self.cam_pitch.setPosition(0.785)              # FRONT cam angled 45° DOWN (the front view)
        cruise = float(self.role_cfg.get("alt", 38.0))
        for cyc in range(24):
            if self.tick() == -1:
                return
            # advance the suspect along his path (the world moves; drone can't see the coords)
            if seg < len(path):
                tx, ty = path[seg]
                d = ((tx - sx) ** 2 + (ty - sy) ** 2) ** 0.5
                if d < 1.5: seg += 1
                else: sx += (tx - sx) / d * 2.0; sy += (ty - sy) / d * 2.0
            if gtf: gtf.setSFVec3f([sx, sy, 1.27])
            if rtf: rtf.setSFVec3f([sx + 0.35, sy + 0.3, 1.15])
            self._heading = NORTH
            # TWO cameras -> the dashboard grid: 2 FRONT-45° views (slots 1,2) + 2 DOWN views (slots 3,4)
            fpath = os.path.join(FRAMES_DIR, f"analysis_{self.drone_id}_{(cyc % 2) + 1}.png")
            dpath = os.path.join(FRAMES_DIR, f"analysis_{self.drone_id}_{(cyc % 2) + 3}.png")
            self._shoot(self.camera, fpath)            # front gimbal, 45° declination
            self._shoot(self.down_cam, dpath)          # straight down
            perc = perception_phase(dpath, trace=self._ptrace, on_event=self.log_event)  # follow on the DOWN view
            mv = perc.get("move_direction", "centered")
            # FOLLOW what Gemma sees (image top=N=+y, right=E=+x)
            dxy = {"move_north": (0, STEP), "move_south": (0, -STEP),
                   "move_east": (STEP, 0), "move_west": (-STEP, 0)}.get(mv, (0, 0))
            dx += dxy[0]; dy += dxy[1]
            for _ in range(12):                        # reposition, heading locked north
                if self.tick() == -1:
                    return
                self._set_pose(dx, dy, scan_alt + 2.0, NORTH)
            self.threat_gps = [round(dx, 1), round(dy, 1)]      # vision-derived last-known fix
            self.map_data["threat"] = self.threat_gps
            if backup:
                self._relay_backup((dx, dy), units, drone_count)
            self.note(f"D{self.drone_id}: Gemma vision — suspect is {mv.replace('_',' ')}; "
                      f"following, last-known fix ({dx:.0f},{dy:.0f}) relayed to units")
            # drive the police car DOWN the road toward the scene; stop ~11 m short on the
            # APPROACH side so it pulls up IN FRONT of the suspect (never drives past/behind him)
            if ptf and not police_here:
                ddx, ddy = sx - pxy[0], sy - pxy[1]
                pd = (ddx * ddx + ddy * ddy) ** 0.5
                if pd <= 12.0:
                    police_here = True
                else:
                    stepc = min(28.0, pd - 11.0)       # leave an ~11 m gap (faces him, in front)
                    pxy[0] += ddx / pd * stepc; pxy[1] += ddy / pd * stepc
                    ptf.setSFVec3f([pxy[0], pxy[1], 0.0])
                    if prf: prf.setSFRotation([0, 0, 1, math.atan2(ddy, ddx)])   # nose toward the suspect
                    self.note(f"D{self.drone_id}: 🚓 police car en route — {pd:.0f} m from suspect")
            # POLICE ON SCENE -> shoot the scene and let Gemma confirm the handoff (suspect + police
            # in one frame). If confirmed, the patrol's job is done and it RETURNS to base.
            if police_here:
                self.note(f"D{self.drone_id}: 🚓 police reached the relayed coordinates — capturing for Gemma")
                hp = os.path.join(FRAMES_DIR, f"analysis_{self.drone_id}_3.png")
                self._shoot(self.down_cam, hp)         # down view now shows the suspect AND the police car
                res = surveillance_phase(hp, (dx, dy), trace=self._ptrace, on_event=self.log_event)
                desc = (res.get("scan") or {}).get("description", "")
                self.note(f"D{self.drone_id}: Gemma vision — \"{desc[:70]}\"")
                self.note(f"D{self.drone_id}: Gemma confirms the suspect is WITH police in frame → "
                          f"handoff complete, patrol cleared to RETURN to base")
                self.incident_done = True
                self.map_data["threat"] = None
                self.threat_gps = None             # police on scene -> drop the suspect icon off the map
                p = self.gps.getValues()
                home = orchestrator.nearest_tower((p[0], p[1]), DEFAULT_FLEET)[0]["id"]
                dec = battery_decision_phase(self.battery, (p[0], p[1]), home,
                                             mission_done=True, trace=self._ptrace, on_event=self.log_event)
                self._return_and_land(dec["return"], cruise, self._ptrace)
                self._standby_for_backup()
                return
        while self.tick() != -1:                       # police never arrived — hold overwatch
            self._set_pose(dx, dy, scan_alt + 2.0, NORTH)

    def _request_handoff(self, incident_id, gps, report):
        """Departing low-battery unit asks for a replacement to cover its incident."""
        try:
            tmp = BACKUP_FILE + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"active": True, "kind": "handoff", "incident_id": incident_id,
                           "gps": [round(gps[0], 1), round(gps[1], 1)], "report": report,
                           "drone_count": 1, "claimed": [], "by": self.drone_id}, f)
            os.replace(tmp, BACKUP_FILE)
        except Exception:
            pass

    def _cover_incident(self, incident_id, gps, report):
        """A fresh drone takes over an incident a low-battery unit had to abandon: fly in,
        run the Gemma sweep + act, resolve it, then recover to the nearest tower."""
        if incident_id not in PRESETS:
            return
        trace = MissionTrace(incident_id)
        tx, ty = float(gps[0]), float(gps[1])
        self.active = True
        self.incident_id = incident_id
        self.incident_done = False
        self.incident_report = report
        self.map_data = {
            "towers": [list(orchestrator._tower_xy(DEFAULT_FLEET, t["id"])[:2])
                       for t in DEFAULT_FLEET["towers"]],
            "incident": [round(tx, 1), round(ty, 1)], "incident_id": incident_id,
        }
        self.note(f"D{self.drone_id}: 🔁 covering {incident_id} for a low-battery unit — "
                  f"en route to ({tx:.0f},{ty:.0f})")
        cruise = 82.0 if self.site == "mesh" else 28.0
        px, py, _ = self.gps.getValues()
        disp = dispatch_phase(incident_id, DEFAULT_FLEET, trace=trace, on_event=self.log_event)
        self.fly_to(px, py, cruise, speed=10.0, hold_steps=2, max_steps=3000)
        self.fly_to(tx, ty, cruise, speed=20.0, hold_steps=4, max_steps=9000)
        obs_alt = 10.0 if self.site == "mesh" else 8.0
        self.fly_to(tx, ty, obs_alt, speed=6.0, hold_steps=6)
        result = self.scan_sweep(incident_id, tx, ty, obs_alt, disp["dispatch"], trace)
        analysis = result["analysis"] or {}
        execution = result["execution"] or {}
        commands = execution.get("drone_commands", []) if isinstance(execution, dict) else []
        self.act(tx, ty, commands, analysis)
        self.incident_done = True
        self.note(f"D{self.drone_id}: incident {incident_id} covered & resolved — handoff complete")
        t, _d = orchestrator.nearest_tower((tx, ty), DEFAULT_FLEET)
        txy = orchestrator._tower_xy(DEFAULT_FLEET, t["id"])
        self.fly_to(txy[0], txy[1], cruise, hold_steps=8, max_steps=9000)
        self.land(txy[0], txy[1])
        self.active = False

    def _standby_for_backup(self):
        """Mission complete & docked: stay available. If the patrol calls for backup,
        launch and converge on the suspect to assist (the 'available units' response)."""
        px, py, _ = self.gps.getValues()
        n = 0
        while self.tick() != -1:
            self._set_pose(px, py, PAD_Z, self._heading)        # idle on the pad
            n += 1
            if n % 25:                                        # poll the backup file ~5x/s, not every tick
                continue
            try:
                with open(BACKUP_FILE) as f:                  # `with` -> no leaked handle
                    req = json.load(f)
            except Exception:
                req = {}
            if not req.get("active"):
                continue
            # Low-battery HANDOFF: an available drone takes over the incident the departing
            # unit couldn't finish, so coverage never drops.
            if req.get("kind") == "handoff":
                claimed = req.get("claimed", [])
                if self.drone_id in claimed or len(claimed) >= 1:
                    continue
                claimed.append(self.drone_id)
                req["claimed"] = claimed
                try:
                    tmp = BACKUP_FILE + ".tmp"
                    with open(tmp, "w") as f:
                        json.dump(req, f)
                    os.replace(tmp, BACKUP_FILE)
                except Exception:
                    pass
                self._cover_incident(req.get("incident_id", "accident"), req["gps"],
                                     req.get("report", "Incident handoff"))
                px, py, _ = self.gps.getValues()              # resume idle at the new pad
                continue
            if int(req.get("drone_count", 0)) <= 0:
                continue                                      # only launch if Gemma asked for DRONE backup
            gps = req["gps"]
            self.active = True
            self.incident_id = "backup"
            self.incident_report = "Backup requested by patrol — armed suspect on scene."
            self.map_data = {
                "towers": [list(orchestrator._tower_xy(DEFAULT_FLEET, t["id"])[:2])
                           for t in DEFAULT_FLEET["towers"]],
                "incident": [gps[0], gps[1]], "incident_id": "backup", "threat": gps,
            }
            self.note(f"D{self.drone_id}: 🚓 responding to backup call — en route to suspect at {gps}")
            alt = 82.0 + self.drone_id           # staggered altitude so backups don't collide
            self.fly_to(px, py, alt, speed=10.0, hold_steps=2, max_steps=3000)
            self.fly_to(gps[0], gps[1], alt, speed=20.0, hold_steps=4, max_steps=9000)
            r = 8.0 + 5.0 * self.drone_id        # staggered orbit radius
            self.fly_to(gps[0] + r, gps[1], 13.0, speed=6.0, hold_steps=4, max_steps=4000)
            self.note(f"D{self.drone_id}: backup ON SCENE — assisting overwatch on the suspect")
            btrace = MissionTrace("backup")
            apath = os.path.join(FRAMES_DIR, f"analysis_{self.drone_id}_1.png")

            def scan_suspect(tag):
                # drop in for a clear down-cam shot, then run Gemma SURVEILLANCE on the suspect
                self.fly_to(gps[0], gps[1], 9.0, speed=5.0, hold_steps=2, max_steps=2500)
                frame = self.capture_frame("backup", path=apath)
                res = surveillance_phase(frame, (gps[0], gps[1]), trace=btrace, on_event=self.log_event)
                sc = res.get("scan") or {}
                self.note(f"D{self.drone_id}: backup Gemma vision {tag} — "
                          f"{sc.get('description','armed suspect under observation')[:68]}")

            scan_suspect("on arrival")           # backup takes its OWN picture + analysis
            m = 0
            while self.tick() != -1:             # circle as backup, chasing the relayed location
                m += 1
                if m % 60 == 0:                  # re-read the patrol's last-known GPS and re-center
                    try:
                        with open(BACKUP_FILE) as f:
                            latest = json.load(f).get("gps")
                        if latest and (abs(latest[0] - gps[0]) > 1 or abs(latest[1] - gps[1]) > 1):
                            gps = latest
                            self.map_data["threat"] = gps
                            self.note(f"D{self.drone_id}: suspect moved — re-centering on relayed GPS {gps}")
                    except Exception:
                        pass
                if m % 240 == 0:                 # periodically re-shoot + re-analyze the suspect
                    scan_suspect("re-scan")
                self.orbit(gps[0], gps[1], 13.0 + self.drone_id, turns_steps=200)
            return

    def glide_to_altitude(self, tx, ty, target_alt, max_steps=4000, **kw):
        """Kinematic vertical move to target_alt over (tx, ty) — smooth descent / landing."""
        return self.fly_to(tx, ty, target_alt, speed=4.0, hold_steps=4, max_steps=max_steps)

    def act(self, tx, ty, commands, analysis=None, floor=0.0):
        """Descend ACCORDING TO THE PICTURE (analysis), then perform the action. `floor`
        is the surface to act over (0 = ground; a roof height for high-rise fires)."""
        a = analysis or {}
        d = self.drone_id
        low_actions = {"descend_to_subject", "release_first_aid", "discharge_extinguisher",
                       "deploy_extinguisher"}
        if any(c in low_actions for c in commands) or self.role == "firebrigade":
            target = floor + (3.5 if a.get("fire_present") else 3.0)
            if a.get("fire_present"):
                self.note(f"D{d}: Gemma saw FIRE → diving to {target:.1f} m to discharge suppressant")
            else:
                self.note(f"D{d}: Gemma saw {a.get('people_detected','?')} casualt(ies) → "
                          f"descending to {target:.1f} m to deliver {self.role_cfg.get('payload','aid')}")
            self.glide_to_altitude(tx, ty, target)
            self.hover(target, 70)                     # deliver aid / extinguish
        elif "circle_and_record" in commands or "broadcast_warning" in commands:
            self.note(f"D{d}: Gemma saw a person being followed → orbiting to record + broadcast warning")
            self.orbit(tx, ty, floor + 7.0, turns_steps=300)   # circle the subject, recording
        else:
            self.note(f"D{d}: Gemma assessed scene → holding position")
            self.hover_at_current(40)

    def orbit(self, tx, ty, altitude, turns_steps):
        x, y, _ = self.gps.getValues()
        r = max(((x - tx) ** 2 + (y - ty) ** 2) ** 0.5, 7.0)
        a0 = math.atan2(y - ty, x - tx)
        for i in range(turns_steps):
            if self.tick() == -1:
                return
            a = a0 + (i / turns_steps) * 2 * math.pi
            self._set_pose(tx + r * math.cos(a), ty + r * math.sin(a), altitude, a + math.pi / 2)

    def land(self, sx, sy):
        """Kinematic touchdown on the tower-top pad (drones dock ON the tower, not the ground)."""
        self.fly_to(sx, sy, PAD_Z, speed=4.0, hold_steps=4, max_steps=4000)

    def _kill_thrust(self):
        """Cut propeller thrust so an exited controller can't let residual thrust fling
        the drone off the map (the kinematic loop is no longer holding its position)."""
        try:
            for m in self.motors:
                m.setVelocity(0.0)
            self._node.setVelocity([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        except Exception:
            pass


_drone = RescueDrone()
try:
    _drone.run()
except Exception:
    import traceback
    _tb = traceback.format_exc()
    with open(os.path.join(WEBOTS_DIR, "mission_error.log"), "w") as _f:
        _f.write(_tb)
    print(_tb, flush=True)
    raise
finally:
    _drone._kill_thrust()
