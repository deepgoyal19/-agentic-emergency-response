"""City-life supervisor — ambient traffic + pedestrians.

Kinematically drives a set of cars and civilians around closed loops on the city
roads/sidewalks (sets translation + heading each step). No physics interaction, so it
never destabilises and never collides with the rescue drone. Loops are kept central,
clear of the three incident sites and the drone's flight corridors.
"""
from controller import Supervisor
import json
import math
import os

robot = Supervisor()
dt = int(robot.getBasicTimeStep())

# Match the drone_agent world transform: standard = (0,0) scale 1, OSM = plaza + scale.
_wp = (robot.getWorldPath() or "").lower()
OSM = "osm" in _wp
OX, OY, SC = (-700.0, 290.0, 0.45) if OSM else (0.0, 0.0, 1.0)


def path_length(path):
    pts = path + [path[0]]
    return sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))


def offset_right(path, d=2.4):
    """Shift a closed path to the RIGHT of travel direction so cars sit in a lane
    instead of on the road centreline (and the there-and-back legs end up in
    opposite lanes automatically)."""
    out = []
    n = len(path)
    for i in range(n):
        ax, ay = path[i]
        bx, by = path[(i + 1) % n]
        dx, dy = bx - ax, by - ay
        L = math.hypot(dx, dy) or 1.0
        out.append((ax + (dy / L) * d, ay + (-dx / L) * d))   # right normal
    return out


def point_on(path, s):
    """Position + heading at arc-length s along the closed polyline `path`."""
    pts = path + [path[0]]
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        d = math.dist(a, b)
        if s <= d or i == len(pts) - 2:
            t = s / d if d > 0 else 0.0
            x = a[0] + (b[0] - a[0]) * t
            y = a[1] + (b[1] - a[1]) * t
            return x, y, math.atan2(b[1] - a[1], b[0] - a[0])
        s -= d
    return path[0][0], path[0][1], 0.0


# (DEF name, closed loop waypoints, speed m/s, z height) — speeds kept to a city crawl
CARS = [
    ("TRAFFIC1", [(-70, -3), (70, -3), (70, 3), (-70, 3)], 6, 0.35),
    ("TRAFFIC2", [(70, 3), (-70, 3), (-70, -3), (70, -3)], 5, 0.35),
    ("TRAFFIC3", [(-3, -70), (-3, 70), (3, 70), (3, -70)], 6, 0.35),
    ("TRAFFIC4", [(3, 70), (3, -70), (-3, -70), (-3, 70)], 4, 0.35),
    ("TRAFFIC5", [(-70, 67), (70, 67), (70, 73), (-70, 73)], 5, 0.35),
    ("TRAFFIC6", [(70, 73), (-70, 73), (-70, 67), (70, 67)], 6, 0.35),
]
PEDS = [
    ("CIVIL1", [(30, 14), (46, 14), (46, 26), (30, 26)], 1.4, 1.27),
    ("CIVIL2", [(-46, 12), (-30, 12), (-30, 0), (-46, 0)], 1.2, 1.27),
    ("CIVIL3", [(10, 34), (22, 34), (22, 46), (10, 46)], 1.5, 1.27),
    ("CIVIL4", [(-20, -24), (-8, -24), (-8, -34), (-20, -34)], 1.3, 1.27),
]

ents = []


def add_entity(name, path, speed, z, transform=True):
    node = robot.getFromDef(name)
    if node is None:
        return
    pts = [(x * SC + OX, y * SC + OY) if transform else (x, y) for (x, y) in path]
    L = path_length(pts)
    if L <= 0:
        return
    ents.append({
        "tf": node.getField("translation"),
        "rf": node.getField("rotation"),
        "path": pts, "speed": speed, "z": z, "len": L,
        "s": (L * 0.17 * len(ents)) % L,    # stagger start positions
    })


if OSM:
    # Cars follow the REAL OSM road centrelines near the plaza (absolute coords);
    # civilians stroll the plaza. osm_roads.json is produced by gen_osm road extraction.
    roads_file = os.path.join(os.path.dirname(__file__), "..", "..", "osm_roads.json")
    try:
        car_paths = json.load(open(roads_file))["car_paths"]
    except Exception:
        car_paths = []
    i = 1
    while car_paths:
        node = robot.getFromDef(f"TRAFFIC{i}")
        if node is None:
            break
        add_entity(f"TRAFFIC{i}", offset_right(car_paths[(i - 1) % len(car_paths)]),
                   speed=4 + (i % 3), z=0.35, transform=False)   # ~4-6 m/s, in-lane
        i += 1
    for nm, path, speed, z in PEDS:
        add_entity(nm, path, speed, z, transform=True)
else:
    for nm, path, speed, z in CARS + PEDS:
        add_entity(nm, path, speed, z, transform=True)

print(f"city_life: animating {len(ents)} entities (OSM roads={OSM})", flush=True)

while robot.step(dt) != -1:
    for e in ents:
        e["s"] = (e["s"] + e["speed"] * dt / 1000.0) % e["len"]
        x, y, h = point_on(e["path"], e["s"])
        e["tf"].setSFVec3f([x, y, e["z"]])
        e["rf"].setSFRotation([0, 0, 1, h])
