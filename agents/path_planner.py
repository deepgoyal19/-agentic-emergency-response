"""Agent 2 — Path Planner.

Input:  drone start (tower) position + incident position + no-fly zones.
Output: ordered 3D waypoints, cruise altitude, ETA. Fed to the Webots controller,
        which flies the drone through them. Can be re-invoked mid-flight if Perception
        asks for a reposition to get a clearer camera angle.
"""
from .cerebras_client import gemma, text_content
from .schemas import PATH_SCHEMA

SYSTEM = (
    "You are the PATH PLANNER agent for an emergency drone fleet. Given a start point, "
    "a target incident location, and no-fly zones, produce a short, safe sequence of 3D "
    "waypoints (ascend to cruise altitude, transit, descend over target). Keep it minimal "
    "and fast. Respond ONLY with the structured schema."
)


def run(start_xyz, target_xyz, no_fly_zones=None, mock_response: dict | None = None):
    no_fly_zones = no_fly_zones or []
    user = (
        f"START (tower): {start_xyz}\n"
        f"TARGET (incident): {target_xyz}\n"
        f"NO-FLY ZONES (list of [x,y,radius]): {no_fly_zones}\n\n"
        "Plan the flight now."
    )
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": [text_content(user)]},
    ]
    return gemma.chat(messages, schema=PATH_SCHEMA, mock_response=mock_response)
