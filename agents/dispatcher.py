"""Agent 1 — Dispatcher (Emergency Command Center).

Input:  a 911 call (text) + the fleet/tower state.
Output: classified incident, severity, required payload, and which drone from which
        tower to launch. Hands a typed assignment to the Path Planner.
"""
from .cerebras_client import gemma, text_content
from .schemas import DISPATCH_SCHEMA

SYSTEM = (
    "You are the DISPATCHER agent at an autonomous drone Emergency Command Center. "
    "Given a 911 call and the available drones/charging towers, classify the incident, "
    "rate severity, decide what payload the drone must carry, and assign the nearest "
    "suitable charged drone from the nearest tower. Be decisive and fast — lives depend "
    "on latency. Respond ONLY with the structured schema."
)


def run(call_text: str, fleet_state: dict, mock_response: dict | None = None):
    towers = fleet_state.get("towers", [])
    drones = fleet_state.get("drones", [])
    user = (
        f"911 CALL:\n{call_text}\n\n"
        f"CHARGING TOWERS (id, x, y, charged_drones): {towers}\n"
        f"DRONE FLEET (id, payload, battery%, tower_id): {drones}\n\n"
        "Classify and assign now."
    )
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": [text_content(user)]},
    ]
    return gemma.chat(messages, schema=DISPATCH_SCHEMA, mock_response=mock_response)
