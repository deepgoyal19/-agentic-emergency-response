"""Agent — Fleet / Emergency Coordinator.

When a drone's vision flags a threat from the air, Gemma 4 decides how to ESCALATE:
whether backup is needed, and — crucially — WHICH KIND of responder fits the situation.
An armed suspect needs a POLICE CAR; injuries need an AMBULANCE; a blaze needs a FIRE
TRUCK; aerial tracking/recon needs another DRONE. This is the multi-agent coordination
call: the patrol doesn't blindly call "backup"; the Coordinator reasons about the threat
and dispatches the right units.
"""
from .cerebras_client import gemma, text_content

UNIT_TYPES = ["police_car", "ambulance", "fire_truck", "drone"]

SYSTEM = (
    "You are the EMERGENCY COORDINATOR for a smart-city response unit. A patrol drone has "
    "spotted a situation from the air and you must dispatch the RIGHT responders. Choose "
    "which kinds of units to send and how many of each. Use these rules:\n"
    "- ARMED or violent suspect (gun/rifle/knife in hand) -> send POLICE_CAR(s); HIGH risk. "
    "The reporting patrol drone ALREADY provides aerial overwatch, so do NOT send more drones "
    "for an armed suspect — the ground response is POLICE_CAR(s).\n"
    "- Injured / collapsed people, medical emergency -> send AMBULANCE(s).\n"
    "- Active fire, smoke, burning structure or vehicle -> send FIRE_TRUCK(s).\n"
    "- ONLY send a DRONE if extra AERIAL search/coverage is genuinely needed (rare).\n"
    "If the scene is harmless or uncertain, request no backup. "
    "Respond as JSON: "
    '{"request_backup": bool, '
    '"units": [{"type": one of ["police_car","ambulance","fire_truck","drone"], "count": int}], '
    '"action": str, "rationale": str}. '
    "Keep rationale to one short sentence; units must be [] when request_backup is false."
)

SCHEMA = {
    "name": "backup_decision",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "request_backup": {"type": "boolean"},
            "units": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": UNIT_TYPES},
                        "count": {"type": "integer"},
                    },
                    "required": ["type", "count"],
                    "additionalProperties": False,
                },
            },
            "action": {"type": "string"},
            "rationale": {"type": "string"},
        },
        "required": ["request_backup", "units", "action", "rationale"],
        "additionalProperties": False,
    },
}


def decide(threat_desc: str, threat_level: str, gps, mock_response=None):
    """Gemma 4 decides whether to escalate and which responder units to dispatch."""
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": [text_content(
            f"Patrol drone reports threat_level={threat_level} at GPS {gps}. "
            f"Aerial vision: \"{threat_desc}\". Which units do we dispatch?")]},
    ]
    return gemma.chat(messages, schema=SCHEMA, max_tokens=320,
                      mock_response=mock_response or {
                          "request_backup": True,
                          "units": [{"type": "police_car", "count": 2}],
                          "action": "dispatch_police_patrol_holds_overwatch",
                          "rationale": "Armed suspect is high-risk; dispatch 2 police cars — the patrol drone already holds aerial overwatch."})
