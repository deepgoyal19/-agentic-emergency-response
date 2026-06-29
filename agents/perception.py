"""Agent 3 — Perception.

The drone's "eyes." Receives live camera frames from the Webots Mavic camera and
decides whether the subject of the incident is clearly framed. If not, it asks the
Path Planner for a reposition (a nice agent-to-agent feedback loop). When the frame
is good, it forwards it to the Analyst.

Uses a fast Gemma 4 vision call (low token budget) for the framing check, so it stays
cheap and quick — the heavy reasoning happens in the Analyst.
"""
from .cerebras_client import gemma, text_content, image_content

SYSTEM = (
    "You are the PERCEPTION agent on an emergency drone, looking straight DOWN at the "
    "scene. Decide if the incident subject (people, vehicle, or fire) is clearly visible "
    "and well centred, and if not, tell the drone which way to move to frame it better. "
    "In the down-camera image the TOP edge is NORTH (+y), bottom is SOUTH, right is EAST "
    "(+x), left is WEST. So a subject near the top -> move_north; a tiny subject -> "
    "descend for a closer look. Also choose target_altitude_m — the right altitude to "
    "INSPECT and ACT on THIS scene, coming DOWN close enough to do the job but NEVER colliding: "
    "a car crash / wreck has a ~1.5 m-tall vehicle, so hold ~3-4 m — close, but safely ABOVE "
    "the wreck; a casualty in the open ~2-3 m; a standing person ~3-4 m; a suppressible fire "
    "(burning vehicle, bike or debris) descend CLOSE ~2-3 m to discharge suppressant. Only a "
    "large, spreading building/structure fire warrants a safe standoff (~7-9 m). Stay low "
    "enough to act, high enough to clear any obstacle. "
    "Respond as JSON: "
    '{"framed": bool, "move_direction": one of '
    '["centered","move_north","move_south","move_east","move_west","descend"], '
    '"target_altitude_m": number (1-12), "reason": str}.'
)

FRAMING_SCHEMA = {
    "name": "framing_check",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "framed": {"type": "boolean"},
            "move_direction": {"type": "string",
                               "enum": ["centered", "move_north", "move_south",
                                        "move_east", "move_west", "descend"]},
            "target_altitude_m": {"type": "number"},
            "reason": {"type": "string"},
        },
        "required": ["framed", "move_direction", "target_altitude_m", "reason"],
        "additionalProperties": False,
    },
}


def check_frame(image_path_or_url: str, mock_response: dict | None = None):
    """Vision gate: is the subject framed, and which way should the drone move?"""
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": [
            text_content("Is the incident subject clearly framed? Which way should the drone move?"),
            image_content(image_path_or_url),
        ]},
    ]
    return gemma.chat(messages, schema=FRAMING_SCHEMA, max_tokens=256,
                      mock_response=mock_response or {"framed": False, "move_direction": "descend",
                                                      "target_altitude_m": 2.0,
                                                      "reason": "subject small — drop lower for a closer look"})
