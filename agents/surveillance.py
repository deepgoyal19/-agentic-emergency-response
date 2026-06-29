"""Patrol surveillance — the patrol drone's aerial threat scanner.

The patrol drone streams a down-camera frame (tagged with its live GPS) on a fixed
cadence. Each frame goes to a fast Gemma 4 vision call that scans for an armed person
(a weapon/gun in hand). When a threat is detected the patrol marks the GPS so it can
return to the exact location — every image is geo-tagged for that reason.
"""
from .cerebras_client import gemma, text_content, image_content

SYSTEM = (
    "You are the SURVEILLANCE agent on a police patrol drone flying over a city. "
    "You receive a single aerial down-looking camera frame. Scan it for a public-safety "
    "threat — specifically a person holding a weapon (a gun/rifle in hand). Be precise: "
    "only report armed=true when you can actually see a weapon. Respond as JSON: "
    '{"armed_person": bool, "people_count": int, "threat_level": "none|low|high", '
    '"description": str}. Keep description to one short sentence.'
)

THREAT_SCHEMA = {
    "name": "threat_scan",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "armed_person": {"type": "boolean"},
            "people_count": {"type": "integer"},
            "threat_level": {"type": "string", "enum": ["none", "low", "high"]},
            "description": {"type": "string"},
        },
        "required": ["armed_person", "people_count", "threat_level", "description"],
        "additionalProperties": False,
    },
}


def scan(image_path_or_url: str, mock_response: dict | None = None):
    """Fast aerial threat scan — armed-person detection for the patrol drone."""
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": [
            text_content("Scan this aerial patrol frame. Is anyone holding a weapon?"),
            image_content(image_path_or_url),
        ]},
    ]
    return gemma.chat(messages, schema=THREAT_SCHEMA, max_tokens=256,
                      mock_response=mock_response or {
                          "armed_person": False, "people_count": 0,
                          "threat_level": "none", "description": "No threat in view."})
