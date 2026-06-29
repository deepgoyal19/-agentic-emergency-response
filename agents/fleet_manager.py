"""Agent 6 — Fleet Manager (battery-aware decision making).

When a drone's battery falls to the low-battery threshold, it pings this agent:
"Can I continue the mission, or must I return — and can I reach my home tower or
only the nearest one?" The Fleet Manager reasons over battery vs. the distances and
returns a typed decision. If it says return, the Path Planner is re-invoked to route
the drone to the chosen tower.
"""
from .cerebras_client import gemma, text_content
from .schemas import FLEET_SCHEMA

SYSTEM = (
    "You are the FLEET MANAGER agent for an emergency drone fleet. A drone reports its "
    "remaining battery and the distances to (a) finishing its current task, (b) its home "
    "tower, and (c) the nearest tower. Decide whether it can CONTINUE the mission or must "
    "RETURN to recharge, and if returning, whether it can reach its HOME tower or must "
    "divert to the NEAREST tower. Always keep a safety reserve so the drone never strands. "
    "Respond ONLY with the structured schema."
)


def run(battery, dist_remaining, dist_home, dist_nearest, reserve,
        mission_done, mock_response: dict | None = None):
    user = (
        f"BATTERY: {battery:.0f}%\n"
        f"DISTANCE to finish current task: {dist_remaining:.0f} m\n"
        f"DISTANCE to home tower: {dist_home:.0f} m\n"
        f"DISTANCE to nearest tower: {dist_nearest:.0f} m\n"
        f"SAFETY RESERVE required on arrival: {reserve:.0f}%\n"
        f"Current task already complete: {mission_done}\n\n"
        "Decide: continue or return, and which tower."
    )
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": [text_content(user)]},
    ]
    return gemma.chat(messages, schema=FLEET_SCHEMA, mock_response=mock_response)
