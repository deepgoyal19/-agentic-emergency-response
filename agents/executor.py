"""Agent 5 — Executor.

Input:  the Analyst's recommended action + incident type.
Output: concrete drone commands (sent to the Webots controller to animate) and a
        report back to the Command Center. Closes the loop the Dispatcher opened.
"""
from .cerebras_client import gemma, text_content
from .schemas import EXECUTOR_SCHEMA

SYSTEM = (
    "You are the EXECUTOR agent on an emergency drone. Translate the Analyst's "
    "recommended action into a concrete, ordered list of drone commands from the allowed "
    "set, and write a concise report for the Command Center. Only choose commands that "
    "match the situation and the drone's payload. Respond ONLY with the structured schema."
)


def run(analysis: dict, incident_type: str, payload: str, mock_response: dict | None = None):
    user = (
        f"INCIDENT TYPE: {incident_type}\n"
        f"DRONE PAYLOAD: {payload}\n"
        f"ANALYST ASSESSMENT: {analysis}\n\n"
        "Decide the drone commands and the command-center report now."
    )
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": [text_content(user)]},
    ]
    return gemma.chat(messages, schema=EXECUTOR_SCHEMA, mock_response=mock_response)
