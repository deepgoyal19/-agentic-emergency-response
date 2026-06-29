"""Agent 4 — Analyst (the multimodal core).

Input:  the drone camera frame (image) + incident context.
Output: structured scene understanding — people count, hazards, fire/injury flags —
        and a recommended action. This is the meaningful multimodal use of Gemma 4 31B:
        real vision reasoning over a real camera image, constrained to a typed schema
        the Executor can act on.
"""
from .cerebras_client import gemma, text_content, image_content
from .schemas import ANALYSIS_SCHEMA

SYSTEM = (
    "You are the ANALYST agent for an emergency drone fleet. You look at a real drone "
    "camera image of an active incident and produce a precise, structured assessment: "
    "describe the scene, count people, list hazards, flag fire and suspected injuries, "
    "and recommend exactly one action. Ground every field in what is visible. "
    "Reason carefully but respond ONLY with the structured schema."
)


def run(image_path_or_url: str, incident_context: str,
        reasoning_effort: str = "none", mock_response: dict | None = None):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": [
            text_content(f"INCIDENT CONTEXT: {incident_context}\n\nAnalyze this drone camera image:"),
            image_content(image_path_or_url),
        ]},
    ]
    # Validated on live gemma-4-31b: reasoning_effort='none' is both faster (~200ms) and
    # MORE reliable here — reasoning tokens were truncating the strict JSON on busy scenes.
    # max_tokens 1024 leaves room for the full structured analysis.
    return gemma.chat(messages, schema=ANALYSIS_SCHEMA, reasoning_effort=reasoning_effort,
                      max_tokens=1024, mock_response=mock_response)
