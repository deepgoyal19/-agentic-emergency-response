"""JSON schemas for Gemma 4 structured outputs (strict mode).

Each agent constrains Gemma's output to one of these schemas so the next agent
in the pipeline always receives well-formed, typed data. Cerebras supports
`strict: true` json_schema response formats, so these are enforced server-side.
"""

# --- Dispatcher: 911 text -> classified incident + assignment -------------
DISPATCH_SCHEMA = {
    "name": "dispatch_decision",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "incident_type": {
                "type": "string",
                "enum": ["medical_accident", "personal_safety", "vehicle_fire", "other"],
            },
            "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
            "summary": {"type": "string"},
            "required_payload": {
                "type": "string",
                "enum": ["first_aid_kit", "none", "extinguisher", "camera_only"],
            },
            "dispatch_drone_id": {"type": "integer"},
            "from_tower_id": {"type": "integer"},
            "rationale": {"type": "string"},
        },
        "required": [
            "incident_type", "severity", "summary", "required_payload",
            "dispatch_drone_id", "from_tower_id", "rationale",
        ],
        "additionalProperties": False,
    },
}

# --- Path Planner: assignment -> ordered waypoints ------------------------
PATH_SCHEMA = {
    "name": "flight_plan",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "waypoints": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "z": {"type": "number"},
                        "label": {"type": "string"},
                    },
                    "required": ["x", "y", "z", "label"],
                    "additionalProperties": False,
                },
            },
            "cruise_altitude": {"type": "number"},
            "eta_seconds": {"type": "number"},
            "notes": {"type": "string"},
        },
        "required": ["waypoints", "cruise_altitude", "eta_seconds", "notes"],
        "additionalProperties": False,
    },
}

# --- Analyst: camera image -> scene understanding + recommended action ----
ANALYSIS_SCHEMA = {
    "name": "scene_analysis",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "scene_description": {"type": "string"},
            "people_detected": {"type": "integer"},
            "hazards": {"type": "array", "items": {"type": "string"}},
            "fire_present": {"type": "boolean"},
            "injuries_suspected": {"type": "boolean"},
            "recommended_action": {
                "type": "string",
                "enum": [
                    "deliver_first_aid", "hold_position_and_warn",
                    "deploy_extinguisher", "relay_to_responders", "continue_observation",
                ],
            },
            "action_detail": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": [
            "scene_description", "people_detected", "hazards", "fire_present",
            "injuries_suspected", "recommended_action", "action_detail", "confidence",
        ],
        "additionalProperties": False,
    },
}

# --- Fleet Manager: battery + distances -> continue or return decision -----
FLEET_SCHEMA = {
    "name": "fleet_decision",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "decision": {"type": "string", "enum": ["continue_mission", "return_to_base"]},
            "return_target": {"type": "string", "enum": ["home_tower", "nearest_tower", "none"]},
            "rationale": {"type": "string"},
        },
        "required": ["decision", "return_target", "rationale"],
        "additionalProperties": False,
    },
}

# --- Executor: action -> drone commands + report to command center --------
EXECUTOR_SCHEMA = {
    "name": "execution_report",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "drone_commands": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "descend_to_subject", "release_first_aid", "activate_siren",
                        "broadcast_warning", "discharge_extinguisher", "circle_and_record",
                        "ascend_and_relay", "return_to_tower",
                    ],
                },
            },
            "command_center_report": {"type": "string"},
            "status": {"type": "string", "enum": ["resolved", "responders_needed", "ongoing"]},
        },
        "required": ["drone_commands", "command_center_report", "status"],
        "additionalProperties": False,
    },
}
