"""The three hero incidents. Same pipeline, different incident data.

At runtime the `image` is replaced by a live frame from the Webots drone camera
(controller calls camera.getImage() -> PNG -> base64). The `image` field here is a
fallback test photo so the vision agents can be developed before the world is built:
drop any matching jpg/png into code/assets/ to exercise Gemma vision in mock-free runs.

Coordinates are in Webots world units (x, y ground plane; z up).
"""
from dataclasses import dataclass, field


@dataclass
class Incident:
    id: str
    title: str
    call_text: str            # what the 911 caller says -> Dispatcher input
    location: tuple           # (x, y, z) incident position in the city
    test_image: str           # fallback photo path (used until live camera frames exist)
    # Absolute world coord of this incident on a real street in the OSM (Morges) town,
    # used when the OSM hero world is loaded (set on the instances below).
    osm_location: tuple = None
    mesh_location: tuple = None
    # Mock outputs (used only in mock mode so the pipeline runs end-to-end pre-API):
    mock_dispatch: dict = field(default_factory=dict)
    mock_analysis: dict = field(default_factory=dict)


ACCIDENT = Incident(
    id="accident",
    title="Highway collision — casualties, no EMS on scene yet",
    call_text=(
        "There's been a bad car crash at the Main St overpass. Two cars, "
        "I see people on the ground and they're not moving much. No ambulance yet!"
    ),
    location=(175.0, 35.0, 0.0),
    osm_location=(-136.5, 528.4, 0.0),
    mesh_location=(-9.0, -249.0, 0.0),
    test_image="assets/accident.jpg",
    mock_dispatch={
        "incident_type": "medical_accident", "severity": "critical",
        "summary": "Two-vehicle collision, suspected casualties, EMS not yet on scene.",
        "required_payload": "first_aid_kit", "dispatch_drone_id": 1, "from_tower_id": 2,
        "rationale": "Tower 2 is nearest; drone 1 is charged and carries a first-aid module.",
    },
    mock_analysis={
        "scene_description": "Two damaged vehicles at an intersection; two persons on the ground beside the cars.",
        "people_detected": 2, "hazards": ["leaking fluid", "broken glass"],
        "fire_present": False, "injuries_suspected": True,
        "recommended_action": "deliver_first_aid",
        "action_detail": "Drop first-aid kit beside the nearest casualty; relay count of 2 injured to EMS.",
        "confidence": 0.87,
    },
)

STALKER = Incident(
    id="stalker",
    title="Person being followed — safety escort",
    call_text=(
        "Someone has been following me for three blocks on Elm Street. "
        "He keeps speeding up when I do. I'm scared, please send help."
    ),
    location=(-170.0, 120.0, 0.0),
    osm_location=(-909.0, 490.5, 0.0),
    mesh_location=(-136.0, 230.0, 0.0),
    test_image="assets/stalker.jpg",
    mock_dispatch={
        "incident_type": "personal_safety", "severity": "high",
        "summary": "Caller reports being followed on foot; potential assault risk.",
        "required_payload": "camera_only", "dispatch_drone_id": 3, "from_tower_id": 1,
        "rationale": "Tower 1 nearest to Elm St; drone 3 has siren + spotlight for deterrence.",
    },
    mock_analysis={
        "scene_description": "A pedestrian on a sidewalk at night with a second person trailing close behind.",
        "people_detected": 2, "hazards": ["subject closing distance"],
        "fire_present": False, "injuries_suspected": False,
        "recommended_action": "hold_position_and_warn",
        "action_detail": "Record the follower, activate siren/spotlight, broadcast that authorities are en route.",
        "confidence": 0.8,
    },
)

FIRE = Incident(
    id="fire",
    title="Vehicle fire — suppression before crews arrive",
    call_text=(
        "A car is on fire in the parking lot off 5th Avenue. Flames coming from "
        "the front of it, lots of smoke. Nobody seems to be inside but hurry."
    ),
    location=(-40.0, -185.0, 0.0),
    osm_location=(-722.1, -112.5, 0.0),
    mesh_location=(170.0, 290.0, 0.0),
    test_image="assets/fire.jpg",
    mock_dispatch={
        "incident_type": "vehicle_fire", "severity": "high",
        "summary": "Engine-compartment vehicle fire in a parking lot, no occupants reported.",
        "required_payload": "extinguisher", "dispatch_drone_id": 2, "from_tower_id": 3,
        "rationale": "Tower 3 nearest; drone 2 carries the suppressant module.",
    },
    mock_analysis={
        "scene_description": "A sedan with active flames and heavy smoke from the engine bay in an open lot.",
        "people_detected": 0, "hazards": ["active fire", "smoke", "fuel tank proximity"],
        "fire_present": True, "injuries_suspected": False,
        "recommended_action": "deploy_extinguisher",
        "action_detail": "Target the engine bay base of the flames; relay live video to fire command.",
        "confidence": 0.91,
    },
)

HIGHRISE_FIRE = Incident(
    id="highrise_fire",
    title="High-rise fire — rooftop blaze, fire-brigade drone squad",
    call_text=(
        "The top of the tower on Harbor Plaza is on fire! Flames on the roof and "
        "the upper floors, heavy black smoke. Ladder trucks can't reach that high — "
        "we need the aerial fire team now."
    ),
    location=(150.0, 150.0, 90.0),     # rooftop of the burning skyscraper (z = roof height)
    osm_location=(150.0, 150.0, 90.0),
    mesh_location=(150.0, 150.0, 90.0),
    test_image="assets/fire.jpg",
    mock_dispatch={
        "incident_type": "structure_fire", "severity": "critical",
        "summary": "Rooftop fire on a high-rise; beyond ladder reach, aerial suppression required.",
        "required_payload": "extinguisher", "dispatch_drone_id": 5, "from_tower_id": 2,
        "rationale": "Tower 2 is nearest the high-rise; fire-brigade drones carry suppressant.",
    },
    mock_analysis={
        "scene_description": "Active flames and heavy smoke across the rooftop and upper facade of a tall building.",
        "people_detected": 0, "hazards": ["active fire", "structural heat", "thick smoke"],
        "fire_present": True, "injuries_suspected": False,
        "recommended_action": "deploy_extinguisher",
        "action_detail": "Surround the roof by sector, target the base of the flames, relay thermal video to fire command.",
        "confidence": 0.9,
    },
)

PRESETS = {p.id: p for p in (ACCIDENT, STALKER, FIRE, HIGHRISE_FIRE)}
