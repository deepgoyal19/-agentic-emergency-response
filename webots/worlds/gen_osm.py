"""Regenerate rescue_city_osm.wbt = OSM Morges world + our rescue scenario.

Reads osm_city.wbt (the raw OSM import) and the flame block from rescue_city.wbt,
applies the world fixes (local GPS, basicTimeStep, sun), and appends our towers,
incidents, drone, and traffic — scaled + offset into the inland plaza.
Run:  python gen_osm.py
"""
import re

OX, OY, SC = -700.0, 290.0, 1.0       # (unused for towers now; kept for traffic init)
NUM_TRAFFIC = 16                      # cars; city_life drives them along real OSM roads

# Charging towers spread across town (match DEFAULT_FLEET osm_xy)
TW1, TW2, TW3 = (-828.9, 363.2), (-231.6, 413.3), (-584.1, -168.0)
# Incident sites on REAL streets ~150 m from each tower (match presets.osm_location)
ACC = (-136.5, 528.4)                 # accident  (near tower2)
STK = (-909.0, 490.5)                 # stalker   (near tower1)
FIR = (-722.1, -112.5)                # fire      (near tower3)


def a(x, y):
    return (round(x * SC + OX, 2), round(y * SC + OY, 2))


src = open('osm_city.wbt', encoding='utf-8').read()
src = src.replace('gpsCoordinateSystem "WGS84"', 'gpsCoordinateSystem "local"', 1)
src = re.sub(r'gpsReference [0-9.]+ [0-9.]+ 0', 'gpsReference 0 0 0', src, count=1)
src = src.replace("  lineScale 12\n}",
    "  lineScale 12\n  basicTimeStep 8\n  defaultDamping Damping {\n    linear 0.2\n    angular 0.5\n  }\n}", 1)
# make the camera FOLLOW the drone up close (importer leaves it ~4km up, static)
src = re.sub(r'Viewpoint \{.*?\n\}',
    'Viewpoint {\n'
    '  orientation -0.42 -0.13 0.90 2.35\n'
    '  position -300 360 70\n'
    '  near 1\n'
    '  follow "Mavic 2 PRO"\n'
    '  followSmoothness 0.2\n'
    '}', src, count=1, flags=re.S)

# ground: flat dark concrete (a tiled texture moires badly over a 3.5km plane when zoomed out)
src = re.sub(r'appearance PBRAppearance \{\s*baseColorMap ImageTexture \{.*?grass\.jpg.*?\}\s*roughness 1\s*metalness 0\s*\}',
             'appearance PBRAppearance { baseColor 0.33 0.34 0.37 roughness 0.95 metalness 0 }',
             src, flags=re.S, count=1)

# give the buildings varied wall/roof textures (the OSM import makes them all identical)
import random
random.seed(7)
_WALLS = ["glass building", "concrete building", "classic building", "residential building",
          "old building", "office building", "factory building"]
src = re.sub(r'wallType "[^"]*"', lambda m: f'wallType "{random.choice(_WALLS)}"', src)
anchor = 'EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/objects/traffic/protos/ParkingLines.proto"'
src = src.replace(anchor, anchor + "\n" + "\n".join(
    'EXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/%s"' % p
    for p in ["vehicles/protos/tesla/TeslaModel3Simple.proto",
              "humans/pedestrian/protos/Pedestrian.proto",
              "appearances/protos/Pavement.proto",
              "robots/dji/mavic/protos/Mavic2Pro.proto"]), 1)
src = src.replace("TexturedBackgroundLight {\n}",
    "TexturedBackgroundLight {\n}\nDirectionalLight {\n  ambientIntensity 0.9\n  color 1 1 0.96\n  direction 0.35 0.25 -1\n  intensity 5\n  castShadows TRUE\n}", 1)


def tower(n, x, y):
    return f'''DEF TOWER{n} Solid {{
  translation {x} {y} 0
  children [
    Pose {{ translation 0 0 0.1 children [ Shape {{ appearance PBRAppearance {{ baseColor 0.15 0.5 0.9 roughness 0.4 metalness 0.3 }} geometry Cylinder {{ radius 2.5 height 0.2 }} }} ] }}
    Pose {{ translation 0 0 2.5 children [ Shape {{ appearance PBRAppearance {{ baseColor 0.3 0.3 0.35 roughness 0.3 metalness 0.6 }} geometry Box {{ size 0.4 0.4 5 }} }} ] }}
    Pose {{ translation 0 0 5.2 children [ Shape {{ appearance PBRAppearance {{ baseColor 0.1 0.85 1 emissiveColor 0.1 0.8 1 }} geometry Sphere {{ radius 0.4 }} }} ] }}
  ]
  name "tower{n}"
}}'''


# towers at their spread sites across town (match DEFAULT_FLEET osm_xy)
t1, t2, t3 = TW1, TW2, TW3
# accident objects around the street point
ac, acb = ACC, (ACC[0] + 2.6, ACC[1] + 1.3)
v1, v2 = (ACC[0] - 0.8, ACC[1] + 1.8), (ACC[0] + 1.6, ACC[1] - 1.3)
# stalker
wo, fo = STK, (STK[0] - 1.6, STK[1] - 1.6)
# fire
fc, fl1, fl2 = FIR, (FIR[0] + 0.8, FIR[1] + 0.4), (FIR[0] + 0.9, FIR[1] + 0.5)
civ = [("CIVIL1", a(30, 14), "0.8 0.2 0.2"), ("CIVIL2", a(-46, 12), "0.2 0.4 0.8"),
       ("CIVIL3", a(10, 34), "0.9 0.7 0.1"), ("CIVIL4", a(-20, -24), "0.3 0.7 0.3")]
flame_kids = re.search(r'DEF FIRE_FLAMES Solid \{\n  translation [^\n]+\n  children \[(.*?)\n  \]\n  name "flames"',
                       open('rescue_city.wbt', encoding='utf-8').read(), re.S).group(1)

block = f'''
# ===================== RESCUE SCENARIO (inland plaza, scale {SC}) =====================
{tower(1,*t1)}
{tower(2,*t2)}
{tower(3,*t3)}
DEF CRASH_CAR_A TeslaModel3Simple {{ translation {ac[0]} {ac[1]} 0.4 rotation 0 0 1 0.35 name "crash car A" }}
DEF CRASH_CAR_B TeslaModel3Simple {{ translation {acb[0]} {acb[1]} 0.4 rotation 0 0 1 -2.5 name "crash car B" }}
DEF VICTIM1 Pedestrian {{ translation {v1[0]} {v1[1]} 0.35 rotation 1 0 0 1.5708 name "victim1" controller "<none>" }}
DEF VICTIM2 Pedestrian {{ translation {v2[0]} {v2[1]} 0.35 rotation 0 1 0 1.5708 name "victim2" controller "<none>" }}
DEF WOMAN Pedestrian {{ translation {wo[0]} {wo[1]} 1.27 rotation 0 0 1 1.5708 name "woman" controller "<none>" }}
DEF FOLLOWER Pedestrian {{ translation {fo[0]} {fo[1]} 1.27 rotation 0 0 1 1.5708 name "follower" shirtColor 0.1 0.1 0.1 pantsColor 0.05 0.05 0.05 controller "<none>" }}
DEF FIRE_CAR TeslaModel3Simple {{ translation {fc[0]} {fc[1]} 0.4 rotation 0 0 1 0.6 name "burning car" }}
DEF FIRE_FLAMES Solid {{
  translation {fc[0]} {fc[1]} 0
  children [{flame_kids}
  ]
  name "flames"
}}
PointLight {{ attenuation 0 0 1 color 1 0.5 0.15 intensity 16 location {fl1[0]} {fl1[1]} 1.5 castShadows FALSE }}
PointLight {{ attenuation 0 0 1 color 1 0.4 0.1 intensity 8 location {fl2[0]} {fl2[1]} 3.2 castShadows FALSE }}
'''
# traffic cars (city_life drives them along real OSM roads; initial spots near plaza)
for i in range(NUM_TRAFFIC):
    cx, cy = a(-60 + (i % 4) * 40, -50 + (i // 4) * 30)
    block += f'DEF TRAFFIC{i+1} TeslaModel3Simple {{ translation {cx} {cy} 0.35 name "traffic{i+1}" }}\n'
for nm, (x, y), col in civ:
    block += f'DEF {nm} Pedestrian {{ translation {x} {y} 1.27 name "{nm.lower()}" shirtColor {col} controller "<none>" }}\n'
block += 'DEF CITY_LIFE Robot { name "city_life" controller "city_life" supervisor TRUE children [] }\n'
block += f'''Mavic2Pro {{
  translation {t2[0]} {t2[1]} 0.3
  rotation 0 0 1 3.14159
  name "Mavic 2 PRO"
  controller "drone_agent"
  supervisor TRUE
  cameraSlot [ Camera {{ width 640 height 400 near 0.2 }} ]
}}
'''
open('rescue_city_osm.wbt', 'w', encoding='utf-8').write(src + block)
print(f"regenerated rescue_city_osm.wbt  ({NUM_TRAFFIC} cars)  plaza={a(0,0)}")
