"""Tiny controller: capture one overhead image of the mesh city, then idle."""
from controller import Robot

r = Robot()
dt = int(r.getBasicTimeStep())
cam = r.getDevice("camera")
cam.enable(dt)
for _ in range(15):
    r.step(dt)
cam.saveImage("F:/Hackathon/code/webots/frames/city_preview.png", 95)
print("saved city_preview.png", flush=True)
for _ in range(8):
    r.step(dt)
