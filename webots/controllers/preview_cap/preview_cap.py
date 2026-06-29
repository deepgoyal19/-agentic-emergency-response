"""Capture one preview image of the city with the live lighting, then quit Webots.
Lets us check the render headlessly instead of eyeballing the GUI."""
from controller import Supervisor

r = Supervisor()
dt = int(r.getBasicTimeStep())
cam = r.getDevice("camera")
cam.enable(dt)
for _ in range(20):
    r.step(dt)
cam.saveImage("F:/Hackathon/code/webots/frames/preview.png", 95)
print("saved preview.png", flush=True)
r.step(dt)
r.simulationQuit(0)
