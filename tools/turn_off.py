"""Move the robot head to (50, 50) to get it out of the way."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from robot.pico_interface import PicoInterface

PORT   = "/dev/ttyACM0"

robot = PicoInterface(port=PORT)

robot.motors_disable()
robot.close()

print("robot off")

