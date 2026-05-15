"""Move the robot head to (50, 50) to get it out of the way."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from robot.pico_interface import PicoInterface

PARK_X = 50
PARK_Y = 50
PORT   = "/dev/ttyACM0"

robot = PicoInterface(port=PORT)
try:
    robot.motors_enable()
    robot.gripper_up()
    robot.go_to(PARK_X, PARK_Y)
    print(f"Head parked at ({PARK_X}, {PARK_Y})")
finally:
    robot.motors_disable()
    robot.close()
