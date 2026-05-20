"""Move robot to calibration positions and wait 10s at each."""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from robot.pico_interface import PicoInterface

PORT = "/dev/ttyACM0"

POSITIONS = [
    ("Oben rechts",  50,  250),
    ("Unten Mitte", 150,  350),
    ("Oben links",  260,  250),
    ("Mitte oben",  147,  245),
]

robot = PicoInterface(port=PORT)
try:
    robot.motors_enable()
    robot.home_z()
    robot.gripper_up()
    robot.home_x()
    robot.home_y()

    for label, x, y in POSITIONS:
        print(f"-> {label}: X={x} Y={y}")
        robot.go_to(x, y)
        print(f"   Warte 10s ...")
        robot.gripper_down()
        time.sleep(10)
        robot.gripper_up()

        print(f"   Fertig.")

    print("Kalibrierung abgeschlossen.")
finally:
    robot.motors_disable()
    robot.close()
