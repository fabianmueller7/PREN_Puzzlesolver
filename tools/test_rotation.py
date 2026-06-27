"""Test the gripper rotation axis.

Resets the current angular position to 0, then rotates the gripper through a
sequence of angles, pausing at each so you can verify the move by eye.

Usage:
    python tools/test_rotation.py                # default sweep: 0 90 180 270 0
    python tools/test_rotation.py 90 -90 45      # rotate to each given angle (deg)

Angles are absolute, relative to the position the axis is in at startup
(declared as 0 via reset_rotation). Positive/negative follow the firmware's
convention for gripper_rotate.
"""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from robot.pico_interface import PicoInterface

PORT  = "/dev/ttyACM0"
PAUSE = 3  # seconds to wait at each angle so you can observe

DEFAULT_ANGLES = [0, 90, 180, 270, 0]


def parse_angles(argv):
    if not argv:
        return DEFAULT_ANGLES
    try:
        return [float(a) for a in argv]
    except ValueError:
        print(f"Invalid angle(s): {argv}. Provide numbers in degrees, e.g. 90 -45 180")
        sys.exit(1)


def main():
    angles = parse_angles(sys.argv[1:])

    robot = PicoInterface(port=PORT)
    try:
        robot.motors_enable()
        robot.gripper_up()
        print("Homing rotation axis...")
        robot.home_a()           # drive to mechanical endstop
        robot.reset_rotation()   # declare homed position as 0 deg
        print(f"Rotation homed and zeroed. Testing angles: {angles}")

        for angle in angles:
            print(f"-> rotating to {angle} deg")
            robot.gripper_rotate(angle)
            time.sleep(PAUSE)
            print("   done.")

        print("Rotation test complete.")
    finally:
        robot.motors_disable()
        robot.close()


if __name__ == "__main__":
    main()
