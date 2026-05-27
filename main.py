import argparse
import glob
import json
import os
import sys

# ---------------------------------------------------------------------------
# Robot / pipeline configuration
# ---------------------------------------------------------------------------
ROBOT_PORT    = "/dev/ttyACM0"   # serial port of the Pico
CAMERA_RESOLUTION = (1920, 1080)  # capture resolution

# rotation_deg from the solver is CCW in screen coordinates (positive = CCW).
# Set ROTATION_SIGN = -1 if gripper_rotate(positive) means CW (most common).
# Set ROTATION_SIGN = +1 if gripper_rotate(positive) means CCW.
ROTATION_SIGN = -1


CAPTURE_PATH        = "debug_output/capture.jpg"
CAPTURE_RAW_PATH    = "debug_output/capture_raw.jpg"
CAPTURE_COARSE_PATH = "debug_output/capture_coarse.jpg"

# Crop region within the captured frame (pixels).
# Set to None to use the full frame.
# Calibrate once by running: python main.py --show-crop
CROP_X =  262  # left edge   (P2 x)
CROP_Y =  115  # top edge    (P1 y)
CROP_W = 1260  # width       (+20 % from previous 1050)
CROP_H = 1056  # height      (+20 % from previous 880)

from gamefield_detection import BORDER_DETECTION, BORDER_OUTPUT_W, BORDER_OUTPUT_H, \
    detect_a4_border, detect_a4_border_two_frame


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_debug_dir():
    debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_output")
    os.makedirs(debug_dir, exist_ok=True)
    for f in glob.glob(os.path.join(debug_dir, "*")):
        os.remove(f)
    os.environ["ZOLVER_TEMP_DIR"] = debug_dir
    print(f"Debug output directory: {debug_dir}")
    return debug_dir


# ---------------------------------------------------------------------------
# Pipeline step 1 — capture
# ---------------------------------------------------------------------------

def _crop(frame):
    """Apply the configured crop to *frame*. Returns the frame unchanged if no crop is set."""
    x = CROP_X or 0
    y = CROP_Y or 0
    h, w = frame.shape[:2]
    x2 = x + CROP_W if CROP_W else w
    y2 = y + CROP_H if CROP_H else h
    return frame[y:y2, x:x2]



def take_picture(save_path: str = CAPTURE_PATH, robot=None) -> str:
    """Capture one or two frames from the Pi camera and return the warped playfield path.

    When *robot* is provided and BORDER_DETECTION is enabled, uses a two-shot
    strategy: LED off → capture detection frame (markers visible) → LED on →
    capture puzzle frame (correct lighting). ArUco transform is computed from
    the dark frame and applied to the lit frame.
    """
    import time
    import cv2
    from picamera2 import Picamera2

    cam = Picamera2()
    cam.configure(cam.create_still_configuration(main={"size": CAMERA_RESOLUTION}))
    cam.start()

    if robot is not None and BORDER_DETECTION:
        robot.led_off()
        time.sleep(0.5)                      # let exposure settle with LED off
        detect_raw = cam.capture_array()
        robot.led_on()
        time.sleep(0.5)                      # let exposure settle with LED on
        puzzle_raw = cam.capture_array()
    else:
        if robot is not None:
            robot.led_on()
        puzzle_raw = cam.capture_array()
        detect_raw = puzzle_raw              # single-shot: same frame for both

    cam.stop()
    cam.close()

    detect_frame = cv2.cvtColor(detect_raw, cv2.COLOR_RGB2BGR)
    puzzle_frame = cv2.cvtColor(puzzle_raw, cv2.COLOR_RGB2BGR)

    cv2.imwrite(CAPTURE_RAW_PATH, puzzle_frame)
    print(f"[1/3] Raw capture saved:       {CAPTURE_RAW_PATH}  ({puzzle_frame.shape[1]}×{puzzle_frame.shape[0]} px)")

    detect_coarse = _crop(detect_frame)
    puzzle_coarse = _crop(puzzle_frame)
    cv2.imwrite(CAPTURE_COARSE_PATH, puzzle_coarse)
    print(f"[1/3] Coarse crop saved:       {CAPTURE_COARSE_PATH}  ({puzzle_coarse.shape[1]}×{puzzle_coarse.shape[0]} px)")

    if BORDER_DETECTION:
        warped = detect_a4_border_two_frame(detect_coarse, puzzle_coarse)
        if warped is not None:
            cv2.imwrite(save_path, warped)
            print(f"[1/3] Border-detected capture: {save_path}  ({warped.shape[1]}×{warped.shape[0]} px)")
            return save_path
        print("[WARN] Border not detected — falling back to static crop")

    cv2.imwrite(save_path, puzzle_coarse)
    print(f"[1/3] Cropped capture saved:   {save_path}  ({puzzle_coarse.shape[1]}×{puzzle_coarse.shape[0]} px)")
    return save_path


# ---------------------------------------------------------------------------
# Pipeline step 2 — solve
# ---------------------------------------------------------------------------

def solve_puzzle(image_path: str, green_screen: bool = False) -> list:
    """
    Run the solver on *image_path*.
    Returns the list of piece records from piece_centers.json, e.g.:
      [{"piece_index": 0,
        "grid_coord": [col, row],
        "start_center_robot_mm": [x, y],
        "end_center_px": [px, py]}, ...]
    """
    from solver.Puzzle.Puzzle import Puzzle

    puzzle = Puzzle(image_path, green_screen=green_screen)
    puzzle.solve_puzzle()

    centers_path = os.path.join(os.environ["ZOLVER_TEMP_DIR"], "piece_centers.json")
    with open(centers_path) as f:
        pieces = json.load(f)
    print(f"[2/3] Solver done — {len(pieces)} pieces detected")
    return pieces


# ---------------------------------------------------------------------------
# Pipeline step 3 — move
# ---------------------------------------------------------------------------

def move_pieces(robot, pieces: list):
    """
    For each piece: pick up, rotate to solved orientation, put back down in place.
    """
    print("[3/3] Starting piece movements")

    robot.motors_enable()
    robot.home_z()
    robot.gripper_up()
    robot.home_x()
    robot.home_y()
    robot.reset_rotation()
    robot.gripper_up()
    robot.reset_rotation()

    for piece in pieces:
        idx     = piece["piece_index"]
        x_s, y_s = piece["start_center_robot_mm"]
        x_e, y_e = piece["end_center_robot_mm"]
        # rotation_deg is CCW in screen coords; apply ROTATION_SIGN to match robot convention.
        angle = ROTATION_SIGN * (piece["rotation_deg"] % 360)
        if angle > 180:
            angle -= 360
        if angle < -180:
            angle += 360

        print(f"  piece {idx}: ({x_s},{y_s}) → ({x_e},{y_e})  rotate {angle}°")

        # Pick up (double-tap: first tap seats the piece, second picks it up)
        robot.go_to(x_s, y_s)
        robot.gripper_down()
        robot.vacuum_pump_on()
        robot.gripper_on()
        robot.gripper_up()

        # Rotate to target orientation while in the air
        robot.reset_rotation()
        robot.gripper_rotate(angle)

        # Move to solved position and place
        robot.go_to(x_e, y_e)
        robot.gripper_down()
        robot.gripper_off()
        robot.vacuum_pump_off()
        robot.gripper_up()

    robot.motors_disable()
    print("[3/3] Done")


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(green_screen: bool = False):
    """End-to-end run: capture → solve → move."""
    from robot.pico_interface import PicoInterface

    _setup_debug_dir()
    robot = PicoInterface(port=ROBOT_PORT)
    try:
        image_path = take_picture(robot=robot)
        pieces     = solve_puzzle(image_path, green_screen=green_screen)
        move_pieces(robot, pieces)
        robot.led_off()
    finally:
        robot.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PREN Puzzlesolver")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--gui",  action="store_true", help="open GUI viewer")
    mode.add_argument("--prod", action="store_true", help="production pipeline: capture → solve → move")
    parser.add_argument("--green-screen", action="store_true", help="enable green background removal")
    args = parser.parse_args()

    if args.gui:
        _setup_debug_dir()
        from PyQt5.QtWidgets import QApplication
        from solver.GUI.Viewer import Viewer
        app = QApplication(sys.argv)
        viewer = Viewer()
        viewer.show()
        sys.exit(app.exec_())

    else:  # --prod
        run_pipeline(green_screen=args.green_screen)


if __name__ == "__main__":
    main()
