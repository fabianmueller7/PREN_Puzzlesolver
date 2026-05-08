import argparse
import glob
import json
import os
import sys

# ---------------------------------------------------------------------------
# Robot / pipeline configuration
# ---------------------------------------------------------------------------
ROBOT_PORT    = "COM22"   # serial port of the Pico
CAMERA_INDEX  = 0         # OpenCV camera index

Z_UP    =   0   # safe travel height [mm]  — TODO: calibrate
Z_PICK  = -50   # height to pick up a piece [mm]  — TODO: calibrate
Z_PLACE = -50   # height to place a piece [mm]  — TODO: calibrate

CAPTURE_PATH     = "debug_output/capture.jpg"
CAPTURE_RAW_PATH = "debug_output/capture_raw.jpg"

# Crop region within the captured frame (pixels).
# Set to None to use the full frame.
# Calibrate once by running: python main.py --show-crop
CROP_X =  0    # left edge   — TODO: calibrate
CROP_Y =  0    # top edge    — TODO: calibrate
CROP_W =  None # width  (None = full width)
CROP_H =  None # height (None = full height)


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


def take_picture(save_path: str = CAPTURE_PATH) -> str:
    """Capture one frame from the camera, save raw and cropped versions."""
    import cv2
    cap = cv2.VideoCapture(CAMERA_INDEX)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Camera capture failed — check CAMERA_INDEX")
    cv2.imwrite(CAPTURE_RAW_PATH, frame)
    print(f"[1/3] Raw capture saved:    {CAPTURE_RAW_PATH}  ({frame.shape[1]}×{frame.shape[0]} px)")
    cropped = _crop(frame)
    cv2.imwrite(save_path, cropped)
    print(f"[1/3] Cropped capture saved: {save_path}  ({cropped.shape[1]}×{cropped.shape[0]} px)")
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
        "end_center": [px, py]}, ...]
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

def _grid_coord_to_robot_mm(grid_coord: list) -> tuple:
    """
    Convert a solved-puzzle grid coordinate [col, row] to robot mm (x, y).

    TODO: implement the mapping once the A5-frame → robot calibration is known.
          The A5 target positions (in A5-frame mm) are in
          debug_output/positions_a5.json after running robot/coordinates.py.
    """
    raise NotImplementedError("A5-frame → robot coordinate mapping not yet calibrated")


def move_pieces(robot, pieces: list):
    """
    For each piece: pick up, rotate to solved orientation, put back down.
    """
    print("[3/3] Starting piece movements")

    robot.motors_enable()
    robot.home_x()
    robot.home_y()
    robot.home_z()
    robot.home_a()

    for piece in pieces:
        idx   = piece["piece_index"]
        x, y  = piece["start_center_robot_mm"]
        angle = piece["rotation_deg"]

        print(f"  piece {idx}: ({x}, {y}) mm  rotation {angle}°")

        robot.go_to(x, y)
        robot.go_to_z(Z_PICK)
        robot.magnet_on()
        robot.go_to_z(Z_UP)

        robot.go_to_a(angle)
        robot.go_to_a(0)

        robot.go_to_z(Z_PLACE)
        robot.magnet_off()
        robot.go_to_z(Z_UP)

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
        image_path = take_picture()
        pieces     = solve_puzzle(image_path, green_screen=green_screen)
        move_pieces(robot, pieces)
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

    else:
        run_pipeline(green_screen=args.green_screen)


if __name__ == "__main__":
    main()
