import argparse
import glob
import json
import os
import sys
from time import sleep

# ---------------------------------------------------------------------------
# Robot / pipeline configuration
# ---------------------------------------------------------------------------
ROBOT_PORT    = "/dev/ttyACM0"   # serial port of the Pico
CAMERA_RESOLUTION = (1920, 1080)  # capture resolution

# rotation_deg from the solver is CCW in screen coordinates (positive = CCW).
# ROTATION_SIGN maps that to the robot's gripper_rotate direction. Determined empirically
# on hardware (the sim's pixel_to_robot reflection does NOT reliably predict it).
ROTATION_SIGN = +1

# Fine-tune pickup position offset (robot mm).
# Positive X = right, positive Y = down (robot coordinate convention).
PICKUP_OFFSET_X =  1   # 1 mm to the right
PICKUP_OFFSET_Y = -3   # 2 mm up

# Pickup settle timing (seconds). With the cup pressed down, PICKUP_SEAT_S lets it flatten
# the piece against the surface; PICKUP_GRIP_S lets the vacuum establish a full seal before
# lifting. Prevents tilted/shifted grabs. Increase if pieces still come up crooked.
PICKUP_SEAT_S = 0.3
PICKUP_GRIP_S = 0.3

# Mechanical home angle of the gripper A axis (firmware home_position_mm). gripper_rotate is
# absolute, so a piece is turned by `angle` via gripper_rotate(ROT_HOME_DEG + angle) after
# the axis is homed (empty) at pickup. Must match the firmware's A home_position.
ROT_HOME_DEG = 180

# End-of-run settle: vibrate the assembly (firmware 'vibrate' on STEPPER2) so pieces
# resting on top of each other drop into place.
VIBE_DELAY_US = 1000    # pulse delay (us); smaller = faster/higher-pitched shake
VIBE_DURATION_S = 10.0  # how long to vibrate (seconds)


# Front-panel button GPIO pins (Pico). The firmware emits a single
# {"type":"event","name":"button_press","pin":<n>} per short press (on release).
# GPI1 (pin 20) = "prepare" → home the robot; GPI2 (pin 21) = "start" → run pipeline.
BUTTON_HOME_PIN  = 20
BUTTON_START_PIN = 21


CAPTURE_PATH        = "debug_output/capture.jpg"
CAPTURE_RAW_PATH    = "debug_output/capture_raw.jpg"
CAPTURE_COARSE_PATH = "debug_output/capture_coarse.jpg"

# Crop region within the captured frame (pixels).
# Set to None to use the full frame.
# Calibrate once by running: python main.py --show-crop
CROP_X =  262  # left edge   (P2 x)
CROP_Y =   90  # top edge    (P1 y, extended up 25 px)
CROP_W = 1260  # width       (+20 % from previous 1050)
CROP_H = 1081  # height      (+25 px to keep bottom edge fixed)

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
    # Prefer the dedicated small (agglomerative) solver; fall back to the grid solver when
    # not applicable. Both emit piece_centers.json.
    puzzle.solve_puzzle_small(fallback=True)

    centers_path = os.path.join(os.environ["ZOLVER_TEMP_DIR"], "piece_centers.json")
    with open(centers_path) as f:
        pieces = json.load(f)
    print(f"[2/3] Solver done — {len(pieces)} pieces detected")
    return pieces


# ---------------------------------------------------------------------------
# Pipeline step 3 — move
# ---------------------------------------------------------------------------

def home_robot(robot):
    """Enable motors and home every axis. Leaves the gripper up and motors enabled."""
    print("[home] homing robot")
    robot.motors_enable()
    robot.home_z()
    robot.gripper_up()
    robot.home_x()
    robot.home_y()
    robot.gripper_up()
    print("[home] done")


def move_pieces(robot, pieces: list, skip_home: bool = False):
    """
    For each piece: pick up, rotate to solved orientation, put back down in place.

    When *skip_home* is True the homing sequence is skipped (motors are still
    enabled) — used when the robot was already homed via the front home button.
    """
    print("[3/3] Starting piece movements")

    if skip_home:
        robot.motors_enable()
    else:
        home_robot(robot)

    for piece in pieces:
        idx     = piece["piece_index"]
        x_s, y_s = piece["start_center_robot_mm"]
        x_e, y_e = piece["end_center_robot_mm"]
        # rotation_deg is CCW in screen coords; apply ROTATION_SIGN to match robot convention.
        angle = ROTATION_SIGN * piece["rotation_deg"]
        if angle > 180:
            angle -= 360
        if angle < -180:
            angle += 360

        print(f"  piece {idx}: ({x_s},{y_s}) → ({x_e},{y_e})  rotate {angle}°")

        # Home the rotation axis BEFORE gripping. The firmware's reset_rotation homes the A
        # axis (to ROT_HOME_DEG), so it must run with an EMPTY gripper — homing while holding
        # a piece spins it by an arbitrary amount and scrambles every placement (the cause of
        # the intermittent diagonal). After this the gripper sits at the known home angle.
        robot.go_to(x_s + PICKUP_OFFSET_X, y_s + PICKUP_OFFSET_Y)
        robot.reset_rotation()
        sleep(0.2)

        # Pick up (single descent) — settle so the cup seats the piece flat and the vacuum
        # seal establishes before lifting (avoids tilted/shifted grabs).
        robot.gripper_down()
        sleep(PICKUP_SEAT_S)
        robot.vacuum_pump_on()
        robot.gripper_on()
        sleep(PICKUP_GRIP_S)
        robot.gripper_up()

        # Rotate to target orientation while in the air. gripper_rotate is ABSOLUTE and home
        # is at ROT_HOME_DEG, so command home + angle to turn the piece by exactly `angle`.
        robot.gripper_rotate(ROT_HOME_DEG + angle)

        # Move to solved position and place
        robot.go_to(x_e, y_e)
        robot.gripper_down()
        robot.gripper_off()
        robot.vacuum_pump_off()
        robot.gripper_up()

    _wiggle_settle(robot, pieces)

    robot.motors_disable()
    print("[3/3] Done")


def _wiggle_settle(robot, pieces):
    """Vibrate the assembly so pieces resting on top of each other drop into place.
    Uses the firmware 'vibrate' command (STEPPER2) via robot.vibe()."""
    print(f"[3/3] Vibrate to settle: delay {VIBE_DELAY_US} us, {VIBE_DURATION_S} s")
    robot.gripper_up()
    robot.vibe(VIBE_DELAY_US, VIBE_DURATION_S)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(green_screen: bool = False):
    """End-to-end run: capture → solve → move."""
    from robot.pico_interface import PicoInterface

    _setup_debug_dir()
    robot = PicoInterface(port=ROBOT_PORT)
    try:
        _run_solve(robot, green_screen=green_screen)
    finally:
        robot.close()


def _run_solve(robot, green_screen: bool = False, skip_home: bool = False):
    """Capture → solve → move on an already-open robot connection."""
    image_path = take_picture(robot=robot)
    pieces     = solve_puzzle(image_path, green_screen=green_screen)
    move_pieces(robot, pieces, skip_home=skip_home)
    robot.led_off()


def run_pickup_test(green_screen: bool = False):
    """Pickup-accuracy check: place ONE piece, the robot detects it, picks it up, and HOLDS
    it in the air so you can see how well the cup is centred on the piece. Ctrl+C releases it
    back down and parks. No solving, no rotation — just capture → detect → pick → hold."""
    import numpy as np
    from robot.pico_interface import PicoInterface
    from solver.Puzzle.Puzzle import Puzzle
    from solver import config

    _setup_debug_dir()
    robot = PicoInterface(port=ROBOT_PORT)
    try:
        home_robot(robot)
        image_path = take_picture(robot=robot)

        puzzle = Puzzle(image_path, green_screen=green_screen)
        pieces = puzzle.pieces_ or []
        if not pieces:
            print("[pickup-test] no piece detected — check the capture")
            return

        # If more than one blob was detected, use the one nearest the image centre.
        import cv2
        h, w = cv2.imread(image_path).shape[:2]
        img_c = np.array([w / 2.0, h / 2.0])
        def _c(p):
            return np.array(p.img_centroid if p.img_centroid is not None
                            else Puzzle._piece_centroid(p), dtype=float)
        piece = min(pieces, key=lambda p: np.linalg.norm(_c(p) - img_c))
        cx, cy = _c(piece)

        sx, sy = config.pixel_to_robot(cx, cy, height_mm=config.PIECE_THICKNESS_MM)
        tx, ty = sx + PICKUP_OFFSET_X, sy + PICKUP_OFFSET_Y
        print(f"[pickup-test] piece centre: px=({cx:.1f},{cy:.1f})  "
              f"robot=({sx},{sy})  target(+offset)=({tx},{ty})")

        # Pick up (single descent) with the same settle timing as production.
        robot.go_to(tx, ty)
        robot.gripper_down()
        sleep(PICKUP_SEAT_S)
        robot.vacuum_pump_on()
        robot.gripper_on()
        sleep(PICKUP_GRIP_S)
        robot.gripper_up()
        print("[pickup-test] Holding piece. Check cup vs piece centre. Ctrl+C to release.")
        try:
            while True:
                sleep(0.5)
        except KeyboardInterrupt:
            print("\n[pickup-test] releasing")

        # Put it back where it was picked and park.
        robot.go_to(tx, ty)
        robot.gripper_down()
        robot.gripper_off()
        robot.vacuum_pump_off()
        robot.gripper_up()
        robot.motors_disable()
        robot.led_off()
    finally:
        robot.close()


# ---------------------------------------------------------------------------
# Status LED
# ---------------------------------------------------------------------------

def set_status_led(robot, status):
    """Drive the front-panel status LED.

      'ready' → green  (idle, waiting for a button press)
      'busy'  → orange (a task is running)
      'error' → red    (the last task crashed)

    NOTE: in the current firmware led_yellow and led_red share GPO2, so orange
    and red are physically the same lamp — green vs. lit is the reliable signal
    until the firmware drives red on its own pin.
    """
    try:
        if status == "ready":
            robot.led_red_off()
            robot.led_yellow_off()
            robot.led_green_on()
        elif status == "busy":
            robot.led_green_off()
            robot.led_red_off()
            robot.led_yellow_on()
        elif status == "error":
            robot.led_green_off()
            robot.led_yellow_off()
            robot.led_red_on()
    except Exception as e:
        print(f"[led] could not set status '{status}': {e}")


# ---------------------------------------------------------------------------
# Front-panel button listener
# ---------------------------------------------------------------------------

def run_button_listener(green_screen: bool = False):
    """Wait for front-panel button presses (serial events from the Pico).

    Button 1 homes the robot; button 2 runs the full capture → solve → move
    pipeline. If the robot was already homed via button 1 (and no run has
    happened since), button 2 skips the homing step.

    Pico event callbacks fire inside PicoInterface's serial-listener thread,
    which is also the thread that receives RPC responses — so the actual work
    is dispatched to a separate worker thread to avoid deadlocking the
    response wait. A busy guard ignores presses while a task is in progress.
    """
    import threading
    import time
    from robot.pico_interface import PicoInterface

    _setup_debug_dir()
    robot = PicoInterface(port=ROBOT_PORT)

    lock  = threading.Lock()
    state = {"homed": False, "busy": False}

    def _dispatch(label, fn):
        def worker():
            set_status_led(robot, "busy")          # orange while operating
            try:
                fn()
            except Exception as e:
                print(f"[buttons] {label} failed: {e}")
                set_status_led(robot, "error")     # red on crash
            else:
                set_status_led(robot, "ready")     # green when done → ready
            finally:
                with lock:
                    state["busy"] = False
        with lock:
            if state["busy"]:
                print(f"[buttons] busy — ignoring {label}")
                return
            state["busy"] = True
        threading.Thread(target=worker, daemon=True).start()

    def on_home(msg=None):
        def task():
            home_robot(robot)
            with lock:
                state["homed"] = True
        _dispatch("home", task)

    def on_start(msg=None):
        def task():
            with lock:
                already_homed = state["homed"]
            _run_solve(robot, green_screen=green_screen, skip_home=already_homed)
            # A completed run ends with motors disabled, so the robot is no
            # longer at a known home position.
            with lock:
                state["homed"] = False
        _dispatch("start", task)

    robot.on_button(BUTTON_HOME_PIN, on_home)
    robot.on_button(BUTTON_START_PIN, on_start)

    set_status_led(robot, "ready")   # green: ready for input

    print(f"Listening for front buttons — pin {BUTTON_HOME_PIN}=home, "
          f"pin {BUTTON_START_PIN}=start. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping button listener")
    finally:
        robot.led_green_off()
        robot.led_yellow_off()
        robot.led_red_off()
        robot.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PREN Puzzlesolver")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--gui",  action="store_true", help="open GUI viewer")
    mode.add_argument("--prod", action="store_true", help="production pipeline: capture → solve → move")
    mode.add_argument("--buttons", action="store_true",
                      help="wait for front-panel buttons (button 1 = home, button 2 = start pipeline)")
    mode.add_argument("--pickup-test", action="store_true",
                      help="place one piece; robot detects, picks it up and holds it (accuracy check)")
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

    elif args.buttons:
        run_button_listener(green_screen=args.green_screen)

    elif args.pickup_test:
        run_pickup_test(green_screen=args.green_screen)

    else:  # --prod
        run_pipeline(green_screen=args.green_screen)


if __name__ == "__main__":
    main()
