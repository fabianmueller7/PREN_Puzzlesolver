DEBUG_FILE_OUTPUT = 1    # Saves debug images/data to debug_output/ and runs processing sequentially
DEBUG_SHOW_DIAGRAMS = 0  # Shows matplotlib diagrams interactively (requires DEBUG_FILE_OUTPUT = 1)
DEBUG_ALT_SOLVER = 0     # Saves debug images for the alternative solver to debug_output/
DEBUG_PIECE_CENTERS = 1  # Writes piece_centers.json to debug_output/ with each piece's start and end center point (0/0 is top left)

EDGE_OFFSET = 1  # pixels (6 pixels ≈ 1mm) — shifts each edge outward to show the manufacturing tolerance band in debug output

# Affine calibration: maps warped-image pixel (px, py) → robot mm (rx, ry).
# Both systems: (0,0) = top-left, X increases right, Y increases down.
# Output image: 906×648 px (ArUco warp, playing field fills the image exactly).
# Least-squares fit from 4 measured playing-field corners (residual ±0.9 mm):
#   Oben rechts  px=(905,   0) → robot=(  6.5, 176)
#   Oben links   px=(  0,   0) → robot=(303.0, 175)
#   Unten links  px=(  0, 647) → robot=(303.0, 386)
#   Unten rechts px=(905, 647) → robot=(  3.0, 383)
#   robot_x = CAL_M[0][0]*px + CAL_M[0][1]*py + CAL_M[0][2]
#   robot_y = CAL_M[1][0]*px + CAL_M[1][1]*py + CAL_M[1][2]
CAL_M = [
    [-0.329558, -0.002705, 303.875],
    [-0.001105,  0.323029, 177.500],
]

# Fine-tune knobs: scale positions outward from CAL_CENTRE after the affine transform.
# 1.0 = no correction. Increase to push outward; decrease to pull inward.
# Tune in steps of 0.005 (~0.5 mm per 100 mm from centre).
CAL_CENTRE_X = 153.88  # robot mm  (midpoint of the 4 corners)
CAL_CENTRE_Y = 280.00  # robot mm
CAL_SCALE_X  = 1.0
CAL_SCALE_Y  = 1.0


def pixel_to_robot(pixel_x, pixel_y):
    """Convert image pixel coordinates to robot mm coordinates."""
    rx = CAL_M[0][0] * pixel_x + CAL_M[0][1] * pixel_y + CAL_M[0][2]
    ry = CAL_M[1][0] * pixel_x + CAL_M[1][1] * pixel_y + CAL_M[1][2]
    rx = CAL_CENTRE_X + (rx - CAL_CENTRE_X) * CAL_SCALE_X
    ry = CAL_CENTRE_Y + (ry - CAL_CENTRE_Y) * CAL_SCALE_Y
    return round(rx), round(ry)


# A5 target field — all values in robot mm.
#
# Coordinate system: robot X increases to the LEFT, robot Y increases DOWNWARD.
#   ge (east)  = column index — X decreases as ge increases (rightward)
#   gn (north) = row index    — Y decreases as gn increases (upward)
#
# Anchor = centre of the piece slot at (ge=0, gn=0) = bottom-left of the target grid.
#
# Calibration procedure:
#   1. Jog robot to centre of bottom-left slot  → set A5_ANCHOR_X, A5_ANCHOR_Y
#   2. Jog to centre of next slot eastward      → A5_CELL_W = ANCHOR_X - that X
#   3. Jog to centre of next slot northward     → A5_CELL_H = ANCHOR_Y - that Y
A5_ANCHOR_X = 203   # TODO: measure physically
A5_ANCHOR_Y = 105   # TODO: measure physically
A5_CELL_W   = 94    # TODO: measure physically
A5_CELL_H   = 64    # TODO: measure physically

# Global rotation offset added to every piece's rotation_deg (degrees, CCW-positive).
# Start at 0. Tune in 90° steps once positions are correct.
PUZZLE_TARGET_ROTATION_DEG = 0.0


def grid_to_robot(ge, gn, grid_W, grid_H):
    """Map solved-puzzle grid coordinate (ge=east, gn=north) to robot mm."""
    rx = round(A5_ANCHOR_X - ge * A5_CELL_W)
    ry = round(A5_ANCHOR_Y - gn * A5_CELL_H)
    return rx, ry

# A4 landscape at 150 DPI
DEBUG_OUTPUT_W = 1782
DEBUG_OUTPUT_H = 1260


def save_debug_img(path, img):
    """Place img on a fixed A4 landscape canvas (1782×1260).
    Rotates 90° when portrait to align with the landscape orientation.
    Shown at 1:1 when it fits; scaled down (aspect-ratio preserved) only when larger."""
    import cv2
    import numpy as np
    h, w = img.shape[:2]
    if h > w:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        h, w = img.shape[:2]
    if h > DEBUG_OUTPUT_H or w > DEBUG_OUTPUT_W:
        scale = min(DEBUG_OUTPUT_W / w, DEBUG_OUTPUT_H / h)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
    channels = img.shape[2] if len(img.shape) == 3 else 1
    shape = (DEBUG_OUTPUT_H, DEBUG_OUTPUT_W) if channels == 1 else (DEBUG_OUTPUT_H, DEBUG_OUTPUT_W, channels)
    canvas = np.zeros(shape, dtype=np.uint8)
    y_off = (DEBUG_OUTPUT_H - h) // 2
    x_off = (DEBUG_OUTPUT_W - w) // 2
    canvas[y_off:y_off + h, x_off:x_off + w] = img
    cv2.imwrite(path, canvas)
