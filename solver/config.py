DEBUG_FILE_OUTPUT = 1    # Saves debug images/data to debug_output/ and runs processing sequentially
DEBUG_SHOW_DIAGRAMS = 0  # Shows matplotlib diagrams interactively (requires DEBUG_FILE_OUTPUT = 1)
DEBUG_ALT_SOLVER = 0     # Saves debug images for the alternative solver to debug_output/
DEBUG_PIECE_CENTERS = 1  # Writes piece_centers.json to debug_output/ with each piece's start and end center point (0/0 is top left)

EDGE_OFFSET = 0  # pixels (6 pixels ≈ 1mm) — shifts each edge outward to show the manufacturing tolerance band in debug output

# Affine calibration: maps cropped-image pixel (px, py) → robot mm (rx, ry).
# Robot zero is top-right; X increases to the left, Y increases downward.
# Crop region: CROP_X=503, CROP_Y=300 (906×648 px).
# Calibrated against 3 measured robot positions:
#   Unten links  px=(242,391) → robot=(222,307)
#   Unten rechts px=(772,366) → robot=( 50,300)
#   Oben Mitte   px=(452,163) → robot=(155,234)
#   robot_x = CAL_M[0][0]*px + CAL_M[0][1]*py + CAL_M[0][2]
#   robot_y = CAL_M[1][0]*px + CAL_M[1][1]*py + CAL_M[1][2]
CAL_M = [
    [-0.324777, -0.005277, 302.659503],
    [ 0.001981,  0.322000, 180.618496],
]


def pixel_to_robot(pixel_x, pixel_y):
    """Convert image pixel coordinates to robot mm coordinates."""
    rx = CAL_M[0][0] * pixel_x + CAL_M[0][1] * pixel_y + CAL_M[0][2]
    ry = CAL_M[1][0] * pixel_x + CAL_M[1][1] * pixel_y + CAL_M[1][2]
    return round(rx), round(ry)

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
