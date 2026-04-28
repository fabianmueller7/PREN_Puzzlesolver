DEBUG_FILE_OUTPUT = 1    # Saves debug images/data to debug_output/ and runs processing sequentially
DEBUG_SHOW_DIAGRAMS = 0  # Shows matplotlib diagrams interactively (requires DEBUG_FILE_OUTPUT = 1)
DEBUG_ALT_SOLVER = 0     # Saves debug images for the alternative solver to debug_output/
DEBUG_PIECE_CENTERS = 1  # Writes piece_centers.json to debug_output/ with each piece's start and end center point (0/0 is top left)

EDGE_OFFSET = 12  # pixels (6 pixels ≈ 1mm) — shifts each edge outward to show the manufacturing tolerance band in debug output

# A4 landscape at 150 DPI
DEBUG_OUTPUT_W = 1782
DEBUG_OUTPUT_H = 1260


def save_debug_img(path, img):
    """Center img on a fixed A4 landscape canvas (1782×1260) without scaling.
    Rotates 90° first when the image is taller than wide (portrait) so it aligns
    with the landscape orientation. Crops to canvas bounds if the image is larger."""
    import cv2
    import numpy as np
    h, w = img.shape[:2]
    if h > w:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        h, w = img.shape[:2]
    channels = img.shape[2] if len(img.shape) == 3 else 1
    shape = (DEBUG_OUTPUT_H, DEBUG_OUTPUT_W) if channels == 1 else (DEBUG_OUTPUT_H, DEBUG_OUTPUT_W, channels)
    canvas = np.zeros(shape, dtype=np.uint8)
    # compute centered placement, clipping to canvas bounds
    y_off = max((DEBUG_OUTPUT_H - h) // 2, 0)
    x_off = max((DEBUG_OUTPUT_W - w) // 2, 0)
    src_h = min(h, DEBUG_OUTPUT_H - y_off)
    src_w = min(w, DEBUG_OUTPUT_W - x_off)
    canvas[y_off:y_off + src_h, x_off:x_off + src_w] = img[:src_h, :src_w]
    cv2.imwrite(path, canvas)
