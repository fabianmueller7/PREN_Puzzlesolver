DEBUG_FILE_OUTPUT = 1    # Saves debug images/data to debug_output/ and runs processing sequentially
DEBUG_SHOW_DIAGRAMS = 0  # Shows matplotlib diagrams interactively (requires DEBUG_FILE_OUTPUT = 1)
DEBUG_ALT_SOLVER = 0     # Saves debug images for the alternative solver to debug_output/
DEBUG_PIECE_CENTERS = 1  # Writes piece_centers.json to debug_output/ with each piece's start and end center point (0/0 is top left)

EDGE_OFFSET = 0  # pixels (6 pixels ≈ 1mm) — shifts each edge outward to show the manufacturing tolerance band in debug output

# A4 landscape at 150 DPI
DEBUG_OUTPUT_W = 1782
DEBUG_OUTPUT_H = 1260


def save_debug_img(path, img):
    """Save img scaled to fit the fixed A4 landscape canvas (1782×1260), letterboxed with black."""
    import cv2
    import numpy as np
    h, w = img.shape[:2]
    scale = min(DEBUG_OUTPUT_W / w, DEBUG_OUTPUT_H / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    channels = img.shape[2] if len(img.shape) == 3 else 1
    shape = (DEBUG_OUTPUT_H, DEBUG_OUTPUT_W) if channels == 1 else (DEBUG_OUTPUT_H, DEBUG_OUTPUT_W, channels)
    canvas = np.zeros(shape, dtype=np.uint8)
    y_off = (DEBUG_OUTPUT_H - new_h) // 2
    x_off = (DEBUG_OUTPUT_W - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    cv2.imwrite(path, canvas)
