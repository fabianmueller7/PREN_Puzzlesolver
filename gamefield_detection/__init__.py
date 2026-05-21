from .config import BORDER_DETECTION, BORDER_OUTPUT_W, BORDER_OUTPUT_H
from .aruco import detect_aruco_border
from .white import detect_white_border


def detect_a4_border(frame):
    """Detect the playfield and return a normalised 906×648 crop.

    Tries ArUco corner markers first; falls back to white-rectangle detection
    when fewer than 4 tags are visible.
    Returns None when neither method finds a clear playfield.
    """
    result = detect_aruco_border(frame)
    if result is not None:
        return result

    print("[WARN] ArUco tags not found — falling back to white-border detection")
    return detect_white_border(frame)
