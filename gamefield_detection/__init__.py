from .config import BORDER_DETECTION, BORDER_OUTPUT_W, BORDER_OUTPUT_H
from .aruco import detect_aruco_border, detect_aruco_border_two_frame
from .white import detect_white_border


def detect_a4_border(frame):
    """Detect ArUco markers and warp *frame*; falls back to white-border detection."""
    result = detect_aruco_border(frame)
    if result is not None:
        return result

    print("[WARN] ArUco tags not found — falling back to white-border detection")
    return detect_white_border(frame)


def detect_a4_border_two_frame(detect_frame, warp_frame):
    """Compute transform from *detect_frame* (LED off), apply to *warp_frame* (LED on).

    Falls back to white-border detection on *warp_frame* if ArUco fails.
    """
    result = detect_aruco_border_two_frame(detect_frame, warp_frame)
    if result is not None:
        return result

    print("[WARN] ArUco tags not found — falling back to white-border detection on lit frame")
    return detect_white_border(warp_frame)
