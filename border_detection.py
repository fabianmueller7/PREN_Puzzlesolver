# Toggle: set to False to skip red-border detection and use the static crop only.
BORDER_DETECTION = False

BORDER_OUTPUT_W = 906
BORDER_OUTPUT_H = 648


def order_corners(pts):
    """Return 4 points sorted as [top-left, top-right, bottom-right, bottom-left]."""
    import numpy as np
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    return np.array([
        pts[s.argmin()],
        pts[diff.argmin()],
        pts[s.argmax()],
        pts[diff.argmax()],
    ], dtype=np.float32)


def detect_a4_border(frame):
    """Detect the red A4 frame and return a perspective-warped fixed-size crop.
    Returns None when no clear red rectangle is found."""
    import cv2
    import numpy as np

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Red wraps around 0°/180° in HSV — combine both ranges
    mask = cv2.bitwise_or(
        cv2.inRange(hsv, (0,   70, 70), (10,  255, 255)),
        cv2.inRange(hsv, (170, 70, 70), (180, 255, 255)),
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 10_000:
        return None

    pts = cv2.boxPoints(cv2.minAreaRect(cnt)).astype(np.float32)
    pts = order_corners(pts)

    dst = np.array([
        [0,                    0],
        [BORDER_OUTPUT_W - 1,  0],
        [BORDER_OUTPUT_W - 1,  BORDER_OUTPUT_H - 1],
        [0,                    BORDER_OUTPUT_H - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(frame, M, (BORDER_OUTPUT_W, BORDER_OUTPUT_H))
