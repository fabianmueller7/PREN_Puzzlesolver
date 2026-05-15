# Toggle: set to False to skip red-border detection and use the static crop only.
BORDER_DETECTION = True

BORDER_OUTPUT_W = 906
BORDER_OUTPUT_H = 648


def detect_a4_border(frame):
    """Detect the red A4 frame, correct rotation with an affine rotate, and crop inside.
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

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Find the largest outer contour (the red ring itself)
    outer = [(i, cv2.contourArea(c)) for i, c in enumerate(contours)
             if hierarchy[0][i][3] == -1]
    if not outer:
        return None
    ring_idx, ring_area = max(outer, key=lambda x: x[1])
    if ring_area < 10_000:
        return None

    # Prefer the inner hole so the red border is excluded from the output
    child_idx = hierarchy[0][ring_idx][2]
    inner_cnt = contours[child_idx] if child_idx != -1 else contours[ring_idx]

    center, (rw, rh), angle = cv2.minAreaRect(inner_cnt)
    # minAreaRect angle is in [-90, 0); ensure the long side is horizontal
    if rw < rh:
        angle += 90

    # Affine rotation — rigid transform, no shear/scale distortion
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]),
                             flags=cv2.INTER_LINEAR)

    # After rotation the inner rect is axis-aligned around the same center.
    # Inset by BORDER_INSET px on each side to trim any residual red slither.
    BORDER_INSET = 6
    cw, ch = int(max(rw, rh)) - 2 * BORDER_INSET, int(min(rw, rh)) - 2 * BORDER_INSET
    cx, cy = int(round(center[0])), int(round(center[1]))
    x0 = max(cx - cw // 2, 0)
    y0 = max(cy - ch // 2, 0)
    x1 = min(x0 + cw, rotated.shape[1])
    y1 = min(y0 + ch, rotated.shape[0])

    return cv2.resize(rotated[y0:y1, x0:x1], (BORDER_OUTPUT_W, BORDER_OUTPUT_H),
                      interpolation=cv2.INTER_AREA)
