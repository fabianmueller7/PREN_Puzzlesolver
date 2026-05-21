# Toggle: set to False to skip playfield detection and use the static crop only.
BORDER_DETECTION = True

BORDER_OUTPUT_W = 906
BORDER_OUTPUT_H = 648

INSET = 6  # px trimmed on each side to remove residual border slither


def detect_a4_border(frame):
    """Detect the white playfield background, correct rotation, and crop to it.
    Returns None when no clear white rectangle is found."""
    import cv2
    import os

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # White = low saturation, high brightness
    mask = cv2.inRange(hsv, (0, 0, 160), (180, 60, 255))
    # Erode first to eliminate thin bright regions outside the playfield,
    # then close to fill holes left by dark puzzle pieces.
    erode_k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 80))
    mask = cv2.erode(mask, erode_k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 50_000:
        return None

    center, (rw, rh), angle = cv2.minAreaRect(largest)
    if rw < rh:
        angle += 90

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]),
                             flags=cv2.INTER_LINEAR)

    debug_dir = os.environ.get("ZOLVER_TEMP_DIR", "debug_output")
    border_path = os.path.join(debug_dir, "capture_with_border.jpg")
    cv2.imwrite(border_path, rotated)
    print(f"[1/3] Rotated (pre-crop):      {border_path}  ({rotated.shape[1]}×{rotated.shape[0]} px)")

    iw, ih = max(rw, rh), min(rw, rh)
    cx, cy = center[0], center[1]
    x0 = max(int(cx - iw / 2) + INSET, 0)
    x1 = min(int(cx + iw / 2) - INSET, rotated.shape[1])
    y0 = max(int(cy - ih / 2) + INSET, 0)
    y1 = min(int(cy + ih / 2) - INSET, rotated.shape[0])

    return cv2.resize(rotated[y0:y1, x0:x1], (BORDER_OUTPUT_W, BORDER_OUTPUT_H),
                      interpolation=cv2.INTER_AREA)
