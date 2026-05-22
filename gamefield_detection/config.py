BORDER_DETECTION = True

BORDER_OUTPUT_W = 906
BORDER_OUTPUT_H = 648

INSET = 6  # px trimmed on each side to remove residual border slither

ARUCO_TAG_IDS = [0, 1, 2, 3]  # TL, TR, BR, BL  (DICT_4X4_50)

# Per-side crop offsets in pixels. Positive shrinks the crop inward (includes less),
# negative expands it outward (includes more).
OFFSET_LEFT   = 15
OFFSET_RIGHT  = 10
OFFSET_TOP    = 40
OFFSET_BOTTOM = 40
