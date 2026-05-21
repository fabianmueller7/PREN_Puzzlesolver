from .config import ARUCO_TAG_IDS, BORDER_OUTPUT_W, BORDER_OUTPUT_H, \
    OFFSET_LEFT, OFFSET_RIGHT, OFFSET_TOP, OFFSET_BOTTOM


def detect_aruco_border(frame):
    """Detect 4 ArUco corner markers on the LED frame and perspective-warp the playfield.

    Tags sit on the long-side rails of the frame, aligned with the paper corners.
    Crop boundary: outer end lengthwise; lower edge for top marks, upper edge
    for bottom marks — so the crop spans the inner facing edges of all 4 tags.
      Tag 0 (TL) → corner[3] (bottom-left of tag)
      Tag 1 (TR) → corner[2] (bottom-right of tag)
      Tag 2 (BR) → corner[1] (top-right of tag)
      Tag 3 (BL) → corner[0] (top-left of tag)

    Returns the warped 906×648 image, or None if not all 4 tags are detected.
    """
    import cv2
    import cv2.aruco as aruco
    import numpy as np
    import os

    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, params)

    corners, ids, _ = detector.detectMarkers(frame)

    if ids is None or len(ids) < 4:
        return None

    ids_flat = ids.flatten().tolist()
    if not all(i in ids_flat for i in ARUCO_TAG_IDS):
        return None

    tag_map = {int(tid): corn[0] for tid, corn in zip(ids_flat, corners)}

    # corner indices: each corners array is [TL, TR, BR, BL] of the tag.
    # Lengthwise (horizontal): outer end of each mark.
    # Heightwise (vertical): lower edge for top marks, upper edge for bottom marks.
    #   Tag 0 (TL) → corner[3] (bottom-left of tag)
    #   Tag 1 (TR) → corner[2] (bottom-right of tag)
    #   Tag 2 (BR) → corner[1] (top-right of tag)
    #   Tag 3 (BL) → corner[0] (top-left of tag)
    ref_corner_idx = {0: 3, 1: 2, 2: 1, 3: 0}
    src_pts = np.float32([
        tag_map[tid][ref_corner_idx[tid]] for tid in ARUCO_TAG_IDS
    ])

    dst_pts = np.float32([
        [0, 0],
        [BORDER_OUTPUT_W - 1, 0],
        [BORDER_OUTPUT_W - 1, BORDER_OUTPUT_H - 1],
        [0, BORDER_OUTPUT_H - 1],
    ])

    src_pts[0] += [ OFFSET_LEFT,  OFFSET_TOP]
    src_pts[1] += [-OFFSET_RIGHT, OFFSET_TOP]
    src_pts[2] += [-OFFSET_RIGHT, -OFFSET_BOTTOM]
    src_pts[3] += [ OFFSET_LEFT,  -OFFSET_BOTTOM]

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(frame, M, (BORDER_OUTPUT_W, BORDER_OUTPUT_H))

    debug_dir = os.environ.get("ZOLVER_TEMP_DIR", "debug_output")
    border_path = os.path.join(debug_dir, "capture_with_border.jpg")
    cv2.imwrite(border_path, warped)
    print(f"[1/3] ArUco warp saved:        {border_path}  ({warped.shape[1]}×{warped.shape[0]} px)")

    return warped
