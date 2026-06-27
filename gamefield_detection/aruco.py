from .config import ARUCO_TAG_IDS, BORDER_OUTPUT_W, BORDER_OUTPUT_H, \
    OFFSET_LEFT, OFFSET_RIGHT, OFFSET_TOP, OFFSET_BOTTOM


def _save_failed_debug(frame, corners, ids, rejected, debug_dir):
    """Save an annotated image when ArUco detection fails, to aid diagnosis."""
    import cv2
    import cv2.aruco as aruco
    dbg = frame.copy()
    if corners:
        aruco.drawDetectedMarkers(dbg, corners, ids)
    for rj in rejected:
        pts = rj[0].astype(int)
        for i in range(4):
            cv2.line(dbg, tuple(pts[i]), tuple(pts[(i + 1) % 4]), (0, 0, 200), 1)
    import os
    cv2.imwrite(os.path.join(debug_dir, "capture_aruco_failed.jpg"), dbg)
    print(f"[ArUco] failed-detection debug saved to {debug_dir}/capture_aruco_failed.jpg")


def _build_detector():
    """Create an ArucoDetector tuned for uneven / glary lighting.

    The default adaptive-threshold window range is narrow (3..23), which fails
    when one corner of the field is darker (shadow) or brighter (glare) than the
    rest. Widening the window range lets the binarizer adapt locally, and
    subpixel corner refinement keeps the warp accurate.
    """
    import cv2.aruco as aruco

    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    params = aruco.DetectorParameters()
    params.detectInvertedMarker = True
    # Adapt the binarization threshold over a much wider range of window sizes so
    # a shadowed/over-lit corner still produces a clean black/white tag.
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 53
    params.adaptiveThreshWinSizeStep = 8
    # Be a little more permissive about tag size/shape so a partially washed-out
    # marker still passes the candidate filter.
    params.minMarkerPerimeterRate = 0.02
    params.polygonalApproxAccuracyRate = 0.05
    # Refine corners to subpixel accuracy for a cleaner perspective warp.
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    return aruco.ArucoDetector(dictionary, params)


def _detect_markers_robust(frame, detector, required_ids):
    """Detect markers, retrying with lighting-normalized variants of *frame*.

    Runs detection on progressively contrast-/brightness-adjusted grayscale
    versions and accumulates markers by ID until every id in *required_ids* is
    found. Returns (tag_map, corners_list, ids_list, last_rejected) where
    tag_map maps id -> 4×2 corner array (first detection of each id wins).
    """
    import cv2
    import numpy as np

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) \
        if frame.ndim == 3 else frame
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    # Ordered so the cheapest/most-faithful pass runs first.
    passes = [
        ("raw", gray),
        ("clahe", clahe.apply(gray)),
        ("bright", cv2.convertScaleAbs(gray, alpha=1.4, beta=40)),
    ]

    tag_map = {}
    last_rejected = []
    for name, img in passes:
        corners, ids, rejected = detector.detectMarkers(img)
        last_rejected = rejected
        new_here = []
        if ids is not None:
            for tid, corn in zip(ids.flatten().tolist(), corners):
                if tid not in tag_map:
                    tag_map[tid] = corn[0]
                    new_here.append(tid)
        print(f"[ArUco]   pass '{name}': +{len(new_here)} new {new_here}  "
              f"(total {sorted(tag_map)})")
        if all(i in tag_map for i in required_ids):
            break

    corners_list = [tm.reshape(1, 4, 2).astype(np.float32)
                    for tm in tag_map.values()]
    ids_list = np.array([[t] for t in tag_map.keys()], dtype=np.int32) \
        if tag_map else None
    return tag_map, corners_list, ids_list, last_rejected


def _compute_aruco_transform(frame, debug_dir):
    """Detect 4 ArUco markers in *frame* and return the perspective transform.

    Returns (M, W, H) where M is the 3×3 warp matrix, or None on failure.
    Saves debug images to *debug_dir*.
    """
    import cv2
    import cv2.aruco as aruco
    import numpy as np
    import os

    cv2.imwrite(os.path.join(debug_dir, "capture_aruco_input.jpg"), frame)

    detector = _build_detector()

    tag_map, corners, ids, rejected = _detect_markers_robust(
        frame, detector, ARUCO_TAG_IDS)

    found_ids = ids.flatten().tolist() if ids is not None else []
    print(f"[ArUco] detected {len(found_ids)} marker(s): {found_ids}  "
          f"(rejected candidates: {len(rejected)})")

    if not all(i in tag_map for i in ARUCO_TAG_IDS):
        missing = [i for i in ARUCO_TAG_IDS if i not in tag_map]
        print(f"[ArUco] missing required IDs {missing} (have {sorted(tag_map)})")
        _save_failed_debug(frame, corners, ids, rejected, debug_dir)
        return None

    ids_flat = ids.flatten().tolist()

    # Save annotated debug image showing detected edges and reference corners
    debug_frame = frame.copy()
    aruco.drawDetectedMarkers(debug_frame, corners, ids)
    ref_corner_idx_debug = {0: 3, 1: 2, 2: 1, 3: 0}
    corner_colors = [(0, 255, 0), (0, 165, 255), (0, 0, 255), (255, 0, 0)]
    for tid, corn in zip(ids_flat, corners):
        pts = corn[0].astype(int)
        cx, cy = pts.mean(axis=0).astype(int)
        cv2.putText(debug_frame, f"ID {tid}", (cx - 20, cy - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        if tid in ref_corner_idx_debug:
            ref_idx = ref_corner_idx_debug[tid]
            rx, ry = pts[ref_idx]
            color = corner_colors[ARUCO_TAG_IDS.index(tid)]
            cv2.circle(debug_frame, (rx, ry), 8, color, -1)
            cv2.putText(debug_frame, f"c{ref_idx}", (rx + 10, ry),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.imwrite(os.path.join(debug_dir, "capture_aruco_debug.jpg"), debug_frame)
    print(f"[ArUco] debug image saved: {debug_dir}/capture_aruco_debug.jpg")

    # corner indices: each corners array is [TL, TR, BR, BL] of the tag.
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

    src_pts[0] += [ OFFSET_LEFT,   OFFSET_TOP]
    src_pts[1] += [-OFFSET_RIGHT,  OFFSET_TOP]
    src_pts[2] += [-OFFSET_RIGHT, -OFFSET_BOTTOM]
    src_pts[3] += [ OFFSET_LEFT,  -OFFSET_BOTTOM]

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return M, BORDER_OUTPUT_W, BORDER_OUTPUT_H


def detect_aruco_border(frame):
    """Detect ArUco markers and warp *frame*. Returns warped image or None."""
    import cv2
    import os

    debug_dir = os.environ.get("ZOLVER_TEMP_DIR", "debug_output")
    result = _compute_aruco_transform(frame, debug_dir)
    if result is None:
        return None
    M, W, H = result
    warped = cv2.warpPerspective(frame, M, (W, H))
    border_path = os.path.join(debug_dir, "capture_with_border.jpg")
    cv2.imwrite(border_path, warped)
    print(f"[ArUco] warp saved: {border_path}  ({W}×{H} px)")
    return warped


def detect_aruco_border_two_frame(detect_frame, warp_frame):
    """Compute ArUco transform from *detect_frame*, apply it to *warp_frame*.

    Use this when the LED is off for detection but on for the puzzle image.
    Returns warped *warp_frame* or None if markers were not found.
    """
    import cv2
    import os

    debug_dir = os.environ.get("ZOLVER_TEMP_DIR", "debug_output")
    result = _compute_aruco_transform(detect_frame, debug_dir)
    if result is None:
        return None
    M, W, H = result
    warped = cv2.warpPerspective(warp_frame, M, (W, H))
    border_path = os.path.join(debug_dir, "capture_with_border.jpg")
    cv2.imwrite(border_path, warped)
    print(f"[ArUco] two-frame warp saved: {border_path}  ({W}×{H} px)")
    return warped
