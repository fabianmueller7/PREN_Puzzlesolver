from .config import ARUCO_TAG_IDS, BORDER_OUTPUT_W, BORDER_OUTPUT_H, \
    OFFSET_LEFT, OFFSET_RIGHT, OFFSET_TOP, OFFSET_BOTTOM


def _save_failed_debug(frame, corners, ids, rejected, debug_dir):
    """Save an annotated image when ArUco detection fails, to aid diagnosis."""
    import cv2
    import cv2.aruco as aruco
    dbg = frame.copy()
    if corners:
        aruco.drawDetectedMarkers(dbg, corners, ids)
    # Draw rejected candidates in red so we can see near-misses
    for rj in rejected:
        import numpy as np
        pts = rj[0].astype(int)
        for i in range(4):
            cv2.line(dbg, tuple(pts[i]), tuple(pts[(i + 1) % 4]), (0, 0, 200), 1)
    import os
    cv2.imwrite(os.path.join(debug_dir, "capture_aruco_failed.jpg"), dbg)
    print(f"[ArUco] failed-detection debug saved to {debug_dir}/capture_aruco_failed.jpg")


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
    params.detectInvertedMarker = True
    detector = aruco.ArucoDetector(dictionary, params)

    corners, ids, rejected = detector.detectMarkers(frame)

    debug_dir = os.environ.get("ZOLVER_TEMP_DIR", "debug_output")

    # Always save the input frame so we can inspect what the detector sees
    cv2.imwrite(os.path.join(debug_dir, "capture_aruco_input.jpg"), frame)

    found_ids = ids.flatten().tolist() if ids is not None else []
    print(f"[ArUco] detected {len(found_ids)} marker(s): {found_ids}  "
          f"(rejected candidates: {len(rejected)})")

    if ids is None or len(ids) < 4:
        _save_failed_debug(frame, corners, ids, rejected, debug_dir)
        return None

    ids_flat = ids.flatten().tolist()
    if not all(i in ids_flat for i in ARUCO_TAG_IDS):
        print(f"[ArUco] need IDs {ARUCO_TAG_IDS}, got {ids_flat}")
        _save_failed_debug(frame, corners, ids, rejected, debug_dir)
        return None

    tag_map = {int(tid): corn[0] for tid, corn in zip(ids_flat, corners)}

    # --- debug: draw tag edges on a copy of the original frame ---
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
    debug_aruco_path = os.path.join(debug_dir, "capture_aruco_debug.jpg")
    cv2.imwrite(debug_aruco_path, debug_frame)
    print(f"[1/3] ArUco debug image saved: {debug_aruco_path}")

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

    border_path = os.path.join(debug_dir, "capture_with_border.jpg")
    cv2.imwrite(border_path, warped)
    print(f"[1/3] ArUco warp saved:        {border_path}  ({warped.shape[1]}×{warped.shape[0]} px)")

    return warped
