import itertools
import math
import os
from multiprocessing import Pool, cpu_count

import numpy as np
import cv2
import scipy
from .. import config

from ..Puzzle.Edge import Edge
from ..Puzzle.Enums import directions, TypeEdge
from ..Puzzle.PuzzlePiece import PuzzlePiece
from .peak_detect import detect_peaks
from ..Puzzle.Distance import rgb2hsl


def get_relative_angles(cnt, sigma=5):
    length = len(cnt)
    angles = []
    last = np.pi

    cnt_tmp = np.array(cnt)
    cnt = np.append(cnt, cnt_tmp, axis=0)
    cnt = np.append(cnt, cnt_tmp, axis=0)
    for i in range(0, len(cnt) - 1):
        d = (cnt[i + 1][0] - cnt[i][0], cnt[i + 1][1] - cnt[i][1])
        angle = math.atan2(-d[1], d[0])
        while angle < last - np.pi:
            angle += 2 * np.pi
        while angle > last + np.pi:
            angle -= 2 * np.pi
        angles.append(angle)
        last = angle

    angles = np.diff(angles)
    k = [0.33, 0.33, 0.33, 0.33, 0.33]
    angles = scipy.ndimage.convolve(angles, k, mode="constant", cval=0.0)
    angles = scipy.ndimage.filters.gaussian_filter(angles, sigma)
    angles = np.roll(np.array(angles), -length)
    return angles[0:length]


def is_maximum_local(index, relative_angles, radius):
    start = max(0, index - radius)
    end = min(relative_angles.shape[0] - 1, index + radius)
    for i in range(start, end + 1):
        if relative_angles[i] > relative_angles[index]:
            return False
    return True


def longest_peak(relative_angles):
    length = relative_angles.shape[0]
    longest = (0, 0)
    j = 0
    for i in range(length):
        if relative_angles[i] >= 0:
            j = i
        if i - j > longest[1] - longest[0]:
            longest = (j, i)
    return longest


def distance_signature(relative_angles):
    flat_angles = relative_angles.flatten()
    length = flat_angles.shape[0]
    l1 = np.array([0, flat_angles[0]])
    l2 = np.array([length - 1, flat_angles[-1]])
    assert np.linalg.norm(l2 - l1) != 0
    signature = np.zeros((length, 1))
    for i in range(length):
        signature[i] = np.linalg.norm(
            np.cross(l2 - l1, l1 - np.array([i, flat_angles[i]]))
        ) / np.linalg.norm(l2 - l1)
    return signature


def flat_score(relative_angles):
    distances = distance_signature(relative_angles)
    return max(abs(distances[i]) for i in range(relative_angles.shape[0]))


def indent_score(relative_angles):
    length = relative_angles.shape[0]
    peak = longest_peak(relative_angles)

    while peak[0] > 0 and not is_maximum_local(peak[0], relative_angles, 10):
        peak = (peak[0] - 1, peak[1])
    while peak[1] < length - 1 and not is_maximum_local(peak[1], relative_angles, 10):
        peak = (peak[0], peak[1] + 1)

    shape = np.zeros((peak[0] + length - peak[1], 1))
    for i in range(peak[0] + 1):
        shape[i] = relative_angles[i]
    for i in range(peak[1], length):
        shape[i - peak[1] + peak[0]] = relative_angles[i]

    if shape.shape[0] == 1:
        return flat_score(relative_angles)
    return flat_score(shape)


def outdent_score(relative_angles):
    return indent_score(-relative_angles)


def compute_comp(combs_l, relative_angles, method="correlate"):
    results_glob = []
    for comb_t in combs_l:
        offset = len(relative_angles) - comb_t[3] - 1
        relative_angles_tmp = np.roll(relative_angles, offset)
        comb_t += offset
        comb_t = [
            (0, comb_t[0]),
            (comb_t[0], comb_t[1]),
            (comb_t[1], comb_t[2]),
            (comb_t[2], comb_t[3]),
        ]
        results_comp = []
        for comb in comb_t:
            hole, head, border = 0, 0, 0
            if method == "flat":
                hole = indent_score(np.ravel(np.array(relative_angles_tmp[comb[0]:comb[1]])))[0]
                head = outdent_score(np.ravel(np.array(relative_angles_tmp[comb[0]:comb[1]])))[0]
                border = flat_score(np.ravel(np.array(relative_angles_tmp[comb[0]:comb[1]])))[0]
            results_comp.append(min(hole, head) if hole != border else border)
        results_glob.append(np.sum(np.array(results_comp)))
    return np.argmin(np.array(results_glob))


def peaks_inside(comb, peaks):
    if len(comb) == 0:
        return []
    return [peak for peak in peaks if comb[0] < peak < comb[-1]]


def is_pattern(comb, peaks):
    cpt = len(peaks_inside(comb, peaks))
    return cpt == 0 or cpt == 2 or cpt == 3


def is_acceptable_comb(combs, peaks, length):
    offset = length - combs[3] - 1
    combs_tmp = combs + offset
    peaks_tmp = (peaks + offset) % length
    return (
        is_pattern([0, combs_tmp[0]], peaks_tmp)
        and is_pattern([combs_tmp[0], combs_tmp[1]], peaks_tmp)
        and is_pattern([combs_tmp[1], combs_tmp[2]], peaks_tmp)
        and is_pattern([combs_tmp[2], combs_tmp[3]], peaks_tmp)
    )


def type_peak(peaks_pos_inside, peaks_neg_inside):
    if len(peaks_pos_inside) == 0 and len(peaks_neg_inside) == 0:
        return TypeEdge.BORDER
    if len(peaks_inside(peaks_pos_inside, peaks_neg_inside)) == 2:
        return TypeEdge.HOLE
    if len(peaks_inside(peaks_neg_inside, peaks_pos_inside)) == 2:
        return TypeEdge.HEAD
    return TypeEdge.UNDEFINED


def _reclassify_undefined(t, pos_in, neg_in, segment_angles=None, flat_threshold=0.05):
    """Permissive fallback for UNDEFINED edges; used after the main sigma loop."""
    if t != TypeEdge.UNDEFINED:
        return t
    if len(peaks_inside(pos_in, neg_in)) >= 1:
        return TypeEdge.HOLE
    if len(peaks_inside(neg_in, pos_in)) >= 1:
        return TypeEdge.HEAD
    if len(pos_in) > 0 and len(neg_in) == 0:
        if segment_angles is not None:
            amp = np.max(np.abs(segment_angles))
            if config.DEBUG_FILE_OUTPUT == 1:
                print(f"[reclassify] single-pos peak, amp={amp:.4f}, thr={flat_threshold:.4f}")
            if amp < flat_threshold:
                return TypeEdge.BORDER
        return TypeEdge.HEAD
    if len(neg_in) > 0 and len(pos_in) == 0:
        if segment_angles is not None:
            amp = np.max(np.abs(segment_angles))
            if config.DEBUG_FILE_OUTPUT == 1:
                print(f"[reclassify] single-neg peak, amp={amp:.4f}, thr={flat_threshold:.4f}")
            if amp < flat_threshold:
                return TypeEdge.BORDER
        return TypeEdge.HOLE
    return TypeEdge.UNDEFINED


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, ord=order, axis=axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def assign_directions_from_geometry(edges_shape):
    if len(edges_shape) != 4:
        raise ValueError(f"Expected 4 edges, got {len(edges_shape)}")

    all_pts = np.vstack([np.asarray(s, dtype=float) for s in edges_shape])
    center = all_pts.mean(axis=0)

    edge_infos = []
    for i, shape in enumerate(edges_shape):
        pts = np.asarray(shape, dtype=float)
        if len(pts) < 2:
            raise ValueError(f"Edge {i} has too few points")
        midpoint = (pts[0] + pts[-1]) / 2.0
        dx = midpoint[0] - center[0]
        dy = midpoint[1] - center[1]
        edge_infos.append({"index": i, "dx": dx, "dy": dy, "angle": np.arctan2(dy, dx)})

    edge_infos.sort(key=lambda e: e["angle"])
    up_pos = min(range(4), key=lambda k: edge_infos[k]["dy"])
    ordered = edge_infos[up_pos:] + edge_infos[:up_pos]

    mids = [np.asarray(edges_shape[e["index"]], dtype=float) for e in ordered]
    mids = [(p[0] + p[-1]) / 2.0 for p in mids]
    area2 = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(mids, mids[1:] + mids[:1]))

    if area2 > 0:
        assigned_dirs = [directions[0], directions[3], directions[2], directions[1]]
    else:
        assigned_dirs = [directions[0], directions[1], directions[2], directions[3]]

    return {ordered[k]["index"]: assigned_dirs[k] for k in range(4)}


def _compute_signatures(contours, green):
    if config.DEBUG_FILE_OUTPUT == 1:
        return [my_find_corner_signature(cnt, green) for cnt in contours]
    with Pool(cpu_count()) as p:
        return p.starmap(my_find_corner_signature, zip(contours, itertools.repeat(green)))


def _check_piece_dimensions(contours, signatures):
    dims = []
    for idx, cnt in enumerate(contours):
        corners, _, _ = signatures[idx]
        if corners is None:
            dims.append(None)
            continue
        pts = np.array([cnt[int(c)][0] for c in corners], dtype=float)
        sides = [np.linalg.norm(pts[(i + 1) % 4] - pts[i]) for i in range(4)]
        short = min((sides[0] + sides[2]) / 2, (sides[1] + sides[3]) / 2)
        long_ = max((sides[0] + sides[2]) / 2, (sides[1] + sides[3]) / 2)
        dims.append((short, long_))

    valid = [d for d in dims if d is not None]
    if len(valid) <= 1:
        return
    med_short = float(np.median([d[0] for d in valid]))
    med_long  = float(np.median([d[1] for d in valid]))
    tol = 0.3
    for i, d in enumerate(dims):
        if d is None:
            continue
        short, long_ = d
        ok = (abs(short - med_short) / (med_short + 1e-9) <= tol
              and abs(long_ - med_long)  / (med_long  + 1e-9) <= tol)
        status = "OK" if ok else "OUTLIER — corner detection may be wrong"
        print(f"  [dim-check] Piece {i}: short={short:.0f}px (med {med_short:.0f}), "
              f"long={long_:.0f}px (med {med_long:.0f}) — {status}")


def _pack_pieces_grid(list_img, modulo, path):
    max_h = max(img.shape[0] for img in list_img)
    max_w = max(img.shape[1] for img in list_img)
    grid = np.zeros(
        [max_h * (len(list_img) // modulo + 1), max_w * modulo],
        dtype=np.uint8,
    )
    for idx, img in enumerate(list_img):
        r, c = (idx // modulo) * max_h, (idx % modulo) * max_w
        grid[r:r + img.shape[0], c:c + img.shape[1]] = img
    config.save_debug_img(path, grid)



def _rectangle_score(pts):
    """Sum of squared cosines of interior angles (0 = perfect rectangle).

    pts: (4, 2) float array of corner coordinates in contour traversal order.
    A perfect rectangle has cos(90°)=0 at every corner, so the total is 0.
    """
    total = 0.0
    for i in range(4):
        v1 = pts[(i - 1) % 4] - pts[i]
        v2 = pts[(i + 1) % 4] - pts[i]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-9 or n2 < 1e-9:
            return float('inf')
        cos_a = np.dot(v1, v2) / (n1 * n2)
        total += cos_a * cos_a
    return total


def _find_corners_curvature_peaks(cnt, green=False):
    """Find the 4 piece corners as the top positive curvature peaks.

    Piece corners are sharp ~90° left-turns in the contour, producing the
    largest positive peaks in the smoothed angular-velocity signal.  Head and
    hole shoulder peaks are smaller and wash out at higher sigma values.

    At each sigma we evaluate all C(K,4) spacing-valid combinations and score
    each by how closely the 4 points form a rectangle.  The global best-scoring
    combination across all sigma levels is returned, with early exit once the
    rectangle score is below a tight threshold.

    Returns a sorted numpy int array of 4 contour indices, or None on failure.
    """
    n = len(cnt)
    cnt_pts = cnt[:, 0, :].astype(float)
    cnt_convert = [c[0] for c in cnt]
    OFFSET_LOW  = n / 10
    OFFSET_HIGH = n / 2.0
    RECT_THRESHOLD = 0.05   # ~cos²(13°) per corner on average; exit early if beaten

    sigma = 5
    max_sigma = 12 if green else 20
    best_corners = None
    best_score = float('inf')

    while sigma <= max_sigma:
        ra = np.array(get_relative_angles(np.array(cnt_convert), sigma=sigma))
        if np.max(ra) <= 0:
            sigma += 1
            continue

        # Collect positive peaks with double-roll trick
        extr = detect_peaks(ra, mph=0.3 * np.max(ra))
        ra_sh = np.roll(ra, int(n / 2))
        extr = np.unique(np.append(
            extr,
            (detect_peaks(ra_sh, mph=0.3 * float(max(ra_sh))) - int(n / 2)) % n,
        ))

        if len(extr) < 4:
            sigma += 1
            continue

        # Limit to top-K by amplitude; C(K,4) << P(extr,4)
        K = min(len(extr), 8)
        top_k = np.sort(extr[np.argsort(ra[extr])[::-1]][:K])

        for comb in itertools.combinations(top_k, 4):
            c = np.array(comb)
            spacings = np.array([(c[(i + 1) % 4] - c[i]) % n for i in range(4)])
            if not (np.all(spacings > OFFSET_LOW) and np.all(spacings < OFFSET_HIGH)):
                continue
            score = _rectangle_score(cnt_pts[c])
            if score < best_score:
                best_score = score
                best_corners = c

        if best_score < RECT_THRESHOLD:
            break   # already a tight rectangle, no need to keep smoothing

        sigma += 1

    return best_corners


def my_find_corner_signature(cnt, green=False):
    """Determine corner/edge positions by analyzing a piece contour."""
    # --- A: Find corners via curvature peaks (top-K combinations search) ---
    corner_indices = _find_corners_curvature_peaks(cnt, green)
    if corner_indices is None:
        print(f"[corner-detect] curvature peak search FAILED (contour len={len(cnt)})")
        return None, None, None

    n = len(cnt)

    # --- B: Compute offset so last corner sits at index n-1 (matches downstream) ---
    offset = n - int(corner_indices[3]) - 1
    best_fit = corner_indices + offset

    # --- C: Sigma loop for edge-type classification only (corners are fixed) ---
    sigma = 5
    max_sigma = 12 if green else 20
    types_pieces = []
    extr = extr_inverse = relative_angles = None

    while sigma <= max_sigma:
        print(f"Classify edges with sigma={sigma}...")

        cnt_convert = [c[0] for c in cnt]
        _ra = np.array(get_relative_angles(np.array(cnt_convert), sigma=sigma))
        _ra_inv = -_ra.copy()

        # double-roll trick to catch wrap-around peaks
        extr_tmp = detect_peaks(_ra, mph=0.3 * np.max(_ra))
        _ra_sh = np.roll(_ra, int(n / 2))
        extr_tmp = np.unique(np.append(
            extr_tmp,
            (detect_peaks(_ra_sh, mph=0.3 * max(_ra_sh)) - int(n / 2)) % n,
        ))
        extr_tmp_inv = detect_peaks(_ra_inv, mph=0.3 * np.max(_ra_inv))
        _ra_inv_sh = np.roll(_ra_inv, int(n / 2))
        extr_tmp_inv = np.unique(np.append(
            extr_tmp_inv,
            (detect_peaks(_ra_inv_sh, mph=0.3 * max(_ra_inv_sh)) - int(n / 2)) % n,
        ))

        _ra_rolled = np.roll(_ra, offset)
        extr         = (extr_tmp     + offset) % n
        extr_inverse = (extr_tmp_inv + offset) % n
        relative_angles = normalized(_ra_rolled[:, np.newaxis], axis=0).ravel()

        segs = [
            [0,                int(best_fit[0])],
            [int(best_fit[0]), int(best_fit[1])],
            [int(best_fit[1]), int(best_fit[2])],
            [int(best_fit[2]), int(best_fit[3])],
        ]
        tmp_types, no_undefined = [], True
        for seg in segs:
            pos_in = sorted(peaks_inside(seg, extr))
            neg_in = sorted(peaks_inside(seg, extr_inverse))
            t = type_peak(pos_in, neg_in)
            tmp_types.append(t)
            if t == TypeEdge.UNDEFINED:
                no_undefined = False
        types_pieces = tmp_types
        sigma += 1
        if no_undefined:
            break

    if not types_pieces:
        print(f"[corner-detect] FAILED after sigma={sigma-1} (contour len={len(cnt)})")
        return None, None, None

    if types_pieces[-1] == TypeEdge.UNDEFINED:
        print("UNDEFINED FOUND - try to continue but something bad happened :(")

    _RECLASS_SIGMA      = 5
    _RECLASS_MPH        = 0.15
    _BORDER_FRAC        = 0.15
    _CORNER_MARGIN_FRAC = 0.10

    _cnt_pts = [c[0] for c in cnt]
    _la_r  = np.roll(np.array(get_relative_angles(np.array(_cnt_pts), sigma=_RECLASS_SIGMA)), offset)
    _la_ri = -_la_r
    _ep = detect_peaks(_la_r,  mph=_RECLASS_MPH * float(np.max(_la_r))  if np.max(_la_r)  > 0 else 1e-6)
    _en = detect_peaks(_la_ri, mph=_RECLASS_MPH * float(np.max(_la_ri)) if np.max(_la_ri) > 0 else 1e-6)
    _la_r_norm = normalized(_la_r[:, np.newaxis], axis=0).ravel()

    _segs = [
        [0,              int(best_fit[0])],
        [int(best_fit[0]), int(best_fit[1])],
        [int(best_fit[1]), int(best_fit[2])],
        [int(best_fit[2]), int(best_fit[3])],
    ]
    _seg_amps = [
        float(np.max(np.abs(_la_r_norm[s[0]:s[1]]))) if s[1] > s[0]
        else float(np.max(np.abs(_la_r_norm[s[0]:])))
        for s in _segs
    ]
    _flat_thr = _BORDER_FRAC * (max(_seg_amps) if _seg_amps else 1.0)

    _types_low = []
    for _seg in _segs:
        _seg_len = max(_seg[1] - _seg[0], 1)
        _margin  = max(3, int(_CORNER_MARGIN_FRAC * _seg_len))
        _pi = [p for p in sorted(peaks_inside(_seg, _ep)) if (p - _seg[0]) > _margin and (_seg[1] - p) > _margin]
        _ni = [p for p in sorted(peaks_inside(_seg, _en)) if (p - _seg[0]) > _margin and (_seg[1] - p) > _margin]
        _t = type_peak(_pi, _ni)
        if _t == TypeEdge.UNDEFINED:
            _sa = _la_r_norm[_seg[0]:_seg[1]] if _seg[1] > _seg[0] else _la_r_norm[_seg[0]:]
            _t = _reclassify_undefined(_t, _pi, _ni, segment_angles=_sa, flat_threshold=_flat_thr)
        _types_low.append(_t)

    if TypeEdge.UNDEFINED not in _types_low:
        types_pieces = _types_low
    else:
        segs = [[0, best_fit[0]], [best_fit[0], best_fit[1]],
                [best_fit[1], best_fit[2]], [best_fit[2], best_fit[3]]]
        for i, (t, seg) in enumerate(zip(types_pieces, segs)):
            if t == TypeEdge.UNDEFINED:
                pos_in = sorted(peaks_inside(seg, extr))
                neg_in = sorted(peaks_inside(seg, extr_inverse))
                sa = relative_angles[seg[0]:seg[1]] if seg[1] > seg[0] else relative_angles[seg[0]:]
                types_pieces[i] = _reclassify_undefined(t, pos_in, neg_in, segment_angles=sa)

    # corner_indices == best_fit - offset by construction
    best_fit_tmp = corner_indices
    edges = []
    for i in range(3):
        edges.append(cnt[best_fit_tmp[i]:best_fit_tmp[i + 1]])
    edges.append(np.concatenate((cnt[best_fit_tmp[3]:], cnt[:best_fit_tmp[0]]), axis=0))
    edges = [np.array([x[0] for x in e]) for e in edges]

    types_pieces.append(types_pieces[0])
    return best_fit, edges, types_pieces[1:]


def export_contours_without_colormatching(
    img, img_bw, contours, path, modulo, viewer=None, green=False, export_img=False
):
    signatures = _compute_signatures(contours, green)
    _check_piece_dimensions(contours, signatures)

    if config.DEBUG_FILE_OUTPUT == 1:
        canvas = img.copy()
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
        for idx, cnt in enumerate(contours):
            c, _, _ = signatures[idx]
            if c is None:
                continue
            cv2.drawContours(canvas, [cnt], 0, (60, 60, 60), 1)
            pts = np.array([cnt[int(ci)][0] for ci in c])
            cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
            cv2.putText(canvas, f"P{idx}", (cx - 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
            for j, pt in enumerate(pts):
                cv2.circle(canvas, tuple(pt.astype(int)), 8, colors[j % 4], -1)
                cv2.putText(canvas, str(j), (int(pt[0]) + 6, int(pt[1]) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        config.save_debug_img(os.path.join(os.environ["ZOLVER_TEMP_DIR"], "corners_vis.png"), canvas)

    puzzle_pieces = []
    list_img = []

    for idx, cnt in enumerate(contours):
        corners, edges_shape, types_edges = signatures[idx]
        if corners is None or len(edges_shape) != 4:
            continue

        dir_map = assign_directions_from_geometry(edges_shape)
        edges = [
            Edge(edges_shape[i], None, edge_type=types_edges[i], direction=dir_map[i],
                 connected=types_edges[i] == TypeEdge.BORDER)
            for i in range(4)
        ]
        piece = PuzzlePiece(edges)
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            piece.img_centroid = (M['m10'] / M['m00'], M['m01'] / M['m00'])  # (col, row)
        puzzle_pieces.append(piece)

        if export_img:
            mask_border = np.zeros_like(img_bw)
            for i in range(4):
                for p in edges_shape[i]:
                    mask_border[p[1], p[0]] = 255
            out = np.zeros_like(img_bw)
            out[mask_border == 255] = img_bw[mask_border == 255]
            x, y, w, h = cv2.boundingRect(cnt)
            list_img.append(out[y:y + h, x:x + w])

    if export_img and list_img:
        _pack_pieces_grid(list_img, modulo, path)

    return puzzle_pieces
