import itertools
import math
import os
import pickle
from multiprocessing import Pool, cpu_count

import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy
import config

from Puzzle.Edge import Edge
from Puzzle.Enums import directions, TypeEdge
from Puzzle.PuzzlePiece import PuzzlePiece
from .peak_detect import detect_peaks
from Puzzle.Distance import rgb2hsl

"""
Modul: Puzzle/img_processing.py

Aufgabe:
- Analysiert die Konturen einzelner Puzzleteile.
- Bestimmt lokale Winkelverläufe, glättet sie und erkennt Eckpunkte (Corners).
- Klassifiziert Kanten in:
    * BORDER – Rahmenkante
    * HOLE – Einbuchtung
    * HEAD – Ausbuchtung
- Extrahiert Farbinformationen entlang der Kanten zur späteren Kanten-Matching-Phase.
- Baut daraus PuzzlePiece-Objekte für den Solver.

Wichtige externe Abhängigkeiten:
- OpenCV (cv2) für Konturerkennung und Bildmasken
- NumPy / SciPy für Signal- und Matrixoperationen
- Matplotlib & Pickle für Debug- und Exportzwecke
"""


COUNT = 0


def get_relative_angles(cnt, export, sigma=5):
    """
    Berechnet die relativen Winkeländerungen entlang einer Kontur.

    - cnt: Liste der (x,y)-Punkte der Kontur
    - sigma: Glättungsparameter des Gaußfilters (höher = stärker geglättet)
    - export: Wenn True → speichert die Signatur und Diagramm zur Kontrolle

    Rückgabe:
    - Array von Winkeländerungen (relative Winkelkurve)
    """

    global COUNT
    COUNT = COUNT + 1

    length = len(cnt)
    angles = []
    last = np.pi

    cnt_tmp = np.array(cnt)
    cnt = np.append(cnt, cnt_tmp, axis=0)
    cnt = np.append(cnt, cnt_tmp, axis=0)
    for i in range(0, len(cnt) - 1):
        dir = (cnt[i + 1][0] - cnt[i][0], cnt[i + 1][1] - cnt[i][1])
        angle = math.atan2(-dir[1], dir[0])
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
    angles = angles[0:length]

    if export:
        pickle.dump(angles, open(os.path.join(os.environ["ZOLVER_TEMP_DIR"], "save" + str(COUNT) + ".p"), "wb"))
        plt.plot(np.append(angles, angles))
        plt.savefig(os.path.join(os.environ["ZOLVER_TEMP_DIR"], "fig" + str(COUNT) + ".png"))
        plt.clf()
        plt.cla()
        plt.close()

    return angles


def is_maximum_local(index, relative_angles, radius):
    """
    Determine if a point at index is a maximum local in radius range of relative_angles function

    :param index: index of the point to check in relative_angles list
    :param relative_angles: list of angles
    :param radius: radius used to check neighbors
    :return: Boolean
    """

    start = max(0, index - radius)
    end = min(relative_angles.shape[0] - 1, index + radius)
    for i in range(start, end + 1):
        if relative_angles[i] > relative_angles[index]:
            return False
    return True


def longest_peak(relative_angles):
    """
    Find the longest area < 0

    :param relative_angles: list of angles
    :return: coordinates of the area
    """

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
    """
    Distance of each points to the line formed by first and last points

    :param relative_angles: list of angles
    :return: List of floats
    """
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
    """
    Compute the flat score of relative_angles

    :param relative_angles: list of angles
    :return: List of floats
    """

    length = relative_angles.shape[0]
    distances = distance_signature(relative_angles)
    diff = 0
    for i in range(length):
        diff = max(diff, abs(distances[i]))
    return diff


def indent_score(relative_angles):
    """
    Compute score for indent part

    :param relative_angles: list of angles
    :return: List of floats
    """

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

    # FIX FOR FUNCTIONS > 0
    if shape.shape[0] == 1:
        return flat_score(relative_angles)
    return flat_score(shape)


def outdent_score(relative_angles):
    """
    Compute score for outdent part

    :param relative_angles: list of angles
    :return: List of floats
    """
    return indent_score(-relative_angles)


def compute_comp(combs_l, relative_angles, method="correlate"):
    """
    Compute score for each combination of 4 points and return the index of the best

    :param combs_l: list of combinations of 4 points
    :param relative_angles: List of angles
    :return: Int
    """

    results_glob = []
    for comb_t in combs_l:
        # Roll the values of relative angles for this combination
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
                hole = indent_score(
                    np.ravel(np.array(relative_angles_tmp[comb[0] : comb[1]]))
                )[0]
                head = outdent_score(
                    np.ravel(np.array(relative_angles_tmp[comb[0] : comb[1]]))
                )[0]
                border = flat_score(
                    np.ravel(np.array(relative_angles_tmp[comb[0] : comb[1]]))
                )[0]
            if hole != border:
                results_comp.append(min(hole, head))
            else:
                results_comp.append(border)
        results_glob.append(np.sum(np.array(results_comp)))
    return np.argmin(np.array(results_glob))


def peaks_inside(comb, peaks):
    """
    Check the number of peaks inside comb

    :param comb: Tuple of coordinates
    :param peaks: List of peaks to check
    :return: Int
    """
    if len(comb) == 0:
        return []
    return [peak for peak in peaks if peak > comb[0] and peak < comb[-1]]


def is_pattern(comb, peaks):
    """
    Check if the peaks formed an outdent or an indent pattern

    :param comb: Tuple of coordinates
    :param peaks: List of peaks
    :return: Int
    """
    cpt = len(peaks_inside(comb, peaks))
    return cpt == 0 or cpt == 2 or cpt == 3


def is_acceptable_comb(combs, peaks, length):
    """
    Check if a combination is composed of acceptable patterns.
    Used to filter the obviously bad combinations quickly.

    :param comb: Tuple of coordinates
    :param peaks: List of peaks
    :param length: Length of the signature (used for offset computation)
    :return: Boolean
    """

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
    """
    Determine the type of lists of pos and neg peaks

    :param peaks_pos_inside: List of positive peaks
    :param peaks_neg_inside: List of negative peaks
    :return: TypeEdge
    """

    if len(peaks_pos_inside) == 0 and len(peaks_neg_inside) == 0:
        return TypeEdge.BORDER
    if len(peaks_inside(peaks_pos_inside, peaks_neg_inside)) == 2:
        return TypeEdge.HOLE
    if len(peaks_inside(peaks_neg_inside, peaks_pos_inside)) == 2:
        return TypeEdge.HEAD
    return TypeEdge.UNDEFINED


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, ord=order, axis=axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)




def _unwrap_contour_points(cnt):
    pts = np.asarray(cnt)
    if pts.ndim == 3:
        pts = pts[:, 0, :]
    return np.asarray(pts, dtype=np.float32)


def _normalize_vec(v):
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n <= 1e-8:
        return np.zeros_like(v)
    return v / n


def _circular_index_distance(a, b, n):
    d = abs(int(a) - int(b))
    return min(d, n - d)


def _ordered_box_corners(box):
    box = np.asarray(box, dtype=np.float32)
    center = np.mean(box, axis=0)
    ang = np.arctan2(box[:, 1] - center[1], box[:, 0] - center[0])
    order = np.argsort(ang)
    return box[order]


def _refine_corner_around_seed(pts, seed_idx, rect_corner, axis_a, axis_b, search_radius, step):
    n = len(pts)
    best_idx = int(seed_idx) % n
    best_score = -1e18
    best_dist = float('inf')

    for delta in range(-search_radius, search_radius + 1):
        idx = (int(seed_idx) + delta) % n
        p_prev = pts[(idx - step) % n]
        p = pts[idx]
        p_next = pts[(idx + step) % n]

        vin = _normalize_vec(p - p_prev)
        vout = _normalize_vec(p_next - p)
        if not np.any(vin) or not np.any(vout):
            continue

        # Real puzzle corners are approximately 90° corners between the two
        # dominant rectangle axes. The direction on each side can flip, hence abs().
        assign1 = abs(float(np.dot(vin, axis_a))) + abs(float(np.dot(vout, axis_b)))
        assign2 = abs(float(np.dot(vin, axis_b))) + abs(float(np.dot(vout, axis_a)))
        axis_score = max(assign1, assign2) / 2.0

        right_angle_score = 1.0 - abs(float(np.dot(vin, vout)))
        dist = float(np.linalg.norm(p - rect_corner))
        dist_score = 1.0 / (1.0 + dist)

        # Favor actual turning points over straight sections near the rectangle corner.
        score = 3.0 * axis_score + 2.0 * right_angle_score + 0.35 * dist_score
        if score > best_score or (abs(score - best_score) < 1e-9 and dist < best_dist):
            best_score = score
            best_idx = idx
            best_dist = dist

    return int(best_idx), float(best_score)


def refine_corner_indices(cnt, initial_indices=None):
    pts = _unwrap_contour_points(cnt)
    n = len(pts)
    if n < 8:
        return None

    rect = cv2.minAreaRect(pts)
    box = _ordered_box_corners(cv2.boxPoints(rect))

    if len(box) != 4:
        return None

    # Rectangle side directions give the two dominant orthogonal corner directions.
    axes = []
    for i in range(4):
        side = box[(i + 1) % 4] - box[i]
        unit = _normalize_vec(side)
        if np.linalg.norm(unit) > 0:
            axes.append(unit)
    if len(axes) < 2:
        return None

    axis_a = axes[0]
    axis_b = None
    for cand in axes[1:]:
        if abs(float(np.dot(cand, axis_a))) < 0.6:
            axis_b = cand
            break
    if axis_b is None:
        axis_b = _normalize_vec(np.array([-axis_a[1], axis_a[0]], dtype=np.float32))

    nearest_rect = [int(np.argmin(np.sum((pts - corner) ** 2, axis=1))) for corner in box]

    if initial_indices is None or len(initial_indices) != 4:
        seeds = nearest_rect
    else:
        initial_indices = [int(i) % n for i in initial_indices]
        available = set(initial_indices)
        seeds = []
        for base_idx in nearest_rect:
            if available:
                best = min(available, key=lambda i: _circular_index_distance(i, base_idx, n))
                seeds.append(best)
                available.remove(best)
            else:
                seeds.append(base_idx)

    search_radius = max(8, min(n // 12, 80))
    step = max(2, min(n // 80, 8))

    refined = []
    scores = []
    for j in range(4):
        corner = box[j]
        prev_side = _normalize_vec(box[(j - 1) % 4] - corner)
        next_side = _normalize_vec(box[(j + 1) % 4] - corner)
        idx, score = _refine_corner_around_seed(
            pts,
            seeds[j],
            corner,
            prev_side,
            next_side,
            search_radius,
            step,
        )
        refined.append(idx)
        scores.append(score)

    # Enforce 4 distinct corners with reasonable spacing.
    min_sep = max(4, n // 10)
    ranked = sorted(range(4), key=lambda i: scores[i], reverse=True)
    chosen = []
    for ridx in ranked:
        idx = refined[ridx]
        if all(_circular_index_distance(idx, other, n) >= min_sep for other in chosen):
            chosen.append(idx)

    if len(chosen) < 4:
        candidates = sorted(set(refined + nearest_rect), key=lambda i: min(_circular_index_distance(i, s, n) for s in refined))
        for idx in candidates:
            if all(_circular_index_distance(idx, other, n) >= min_sep for other in chosen):
                chosen.append(idx)
            if len(chosen) == 4:
                break

    if len(chosen) < 4:
        return None

    return np.sort(np.asarray(chosen[:4], dtype=int))


def contour_signed_area(points):
    pts = np.asarray(points, dtype=float)
    if pts.ndim == 3:
        pts = pts[:, 0, :]
    if len(pts) < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def edge_line_metrics(shape):
    pts = np.asarray(shape, dtype=float)
    if pts.ndim == 3:
        pts = pts[:, 0, :]
    if len(pts) < 2:
        return {"chord": 0.0, "max_dev": 0.0, "norm_dev": 0.0, "signed_mean": 0.0}

    start = pts[0]
    end = pts[-1]
    chord_vec = end - start
    chord = float(np.linalg.norm(chord_vec))
    if chord < 1e-6:
        return {"chord": 0.0, "max_dev": 0.0, "norm_dev": 0.0, "signed_mean": 0.0}

    rel = pts - start
    signed = (chord_vec[0] * rel[:, 1] - chord_vec[1] * rel[:, 0]) / chord
    max_dev = float(np.max(np.abs(signed))) if len(signed) else 0.0
    usable = signed[1:-1] if len(signed) > 2 else signed
    signed_mean = float(np.mean(usable)) if len(usable) else 0.0
    return {
        "chord": chord,
        "max_dev": max_dev,
        "norm_dev": max_dev / max(chord, 1.0),
        "signed_mean": signed_mean,
    }


def classify_edge_from_shape(shape, contour_orientation, border_threshold=0.10):
    pts = np.asarray(shape, dtype=float)
    if pts.ndim == 3:
        pts = pts[:, 0, :]
    if len(pts) < 3:
        return TypeEdge.UNDEFINED

    metrics = edge_line_metrics(pts)
    if metrics["chord"] <= 1.0:
        return TypeEdge.UNDEFINED
    if metrics["norm_dev"] <= border_threshold:
        return TypeEdge.BORDER

    start = pts[0]
    end = pts[-1]
    chord_vec = end - start
    chord = float(np.linalg.norm(chord_vec))
    if chord <= 1e-6:
        return TypeEdge.UNDEFINED

    rel = pts - start
    signed = (chord_vec[0] * rel[:, 1] - chord_vec[1] * rel[:, 0]) / chord
    usable = signed[1:-1] if len(signed) > 2 else signed
    if len(usable) == 0:
        return TypeEdge.UNDEFINED

    # Ignore tiny noise around the baseline.
    amp = float(np.max(np.abs(usable)))
    if amp <= max(1.5, 0.08 * chord):
        return TypeEdge.BORDER

    active = usable[np.abs(usable) >= 0.25 * amp]
    if len(active) == 0:
        return TypeEdge.UNDEFINED

    pos = int(np.sum(active > 0))
    neg = int(np.sum(active < 0))
    dominant_ratio = max(pos, neg) / max(pos + neg, 1)

    # Mixed edges often come from bad corner splits; do not force a sign.
    if dominant_ratio < 0.72:
        return TypeEdge.UNDEFINED

    signed_mean = float(np.median(active))
    interior_sign = 1.0 if contour_orientation > 0 else -1.0
    if signed_mean * interior_sign > 0:
        return TypeEdge.HOLE
    return TypeEdge.HEAD


def split_contour_by_corner_indices(cnt, corner_indices):
    pts = np.asarray(cnt)
    n = len(pts)
    if n == 0 or len(corner_indices) != 4:
        return None

    corners = np.sort(np.asarray(corner_indices, dtype=int) % n)
    edges = []
    for i in range(3):
        seg = pts[corners[i]:corners[i + 1]]
        if len(seg) < 2:
            return None
        edges.append(seg)
    seg = np.concatenate((pts[corners[3]:], pts[:corners[0] + 1]), axis=0)
    if len(seg) < 2:
        return None
    edges.append(seg)
    return [np.array([x[0] for x in e]) if np.asarray(e).ndim == 3 else np.asarray(e) for e in edges]


def fallback_find_corner_signature(cnt):
    pts2 = _unwrap_contour_points(cnt)
    if len(pts2) < 8:
        return None, None, None

    corner_indices = refine_corner_indices(cnt)
    if corner_indices is None or len(set(corner_indices)) < 4:
        return None, None, None

    edges = split_contour_by_corner_indices(cnt, corner_indices)
    if edges is None or len(edges) != 4:
        return None, None, None

    orientation = contour_signed_area(pts2)
    types = [classify_edge_from_shape(e, orientation) for e in edges]
    types = refine_edge_types(pts2, edges, types)
    return np.sort(np.asarray(corner_indices, dtype=int)), edges, types


def refine_edge_types(cnt, edges, initial_types=None):
    """Refine noisy edge labels conservatively.

    On hard pieces the split may cut slightly before/after the real rectangular
    corner. In that case an edge can contain both an in- and an outdent signal.
    Those edges should stay UNDEFINED instead of being forced into HOLE/HEAD.
    """
    orientation = contour_signed_area(cnt)
    metrics = [edge_line_metrics(edge) for edge in edges]

    refined = []
    for i, edge in enumerate(edges):
        geom_type = classify_edge_from_shape(edge, orientation)
        original = None if initial_types is None or i >= len(initial_types) else initial_types[i]

        if geom_type == TypeEdge.BORDER:
            refined.append(TypeEdge.BORDER)
        elif geom_type in (TypeEdge.HOLE, TypeEdge.HEAD):
            refined.append(geom_type)
        elif original in (TypeEdge.HOLE, TypeEdge.HEAD) and metrics[i]["norm_dev"] > 0.12:
            refined.append(original)
        else:
            refined.append(TypeEdge.UNDEFINED)

    border_idx = [i for i, t in enumerate(refined) if t == TypeEdge.BORDER]
    if len(border_idx) > 2:
        border_idx = sorted(border_idx, key=lambda i: metrics[i]["norm_dev"])
        keep = set(border_idx[:2])
        for i in range(len(refined)):
            if refined[i] == TypeEdge.BORDER and i not in keep:
                refined[i] = TypeEdge.UNDEFINED

    if len(border_idx) == 0 and len(metrics) == 4:
        ordered = sorted(range(4), key=lambda i: metrics[i]["norm_dev"])
        if metrics[ordered[0]]["norm_dev"] < 0.60 * metrics[ordered[1]]["norm_dev"]:
            refined[ordered[0]] = TypeEdge.BORDER

    return refined

def my_find_corner_signature(cnt, green=False):
    """
    Determine the corner/edge positions by analyzing contours.

    :param cnt: contour to analyze
    :param green: boolean used to activate green background mode
    :type cnt: list of tuple of points
    :return: Corners coordinates, Edges lists of points, type of pieces
    """

    edges = []
    types_pieces = []

    sigma = 5
    max_sigma = 12
    if not green:
        sigma = 5
        max_sigma = 15

    best_fit = None
    best_offset = None

    while sigma <= max_sigma:
        print("Smooth curve with sigma={}...".format(sigma))

        tmp_combs_final = []

        # Find relative angles
        cnt_convert = [c[0] for c in cnt]
        relative_angles = get_relative_angles(
            np.array(cnt_convert),
            export=(config.DEBUG_MODE == 1),
            sigma=sigma
        )
        relative_angles = np.array(relative_angles)
        relative_angles_inverse = -np.array(relative_angles)

        # Positive peaks
        max_rel = np.max(relative_angles) if len(relative_angles) > 0 else 0
        extr_tmp = detect_peaks(relative_angles, mph=0.3 * max_rel) if max_rel > 0 else np.array([], dtype=int)

        relative_angles = np.roll(relative_angles, int(len(relative_angles) / 2))
        max_rel_rolled = max(relative_angles) if len(relative_angles) > 0 else 0
        extra_peaks = (
            detect_peaks(relative_angles, mph=0.3 * max_rel_rolled)
            if max_rel_rolled > 0
            else np.array([], dtype=int)
        )
        extr_tmp = np.append(
            extr_tmp,
            (extra_peaks - int(len(relative_angles) / 2)) % len(relative_angles),
            axis=0,
            )
        relative_angles = np.roll(relative_angles, -int(len(relative_angles) / 2))
        extr_tmp = np.unique(extr_tmp)

        # Negative peaks
        max_rel_inv = np.max(relative_angles_inverse) if len(relative_angles_inverse) > 0 else 0
        extr_tmp_inverse = (
            detect_peaks(relative_angles_inverse, mph=0.3 * max_rel_inv)
            if max_rel_inv > 0
            else np.array([], dtype=int)
        )

        relative_angles_inverse = np.roll(
            relative_angles_inverse, int(len(relative_angles_inverse) / 2)
        )
        max_rel_inv_rolled = max(relative_angles_inverse) if len(relative_angles_inverse) > 0 else 0
        extra_peaks_inv = (
            detect_peaks(relative_angles_inverse, mph=0.3 * max_rel_inv_rolled)
            if max_rel_inv_rolled > 0
            else np.array([], dtype=int)
        )
        extr_tmp_inverse = np.append(
            extr_tmp_inverse,
            (
                    extra_peaks_inv - int(len(relative_angles_inverse) / 2)
            ) % len(relative_angles_inverse),
            axis=0,
            )
        relative_angles_inverse = np.roll(
            relative_angles_inverse, -int(len(relative_angles_inverse) / 2)
        )
        extr_tmp_inverse = np.unique(extr_tmp_inverse)

        extr = extr_tmp
        extr_inverse = extr_tmp_inverse

        if len(relative_angles) == 0 or len(extr) < 4:
            sigma += 1
            continue

        relative_angles = normalized(relative_angles[:, np.newaxis], axis=0).ravel()

        # Build list of permutations of 4 points
        combs = itertools.permutations(extr, 4)
        combs_l = list(combs)
        OFFSET_LOW = len(relative_angles) / 8
        OFFSET_HIGH = len(relative_angles) / 2.0

        for comb in combs_l:
            if (
                    (comb[0] > comb[1])
                    and (comb[1] > comb[2])
                    and (comb[2] > comb[3])
                    and ((comb[0] - comb[1]) > OFFSET_LOW)
                    and ((comb[0] - comb[1]) < OFFSET_HIGH)
                    and ((comb[1] - comb[2]) > OFFSET_LOW)
                    and ((comb[1] - comb[2]) < OFFSET_HIGH)
                    and ((comb[2] - comb[3]) > OFFSET_LOW)
                    and ((comb[2] - comb[3]) < OFFSET_HIGH)
                    and ((comb[3] + (len(relative_angles) - comb[0])) > OFFSET_LOW)
                    and ((comb[3] + (len(relative_angles) - comb[0])) < OFFSET_HIGH)
            ):
                candidate = (comb[3], comb[2], comb[1], comb[0])
                if is_acceptable_comb(candidate, extr, len(relative_angles)) and is_acceptable_comb(
                        candidate, extr_inverse, len(relative_angles)
                ):
                    tmp_combs_final.append(candidate)

        if len(tmp_combs_final) == 0:
            sigma += 1
            continue

        # Best corner fit for this sigma
        best_fit = np.array(
            tmp_combs_final[
                compute_comp(tmp_combs_final, relative_angles, method="flat")
            ],
            dtype=int,
        )

        # Roll the values of relative angles for this combination
        best_offset = len(relative_angles) - best_fit[3] - 1
        relative_angles = np.roll(relative_angles, best_offset)
        best_fit = best_fit + best_offset
        extr = (extr + best_offset) % len(relative_angles)
        extr_inverse = (extr_inverse + best_offset) % len(relative_angles)

        tmp_types_pieces = []
        no_undefined = True

        for best_comb in [
            [0, best_fit[0]],
            [best_fit[0], best_fit[1]],
            [best_fit[1], best_fit[2]],
            [best_fit[2], best_fit[3]],
        ]:
            pos_peaks_inside = peaks_inside(best_comb, extr)
            neg_peaks_inside = peaks_inside(best_comb, extr_inverse)
            pos_peaks_inside.sort()
            neg_peaks_inside.sort()
            tmp_types_pieces.append(type_peak(pos_peaks_inside, neg_peaks_inside))
            if tmp_types_pieces[-1] == TypeEdge.UNDEFINED:
                no_undefined = False

        types_pieces = tmp_types_pieces

        if no_undefined:
            break

        sigma += 1

    # No valid fit found at all -> try geometric fallback based on minAreaRect
    if best_fit is None or best_offset is None or len(types_pieces) == 0:
        return fallback_find_corner_signature(cnt)

    # Back to original contour indexing and refine the 4 corners geometrically.
    best_fit_tmp = np.sort((best_fit - best_offset).astype(int))
    refined_corner_idx = refine_corner_indices(cnt, best_fit_tmp)
    if refined_corner_idx is not None and len(refined_corner_idx) == 4:
        best_fit_tmp = refined_corner_idx

    edges = split_contour_by_corner_indices(cnt, best_fit_tmp)
    if edges is None or len(edges) != 4:
        return fallback_find_corner_signature(cnt)

    refined_types = refine_edge_types(np.array([c[0] for c in cnt]), edges, types_pieces)
    return best_fit_tmp, edges, refined_types


def export_contours(
    img, img_bw, contours, path, modulo, viewer=None, green=False, export_img=False
):
    """
    Baut aus allen Konturen die einzelnen Puzzle-Teile auf.

    Schritte:
    1. Für jede Kontur → finde Kanten und Typen (via my_find_corner_signature).
    2. Erstelle Masken und schneide das Puzzleteil aus dem Originalbild.
    3. Berechne Median-Farben entlang jeder Kante.
    4. Erzeuge PuzzlePiece-Objekte mit geometrischen und Farbdaten.
    5. (Optional) Exportiere Debug-Bilder und zeige sie im Viewer.

    Parameter:
    - img: Originalbild (RGB)
    - img_bw: Binärbild (Schwarz/Weiss)
    - contours: erkannte Konturen
    - path: Speicherort für exportierte Bilder
    - modulo: Spaltenanzahl bei exportierten Bildrastern
    - viewer: GUI-Objekt (zum Anzeigen in Zolver)
    - green: Aktiviert Grün-Screen-Modus (andere Filterparameter)
    - export_img: True → speichert Debug-Bilder
    - return: puzzle array
    """

    puzzle_pieces = []
    list_img = []
    out_color = np.zeros_like(img)

    if config.DEBUG_MODE == 1:
        signatures = [
            my_find_corner_signature(cnt, green)
            for cnt in contours
        ]
    else:
        with Pool(cpu_count()) as p:
            signatures = p.starmap(
                my_find_corner_signature, zip(contours, itertools.repeat(green))
            )

    for idx, cnt in enumerate(contours):
        corners, edges_shape, types_edges = signatures[idx]
        if corners is None:
            return None

        mask_border = np.zeros_like(img_bw)
        mask_full = np.zeros_like(img_bw)
        mask_full = cv2.drawContours(mask_full, contours, idx, 255, -1)
        mask_border = cv2.drawContours(mask_border, contours, idx, 255, 1)

        img_piece = np.zeros_like(img)
        img_piece[mask_full == 255] = img[mask_full == 255]

        pixels = {
            (x, y): img_piece[x, y] for x, y in tuple(zip(*np.where(mask_full == 255)))
        }

        # go faster, use only a subset of the img with the piece
        x_bound, y_bound, w_bound, h_bound = cv2.boundingRect(cnt)
        img_piece_tiny = img_piece[
            y_bound : y_bound + h_bound, x_bound : x_bound + w_bound
        ]
        mask_border_tiny = mask_border[
            y_bound : y_bound + h_bound, x_bound : x_bound + w_bound
        ]
        mask_full_tiny = mask_full[
            y_bound : y_bound + h_bound, x_bound : x_bound + w_bound
        ]

        mask_around_tiny = np.zeros_like(mask_full_tiny)
        mask_inv_border_tiny = cv2.bitwise_not(mask_border_tiny)
        mask_full_tiny = cv2.bitwise_and(
            mask_full_tiny, mask_full_tiny, mask=mask_inv_border_tiny
        )

        # Find mean color around edge
        color_vect = []
        for i in range(4):
            color_edge = []
            shape = edges_shape[i]
            for ip, p in enumerate(shape):
                if ip != 0:
                    p2 = shape[ip - 1]
                    cv2.circle(
                        mask_around_tiny,
                        (p2[0] - x_bound, p2[1] - y_bound),
                        5,
                        0,
                        -1,
                    )
                cv2.circle(
                    mask_around_tiny,
                    (p[0] - x_bound, p[1] - y_bound),
                    5,
                    255,
                    -1,
                )

                mask_around_tiny = cv2.bitwise_and(
                    mask_around_tiny, mask_around_tiny, mask=mask_full_tiny
                )

                neighbors_color = [
                    img_piece_tiny[y, x]
                    for y, x in tuple(zip(*np.where(mask_around_tiny == 255)))
                ]
                rgb = np.median(neighbors_color, axis=0)
                color_edge.append(rgb2hsl(*rgb))
                out_color[p[1], p[0]] = rgb

            color_vect.append(np.array(color_edge))

        edges = [
            Edge(
                s,
                c,
                edge_type=types_edges[i],
                direction=directions[i],
                connected=types_edges[i] == TypeEdge.BORDER,
            )
            for i, (s, c) in enumerate(zip(edges_shape, color_vect))
        ]
        puzzle_pieces.append(PuzzlePiece(edges, pixels))

        if export_img:
            mask_border = np.zeros_like(img_bw)
            for i in range(4):
                for p in edges_shape[i]:
                    mask_border[p[1], p[0]] = 255

            out = np.zeros_like(img_bw)
            out[mask_border == 255] = img_bw[mask_border == 255]

            x, y, w, h = cv2.boundingRect(cnt)
            out2 = out[y : y + h, x : x + w]

            list_img.append(out2)

    if export_img:
        max_height = max([x.shape[0] for x in list_img])
        max_width = max([x.shape[1] for x in list_img])
        pieces_img = np.zeros(
            [max_height * (int(len(list_img) / modulo) + 1), max_width * modulo],
            dtype=np.uint8,
        )
        for index, image in enumerate(list_img):
            pieces_img[
                (max_height * int(index / modulo)) : (
                    max_height * int(index / modulo) + image.shape[0]
                ),
                (max_width * (index % modulo)) : (
                    max_width * (index % modulo) + image.shape[1]
                ),
            ] = image
        cv2.imwrite(path, pieces_img)
    if viewer:
        cv2.imwrite(os.path.join(os.environ["ZOLVER_TEMP_DIR"], "color_border.png"), out_color)
        viewer.addImage("Extracted colored border", os.path.join(os.environ["ZOLVER_TEMP_DIR"], "color_border.png"))

    return puzzle_pieces

def export_contours_without_colormatching(
        img, img_bw, contours, path, modulo, viewer=None, green=False, export_img=False
):
    puzzle_pieces = []
    list_img = []

    from multiprocessing import Pool, cpu_count
    import itertools

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
            angle = np.arctan2(dy, dx)
            edge_infos.append({
                "index": i,
                "midpoint": midpoint,
                "dx": dx,
                "dy": dy,
                "angle": angle,
            })

        # sort cyclically around the center
        edge_infos.sort(key=lambda e: e["angle"])

        # find the edge whose midpoint is highest (smallest y relative to center)
        up_pos = min(range(4), key=lambda k: edge_infos[k]["dy"])

        # rotate sorted list so it starts with UP
        ordered = edge_infos[up_pos:] + edge_infos[:up_pos]

        # Determine whether cyclic order is clockwise or counterclockwise
        # In image coords, y grows downward
        mids = [e["midpoint"] for e in ordered]
        area2 = 0.0
        for a, b in zip(mids, mids[1:] + mids[:1]):
            area2 += a[0] * b[1] - b[0] * a[1]

        if area2 > 0:
            # counterclockwise -> UP, LEFT, DOWN, RIGHT
            assigned_dirs = [directions[0], directions[3], directions[2], directions[1]]
        else:
            # clockwise -> UP, RIGHT, DOWN, LEFT
            assigned_dirs = [directions[0], directions[1], directions[2], directions[3]]

        dir_map = {
            ordered[k]["index"]: assigned_dirs[k]
            for k in range(4)
        }

        return dir_map

    if config.DEBUG_MODE == 1:
        signatures = [
            my_find_corner_signature(cnt, green)
            for cnt in contours
        ]
    else:
        with Pool(cpu_count()) as p:
            signatures = p.starmap(
                my_find_corner_signature, zip(contours, itertools.repeat(green))
            )

    for idx, cnt in enumerate(contours):
        corners, edges_shape, types_edges = signatures[idx]
        if corners is None or edges_shape is None or len(edges_shape) != 4:
            print(f"Skipping contour {idx}: could not derive 4 edges reliably")
            continue

        types_edges = refine_edge_types(np.array([c[0] for c in cnt]), edges_shape, types_edges)

        mask_border = np.zeros_like(img_bw)
        mask_full = np.zeros_like(img_bw)
        mask_full = cv2.drawContours(mask_full, contours, idx, 255, -1)
        mask_border = cv2.drawContours(mask_border, contours, idx, 255, 1)

        img_piece = np.zeros_like(img)
        img_piece[mask_full == 255] = img[mask_full == 255]

        xs, ys = np.where(mask_full == 255)
        pixels = {(x, y): img_piece[x, y] for x, y in zip(xs, ys)}

        dir_map = assign_directions_from_geometry(edges_shape)

        edges = [
            Edge(
                edges_shape[i],
                None,
                edge_type=types_edges[i],
                direction=dir_map[i],
                connected=types_edges[i] == TypeEdge.BORDER,
            )
            for i in range(4)
        ]

        puzzle_pieces.append(PuzzlePiece(edges, pixels))

        if export_img:
            mask_border = np.zeros_like(img_bw)
            for i in range(4):
                for p in edges_shape[i]:
                    mask_border[p[1], p[0]] = 255

            out = np.zeros_like(img_bw)
            out[mask_border == 255] = img_bw[mask_border == 255]

            x, y, w, h = cv2.boundingRect(cnt)
            out2 = out[y:y+h, x:x+w]
            list_img.append(out2)

    if export_img and list_img:
        max_height = max(x.shape[0] for x in list_img)
        max_width = max(x.shape[1] for x in list_img)
        pieces_img = np.zeros(
            [max_height * (int(len(list_img) / modulo) + 1), max_width * modulo],
            dtype=np.uint8,
        )
        for index, image in enumerate(list_img):
            pieces_img[
                (max_height * int(index / modulo)):(max_height * int(index / modulo) + image.shape[0]),
                (max_width * (index % modulo)):(max_width * (index % modulo) + image.shape[1]),
            ] = image
        cv2.imwrite(path, pieces_img)

    return puzzle_pieces