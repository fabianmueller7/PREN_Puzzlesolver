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


def is_pattern(comb, peaks, strict=True):
    """
    Check if the peaks formed an outdent or an indent pattern

    :param comb: Tuple of coordinates
    :param peaks: List of peaks
    :param strict: If True, only 0/2/3 peaks allowed; if False, 0/1/2/3 peaks allowed
    :return: Int
    """
    cpt = len(peaks_inside(comb, peaks))
    if strict:
        return cpt == 0 or cpt == 2 or cpt == 3
    return cpt == 0 or cpt == 1 or cpt == 2 or cpt == 3


def is_acceptable_comb(combs, peaks, length, strict=True):
    """
    Check if a combination is composed of acceptable patterns.
    Used to filter the obviously bad combinations quickly.

    :param comb: Tuple of coordinates
    :param peaks: List of peaks
    :param length: Length of the signature (used for offset computation)
    :param strict: Passed through to is_pattern; False allows count=1 segments
    :return: Boolean
    """

    offset = length - combs[3] - 1
    combs_tmp = combs + offset
    peaks_tmp = (peaks + offset) % length
    return (
        is_pattern([0, combs_tmp[0]], peaks_tmp, strict)
        and is_pattern([combs_tmp[0], combs_tmp[1]], peaks_tmp, strict)
        and is_pattern([combs_tmp[1], combs_tmp[2]], peaks_tmp, strict)
        and is_pattern([combs_tmp[2], combs_tmp[3]], peaks_tmp, strict)
    )


def type_peak(peaks_pos_inside, peaks_neg_inside):
    """
    Determine the type of lists of pos and neg peaks

    :param peaks_pos_inside: List of positive peaks
    :param peaks_neg_inside: List of negative peaks
    :return: TypeEdge
    """

    n_pos = len(peaks_pos_inside)
    n_neg = len(peaks_neg_inside)

    if n_pos == 0 and n_neg == 0:
        return TypeEdge.BORDER

    # Strict pattern: exactly 2 of one type sandwiched inside the other.
    n_neg_inside_pos = len(peaks_inside(peaks_pos_inside, peaks_neg_inside))
    n_pos_inside_neg = len(peaks_inside(peaks_neg_inside, peaks_pos_inside))

    if n_neg_inside_pos == 2:
        return TypeEdge.HOLE
    if n_pos_inside_neg == 2:
        return TypeEdge.HEAD

    # Lenient fallbacks for complex (e.g. circular/smooth) features where
    # strict peak counting fails:

    # Only one polarity present → clear direction.
    if n_pos > 0 and n_neg == 0:
        return TypeEdge.HEAD
    if n_neg > 0 and n_pos == 0:
        return TypeEdge.HOLE

    # More than 4 peaks total → likely a circular feature with many small peaks.
    # Use majority vote on inner counts.
    if n_pos + n_neg > 4:
        if n_neg_inside_pos > n_pos_inside_neg:
            return TypeEdge.HOLE
        if n_pos_inside_neg > n_neg_inside_pos:
            return TypeEdge.HEAD
        # Tie: more outer-positive than outer-negative → outward bump → HEAD
        return TypeEdge.HEAD if n_pos >= n_neg else TypeEdge.HOLE

    return TypeEdge.UNDEFINED


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, ord=order, axis=axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def _pick_topN_spaced(scores, min_gap, N=8):
    """
    Greedy selection of up to N indices from scores with minimum spacing min_gap.
    Returns indices sorted in ascending order.
    """
    result = []
    blocked = np.zeros(len(scores), dtype=bool)
    for idx in np.argsort(scores)[::-1]:
        if not blocked[idx]:
            result.append(int(idx))
            lo = max(0, idx - min_gap)
            hi = min(len(scores), idx + min_gap)
            blocked[lo:hi] = True
        if len(result) == N:
            break
    return sorted(result)


def _rect_fitness(pts):
    """
    Measure how well 4 points (in order) form a rectangle.
    Returns a non-negative score where 0 = perfect rectangle.
    Two components:
      - side_penalty: opposite sides should be equal length
      - angle_penalty: each interior angle should be 90 degrees
    """
    def _dist(a, b):
        return np.linalg.norm(a.astype(float) - b.astype(float))

    def _angle_at(prev, cur, nxt):
        v1 = prev.astype(float) - cur.astype(float)
        v2 = nxt.astype(float) - cur.astype(float)
        cos_val = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        return np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0)))

    sides = [_dist(pts[i], pts[(i + 1) % 4]) for i in range(4)]
    side_pen = (
        abs(sides[0] - sides[2]) / (sides[0] + sides[2] + 1e-9)
        + abs(sides[1] - sides[3]) / (sides[1] + sides[3] + 1e-9)
    )
    angle_pen = (
        sum(abs(_angle_at(pts[(i - 1) % 4], pts[i], pts[(i + 1) % 4]) - 90.0)
            for i in range(4))
        / 90.0
    )
    return side_pen + angle_pen


def _nearest_contour_idx(cnt_pts, target):
    """
    Find the index in cnt_pts (Nx2 array) closest to target (2D point).
    """
    diffs = cnt_pts.astype(float) - np.array(target, dtype=float)
    return int(np.argmin(np.linalg.norm(diffs, axis=1)))


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

    cnt_convert = [c[0] for c in cnt]
    cnt_pts = np.array(cnt_convert)

    # -----------------------------------------------------------------------
    # Fast path: high-sigma scoring + rectangle fitness
    # At very high sigma, tabs/holes smooth away (net angle ≈ 0) while the
    # 4 real corner turns (~90°) survive as the dominant positive peaks.
    # We pick the 8 highest-scoring candidate points, try all C(8,4)=70
    # combinations, and select the one that best fits a rectangle.
    # -----------------------------------------------------------------------
    COARSE_SIGMA = 35
    FITNESS_THRESHOLD = 0.35

    coarse_angles_raw = get_relative_angles(cnt_pts, export=False, sigma=COARSE_SIGMA)
    min_gap = int(len(coarse_angles_raw) / 8)
    candidates = _pick_topN_spaced(coarse_angles_raw, min_gap, N=8)

    OFFSET_LOW_c = len(coarse_angles_raw) / 8
    OFFSET_HIGH_c = len(coarse_angles_raw) / 2.0
    SCORE_WEIGHT = 0.5  # weight of corner-confidence penalty vs. geometry

    # Normalize coarse angle scores to [0, 1] for confidence term
    norm_scores = coarse_angles_raw / (np.max(coarse_angles_raw) + 1e-9)

    def _valid_spacing(sp):
        gaps = [sp[1] - sp[0], sp[2] - sp[1], sp[3] - sp[2],
                sp[0] + len(coarse_angles_raw) - sp[3]]
        return all(OFFSET_LOW_c < g < OFFSET_HIGH_c for g in gaps)

    def _combined_fitness(sp):
        # Joint score: rectangle geometry + corner confidence
        rect_f = _rect_fitness(cnt_pts[list(sp)])
        conf_f = 1.0 - float(np.mean(norm_scores[list(sp)]))
        return rect_f + SCORE_WEIGHT * conf_f

    best_fit_coarse = None
    best_fitness = float('inf')
    best_strategy = None

    # Strategy 2a: 3 high-confidence anchors → derive 4th from rectangle law
    # D = A + C - B  (parallelogram law; diagonals bisect each other)
    for trio in itertools.combinations(candidates, 3):
        for mid_i in range(3):
            idxA = trio[(mid_i - 1) % 3]
            idxB = trio[mid_i]          # the corner between A and C
            idxC = trio[(mid_i + 1) % 3]
            A = cnt_pts[idxA].astype(float)
            B = cnt_pts[idxB].astype(float)
            C = cnt_pts[idxC].astype(float)
            D_computed = A + C - B
            idxD = _nearest_contour_idx(cnt_pts, D_computed)
            sp = sorted([idxA, idxB, idxC, idxD])
            if len(set(sp)) < 4 or not _valid_spacing(sp):
                continue
            fitness = _combined_fitness(sp)
            if fitness < best_fitness:
                best_fitness = fitness
                best_fit_coarse = np.array(sp, dtype=int)
                best_strategy = "3-anchor"

    # Strategy 2b: all 4 chosen from top candidates
    for combo in itertools.combinations(candidates, 4):
        sp = sorted(combo)
        if not _valid_spacing(sp):
            continue
        fitness = _combined_fitness(sp)
        if fitness < best_fitness:
            best_fitness = fitness
            best_fit_coarse = np.array(sp, dtype=int)
            best_strategy = "4-candidate"


    # Post-correction: for each of the 4 found corners, try replacing it with
    # the rectangle-derived position from the other 3 (D = A + C - B).
    # Handles the case where 3 corners are correct but 1 is slightly off.
    if best_fit_coarse is not None:
        for replace_i in range(4):
            others = [int(best_fit_coarse[j]) for j in range(4) if j != replace_i]
            for mid_i in range(3):
                idxA = others[(mid_i - 1) % 3]
                idxB = others[mid_i]   # middle corner (right angle between A and C)
                idxC = others[(mid_i + 1) % 3]
                D = (cnt_pts[idxA].astype(float) + cnt_pts[idxC].astype(float)
                     - cnt_pts[idxB].astype(float))
                idxD = _nearest_contour_idx(cnt_pts, D)
                new_sp = sorted([idxA, idxB, idxC, idxD])
                if len(set(new_sp)) < 4 or not _valid_spacing(new_sp):
                    continue
                new_fitness = _combined_fitness(new_sp)
                if new_fitness < best_fitness:
                    best_fitness = new_fitness
                    best_fit_coarse = np.array(new_sp, dtype=int)
                    best_strategy = "post-correction"
                    break  # take first improvement for this corner, move to next

    # Post-correction 2: for each of the 4 found corners, try replacing it with
    # each unused top candidate directly.  This breaks the "self-consistent wrong
    # corner" deadlock where D=A+C-B from the other 3 returns the same bad index.
    if best_fit_coarse is not None:
        for replace_i in range(4):
            others = [int(best_fit_coarse[j]) for j in range(4) if j != replace_i]
            for cand in candidates:
                if cand in others:
                    continue
                new_sp = sorted(others + [cand])
                if len(set(new_sp)) < 4 or not _valid_spacing(new_sp):
                    continue
                new_fitness = _combined_fitness(new_sp)
                if new_fitness < best_fitness:
                    best_fitness = new_fitness
                    best_fit_coarse = np.array(new_sp, dtype=int)
                    best_strategy = "post-correction-cand"

    print(f"  [coarse sigma={COARSE_SIGMA}] best_fitness={best_fitness:.3f} "
          f"strategy={best_strategy} "
          f"-> {'FAST PATH' if best_fitness < FITNESS_THRESHOLD else 'fallback to permutation'}")
    if best_fit_coarse is not None and best_fitness < FITNESS_THRESHOLD:
        # Classify edges using sigma=5 angle peaks
        class_angles = get_relative_angles(
            cnt_pts, export=(config.DEBUG_FILE_OUTPUT == 1), sigma=5
        )
        class_angles = np.array(class_angles)
        class_angles_inv = -class_angles

        max_rel = np.max(class_angles) if len(class_angles) > 0 else 0
        extr_c = detect_peaks(class_angles, mph=0.3 * max_rel) if max_rel > 0 else np.array([], dtype=int)
        max_rel_inv = np.max(class_angles_inv) if len(class_angles_inv) > 0 else 0
        extr_c_inv = detect_peaks(class_angles_inv, mph=0.3 * max_rel_inv) if max_rel_inv > 0 else np.array([], dtype=int)

        # Roll so the last corner lands near the end (matches existing logic)
        best_offset_c = len(class_angles) - best_fit_coarse[3] - 1
        best_fit_rolled = best_fit_coarse + best_offset_c
        extr_c = (extr_c + best_offset_c) % len(class_angles)
        extr_c_inv = (extr_c_inv + best_offset_c) % len(class_angles)

        for seg in [
            [0, best_fit_rolled[0]],
            [best_fit_rolled[0], best_fit_rolled[1]],
            [best_fit_rolled[1], best_fit_rolled[2]],
            [best_fit_rolled[2], best_fit_rolled[3]],
        ]:
            pos_inside = peaks_inside(seg, extr_c)
            neg_inside = peaks_inside(seg, extr_c_inv)
            pos_inside.sort()
            neg_inside.sort()
            t = type_peak(pos_inside, neg_inside)
            angle_sum = float(np.sum(class_angles[seg[0]:seg[1]]))
            # Low-sum UNDEFINED: the net angle change is small → likely noisy border edge.
            if t == TypeEdge.UNDEFINED and abs(angle_sum) < 1.0:
                t = TypeEdge.BORDER
            types_pieces.append(t)

        # Refine corner positions with sigma=2, wider window since coarse is imprecise
        _refined_angles = get_relative_angles(cnt_pts, export=False, sigma=2)
        _refine_win = max(COARSE_SIGMA, 15)
        _refined = []
        for _k in range(4):
            _orig = int(best_fit_coarse[_k])
            _i0 = max(0, _orig - _refine_win)
            _i1 = min(len(_refined_angles) - 1, _orig + _refine_win)
            _refined.append(int(_i0 + np.argmax(_refined_angles[_i0:_i1 + 1])))
        best_fit_tmp = np.array(_refined, dtype=int)

        for i in range(3):
            edges.append(cnt[best_fit_tmp[i]:best_fit_tmp[i + 1]])
        edges.append(
            np.concatenate((cnt[best_fit_tmp[3]:], cnt[:best_fit_tmp[0]]), axis=0)
        )
        edges = [np.array([x[0] for x in e]) for e in edges]

        n_borders_fast = sum(1 for t in types_pieces if t == TypeEdge.BORDER)
        if n_borders_fast > 0:
            types_pieces.append(types_pieces[0])
            return best_fit_tmp, edges, types_pieces[1:]

        # 0 border edges at sigma=5 → the border edge may have small noise peaks.
        # Retry classification at higher sigma where the border edge becomes truly flat.
        for hi_sigma in [10, 15, 20]:
            hi_angles = get_relative_angles(cnt_pts, export=False, sigma=hi_sigma)
            hi_angles = np.array(hi_angles)
            hi_angles_inv = -hi_angles
            max_hi = np.max(hi_angles) if len(hi_angles) > 0 else 0
            extr_hi = detect_peaks(hi_angles, mph=0.3 * max_hi) if max_hi > 0 else np.array([], dtype=int)
            max_hi_inv = np.max(hi_angles_inv) if len(hi_angles_inv) > 0 else 0
            extr_hi_inv = detect_peaks(hi_angles_inv, mph=0.3 * max_hi_inv) if max_hi_inv > 0 else np.array([], dtype=int)

            hi_offset = len(hi_angles) - best_fit_coarse[3] - 1
            hi_rolled = best_fit_coarse + hi_offset
            extr_hi = (extr_hi + hi_offset) % len(hi_angles)
            extr_hi_inv = (extr_hi_inv + hi_offset) % len(hi_angles)

            hi_types = []
            for seg in [
                [0, hi_rolled[0]],
                [hi_rolled[0], hi_rolled[1]],
                [hi_rolled[1], hi_rolled[2]],
                [hi_rolled[2], hi_rolled[3]],
            ]:
                pos_in = peaks_inside(seg, extr_hi)
                neg_in = peaks_inside(seg, extr_hi_inv)
                pos_in.sort(); neg_in.sort()
                t = type_peak(pos_in, neg_in)
                hi_sum = float(np.sum(hi_angles[seg[0]:seg[1]]))
                if t == TypeEdge.UNDEFINED and abs(hi_sum) < 1.0:
                    t = TypeEdge.BORDER
                hi_types.append(t)

            n_borders_hi = sum(1 for t in hi_types if t == TypeEdge.BORDER)
            print(f"  [fast path sigma={hi_sigma} retry] borders={n_borders_hi} types={[t.name for t in hi_types]}")
            if n_borders_hi > 0:
                types_pieces = hi_types
                types_pieces.append(types_pieces[0])
                return best_fit_tmp, edges, types_pieces[1:]

        # Still 0 borders after higher-sigma retries → fall to sigma=5 permutation
        print(f"  [fast path] 0 borders detected → falling back to sigma=5 permutation")
        types_pieces.clear()
        edges.clear()

    # -----------------------------------------------------------------------
    # Fallback: original sigma-escalation + permutation search
    # -----------------------------------------------------------------------
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
        relative_angles = get_relative_angles(
            np.array(cnt_convert),
            export=(config.DEBUG_FILE_OUTPUT == 1),  # saves fig*.png and save*.p
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
            print(f"  [sigma={sigma}] Too few peaks: {len(extr)} positive peaks found (need >= 4)")
            sigma += 1
            continue

        relative_angles = normalized(relative_angles[:, np.newaxis], axis=0).ravel()

        # Build list of permutations of 4 points
        combs = itertools.permutations(extr, 4)
        combs_l = list(combs)
        OFFSET_LOW = len(relative_angles) / 8
        OFFSET_HIGH = len(relative_angles) / 2.0

        # Use strict peak-count rules only for low sigma; high sigma is a last-resort
        # fallback for imperfect (e.g. paper-cutout) pieces where a stray peak can
        # land alone in one segment and block all candidates.
        strict = (sigma < 10)

        passed_length = 0
        passed_comb = 0
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
                passed_length += 1
                candidate = (comb[3], comb[2], comb[1], comb[0])
                if is_acceptable_comb(candidate, extr, len(relative_angles), strict) and is_acceptable_comb(
                        candidate, extr_inverse, len(relative_angles), strict
                ):
                    passed_comb += 1
                    tmp_combs_final.append(candidate)

        if len(tmp_combs_final) == 0:
            print(f"  [sigma={sigma}] peaks={len(extr)} passed_length={passed_length} passed_comb={passed_comb} (OFFSET_LOW={OFFSET_LOW:.0f}, peaks_pos={list(extr)})")
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

    # No valid fit found at all
    if best_fit is None or best_offset is None or len(types_pieces) == 0:
        return None, None, None

    if types_pieces[-1] == TypeEdge.UNDEFINED:
        print("UNDEFINED FOUND - try to continue but something bad happened :(")
        print(types_pieces[-1])

    # Back to original contour indexing — then refine each corner using a
    # lightly-smoothed (sigma=2) angle curve so the peak is closer to the true
    # geometric corner and both matching edges end up with consistent lengths.
    _refined_angles = get_relative_angles(
        np.array(cnt_convert), export=False, sigma=2
    )
    _refine_win = max(int(sigma * 1.5), 5)
    _coarse = (best_fit - best_offset).tolist()
    _refined = []
    for _k in range(4):
        _orig = _coarse[_k]
        _i0 = max(0, _orig - _refine_win)
        _i1 = min(len(_refined_angles) - 1, _orig + _refine_win)
        _refined.append(int(_i0 + np.argmax(_refined_angles[_i0 : _i1 + 1])))
    best_fit_tmp = np.array(_refined, dtype=int)

    for i in range(3):
        edges.append(cnt[best_fit_tmp[i] : best_fit_tmp[i + 1]])
    edges.append(
        np.concatenate((cnt[best_fit_tmp[3] :], cnt[: best_fit_tmp[0]]), axis=0)
    )

    # quick'n'dirty fix of the shape
    edges = [np.array([x[0] for x in e]) for e in edges]

    # Preserve original return format
    types_pieces.append(types_pieces[0])
    return best_fit_tmp, edges, types_pieces[1:]


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

    if config.DEBUG_FILE_OUTPUT == 1:
        signatures = [
            my_find_corner_signature(cnt, green)
            for cnt in contours
        ]
    else:
        with Pool(cpu_count()) as p:
            signatures = p.starmap(
                my_find_corner_signature, zip(contours, itertools.repeat(green))
            )

    # Cross-piece dimension check: all puzzle pieces should have the same
    # width and height (within ~30%). An outlier likely has bad corner detection.
    _dims = []
    for idx, cnt in enumerate(contours):
        corners_chk, _, _ = signatures[idx]
        if corners_chk is None:
            _dims.append(None)
            continue
        pts = np.array([cnt[int(c)][0] for c in corners_chk], dtype=float)
        sides = [np.linalg.norm(pts[(i + 1) % 4] - pts[i]) for i in range(4)]
        short = min((sides[0] + sides[2]) / 2, (sides[1] + sides[3]) / 2)
        long_ = max((sides[0] + sides[2]) / 2, (sides[1] + sides[3]) / 2)
        _dims.append((short, long_))

    _valid_dims = [d for d in _dims if d is not None]
    if len(_valid_dims) > 1:
        med_short = float(np.median([d[0] for d in _valid_dims]))
        med_long = float(np.median([d[1] for d in _valid_dims]))
        _DIM_TOL = 0.3
        for i, d in enumerate(_dims):
            if d is None:
                continue
            short, long_ = d
            short_err = abs(short - med_short) / (med_short + 1e-9)
            long_err = abs(long_ - med_long) / (med_long + 1e-9)
            status = "OUTLIER — corner detection may be wrong" if (short_err > _DIM_TOL or long_err > _DIM_TOL) else "OK"
            print(f"  [dim-check] Piece {i}: short={short:.0f}px (med {med_short:.0f}), "
                  f"long={long_:.0f}px (med {med_long:.0f}) — {status}")

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

    if config.DEBUG_FILE_OUTPUT == 1:
        signatures = [
            my_find_corner_signature(cnt, green)
            for cnt in contours
        ]
    else:
        with Pool(cpu_count()) as p:
            signatures = p.starmap(
                my_find_corner_signature, zip(contours, itertools.repeat(green))
            )

    # Cross-piece dimension check
    _dims2 = []
    for idx, cnt in enumerate(contours):
        corners_chk, _, _ = signatures[idx]
        if corners_chk is None:
            _dims2.append(None)
            continue
        pts = np.array([cnt[int(c)][0] for c in corners_chk], dtype=float)
        sides = [np.linalg.norm(pts[(i + 1) % 4] - pts[i]) for i in range(4)]
        short = min((sides[0] + sides[2]) / 2, (sides[1] + sides[3]) / 2)
        long_ = max((sides[0] + sides[2]) / 2, (sides[1] + sides[3]) / 2)
        _dims2.append((short, long_))
    _valid_dims2 = [d for d in _dims2 if d is not None]
    if len(_valid_dims2) > 1:
        med_short2 = float(np.median([d[0] for d in _valid_dims2]))
        med_long2 = float(np.median([d[1] for d in _valid_dims2]))
        _DIM_TOL2 = 0.3
        for i, d in enumerate(_dims2):
            if d is None:
                continue
            short, long_ = d
            status = "OUTLIER — corner detection may be wrong" if (
                abs(short - med_short2) / (med_short2 + 1e-9) > _DIM_TOL2
                or abs(long_ - med_long2) / (med_long2 + 1e-9) > _DIM_TOL2
            ) else "OK"
            print(f"  [dim-check] Piece {i}: short={short:.0f}px (med {med_short2:.0f}), "
                  f"long={long_:.0f}px (med {med_long2:.0f}) — {status}")

    # Debug: draw all detected corners on the original image
    if config.DEBUG_FILE_OUTPUT == 1:
        _corner_canvas = img.copy()
        _corner_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
        for idx, cnt in enumerate(contours):
            _c, _, _ = signatures[idx]
            if _c is None:
                continue
            cv2.drawContours(_corner_canvas, [cnt], 0, (60, 60, 60), 1)
            cnt_pts_dbg = np.array([cnt[int(ci)][0] for ci in _c])
            # label piece index near centroid
            _cx = int(cnt_pts_dbg[:, 0].mean())
            _cy = int(cnt_pts_dbg[:, 1].mean())
            cv2.putText(_corner_canvas, f"P{idx}", (_cx - 10, _cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
            for j, pt in enumerate(cnt_pts_dbg):
                cv2.circle(_corner_canvas, tuple(pt.astype(int)), 8,
                           _corner_colors[j % 4], -1)
                cv2.putText(_corner_canvas, str(j), (int(pt[0]) + 6, int(pt[1]) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imwrite(
            os.path.join(os.environ["ZOLVER_TEMP_DIR"], "corners_vis.png"),
            _corner_canvas,
        )

    for idx, cnt in enumerate(contours):
        corners, edges_shape, types_edges = signatures[idx]
        if corners is None or len(edges_shape) != 4:
            continue

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