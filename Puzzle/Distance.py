import math
import cv2

import numpy as np
from numba import njit


@njit(cache=True)
def rgb2hsl(r, g, b):
    r /= 255
    g /= 255
    b /= 255
    minimum = min(r, g, b)
    maximum = max(r, g, b)
    med = (maximum + minimum) / 2
    h, l, s = med, med, med
    if maximum == minimum:
        return 0, 0, l

    d = maximum - minimum
    s = d / (2 - maximum - minimum) if l > 0.5 else d / (maximum + minimum)
    if maximum == r:
        h = (g - b) / d + (6 if g < b else 0)
    elif maximum == g:
        h = (b - r) / d + 2
    elif maximum == b:
        h = (r - g) / d + 4
    return h / 6, s, l


@njit(cache=True)
def rgb2lab(r, g, b, drop_l=False):
    """Fast Implementation of rgb2lab function (skimage too slow)"""
    exp = 1 / 3

    r /= 255
    r = (((r + 0.055) / 1.055) ** 2.4 if r > 0.04045 else r / 12.92) * 100

    g = g / 255
    g = (((g + 0.055) / 1.055) ** 2.4 if g > 0.04045 else g / 12.92) * 100

    b = b / 255
    b = (((b + 0.055) / 1.055) ** 2.4 if b > 0.04045 else b / 12.92) * 100

    x = r * 0.4124 + g * 0.3576 + b * 0.1805
    x = round(x, 4) / 95.047
    x = x**exp if x > 0.008856 else (7.787 * x) + (16.0 / 116.0)

    y = r * 0.2126 + g * 0.7152 + b * 0.0722
    y = round(y, 4) / 100.0
    y = y**exp if y > 0.008856 else (7.787 * y) + (16.0 / 116.0)

    z = r * 0.0193 + g * 0.1192 + b * 0.9505
    z = round(z, 4) / 108.883
    z = z**exp if z > 0.008856 else (7.787 * z) + (16.0 / 116.0)

    L = (116.0 * y) - 16
    a = 500.0 * (x - y)
    b = 200.0 * (y - z)

    return round(L, 4) if not drop_l else 0.0, round(a, 4), round(b, 4)


@njit(cache=True)
def dist(p1, p2):
    """
    Compute euclidean distance

    :param p1: first coordinate tuple
    :param p2: second coordinate tuple
    :return: distance Float
    """
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p1[1] - p2[1]) ** 2)


@njit(cache=True)
def dist_edge(e1_begin, e1_end, e2_begin, e2_end):
    """
    Compute the size difference between two edges

    :param e1: Matrix of coordinates of points composing the first edge
    :param e2: Matrix of coordinates of points composing the second edge
    :return: Float
    """
    dist_e1 = dist(e1_begin, e1_end)
    dist_e2 = dist(e2_begin, e2_end)
    res = math.fabs(dist_e1 - dist_e2)
    val = (dist_e1 + dist_e2) / 2
    return res, val


@njit(cache=True)
def have_edges_similar_length(e1_begin, e1_end, e2_begin, e2_end, percent):
    """
    Return a boolean to determine if the difference between two edges is > 20%

    :param e1: Matrix of coordinates of points composing the first edge
    :param e2: Matrix of coordinates of points composing the second edge
    :return: Boolean
    """
    res, val = dist_edge(e1_begin, e1_end, e2_begin, e2_end)
    return res < (val * percent)


def normalize_vect_len(e1, e2):
    """
    Return the shortest and the longest edges.

    :param e1: Matrix of coordinates of points composing the first edge
    :param e2: Matrix of coordinates of points composing the second edge
    :return: Matrix of coordinates, Matrix of coordinates
    """

    longest = e1 if len(e1) > len(e2) else e2
    shortest = e2 if len(e1) > len(e2) else e1
    return shortest, longest


def diff_match_edges(e1, e2, reverse=True):
    """
    Return the distance between two edges.

    :param e1: Matrix of coordinates of points composing the first edge
    :param e2: Matrix of coordinates of points composing the second edge
    :param reverse: Optional parameter to reverse the second edge
    :return: distance Float
    """

    shortest, longest = normalize_vect_len(e1, e2)
    diff = 0
    for i, p in enumerate(shortest):
        ratio = i / len(shortest)
        j = int(len(longest) * ratio)
        x1 = longest[j]
        x2 = shortest[len(shortest) - i - 1] if reverse else shortest[i]
        diff += (x2 - x1) ** 2
    return diff / len(shortest)


def diff_match_edges2(e1, e2, reverse=True, thres=5, pad=False):
    """
    Return the distance between two edges by performing a simple norm on each points.

        :param e1: Matrix of coordinates of points composing the first edge
        :param e2: Matrix of coordinates of points composing the second edge
        :param reverse: Optional parameter to reverse the second edge
        :return: distance Float
    """
    if e2.shape[0] > e1.shape[0]:
        e1, e2 = e2, e1

    if pad:
        pad_length = (e1.shape[0] - e2.shape[0]) // 2
        pad_left, pad_right = (
            pad_length,
            (
                pad_length
                if pad_length * 2 == (e1.shape[0] - e2.shape[0])
                else pad_length + 1
            ),
        )
        e2 = np.lib.pad(
            e2, ((pad_left, pad_right), (0, 0)), "constant", constant_values=(0, 0)
        )
    else:
        e1 = e1[: e2.shape[0]]

    if reverse:
        e2 = np.flip(e2, 0)

    d = np.linalg.norm(e1 - e2, axis=1)
    return np.sum(d > thres) / max(e1.shape[0], 1), 0


@njit(cache=True)
def dist_color(t1_0, t1_1, t1_2, t2_0, t2_1, t2_2):
    return np.sqrt((t1_0 - t2_0) ** 2 + (t1_1 - t2_1) ** 2 + (t1_2 - t2_2) ** 2)


def euclidean_distance(e1_lab_colors, e2_lab_colors):
    len1 = len(e1_lab_colors)
    len2 = len(e2_lab_colors)
    maximum = max(len1, len2)
    t1 = len1 / maximum
    t2 = len2 / maximum
    return sum(
        [
            dist_color(*e1_lab_colors[int(t1 * i)], *e2_lab_colors[int(t2 * i)])
            for i in range(maximum)
        ]
    )


@njit(cache=True)
def hue2rgb(p, q, t):
    if t < 0:
        t += 1
    if t > 1:
        t -= 1
    if t < 1 / 6:
        return p + (q - p) * 6 * t
    if t < 1 / 2:
        return q
    if t < 2 / 3:
        return p + (q - p) * (2 / 3 - t) * 6
    return p


@njit(cache=True)
def hsl2rgb(h, s, l):
    if s == 0:
        return l, l, l
    q = l * (1 + s) if l < 0.5 else l + s - l * s
    p = 2 * l - q
    return (
        hue2rgb(p, q, h + 1.0 / 3) * 255,
        hue2rgb(p, q, h) * 255,
        hue2rgb(p, q, h - 1.0 / 3) * 255,
    )


def get_colors(edge):
    return [
        rgb2lab(*hsl2rgb(col[0], col[1], col[2]), drop_l=True) for col in edge.color
    ]


def real_edge_compute(e1, e2):
    return _edge_match_score(e1, e2)


def generated_edge_compute(e1, e2):
    return _edge_match_score(e1, e2)


def _polyline_length(points):
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def _resample_curve(points, n_points=128):
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) == 0:
        return np.zeros((n_points, 2), dtype=np.float32)
    if len(pts) == 1:
        return np.repeat(pts[:1], n_points, axis=0)

    deltas = np.diff(pts, axis=0)
    seg_len = np.linalg.norm(deltas, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(s[-1])
    if total <= 1e-8:
        return np.repeat(pts[:1], n_points, axis=0)

    t = np.linspace(0.0, total, n_points)
    out = np.empty((n_points, 2), dtype=np.float32)
    j = 0
    for i, tt in enumerate(t):
        while j + 1 < len(s) and s[j + 1] < tt:
            j += 1
        if j + 1 >= len(pts):
            out[i] = pts[-1]
        else:
            denom = s[j + 1] - s[j]
            alpha = 0.0 if denom <= 1e-8 else (tt - s[j]) / denom
            out[i] = pts[j] + alpha * (pts[j + 1] - pts[j])
    return out


def _signed_profile(points, n_points=128):
    curve = _resample_curve(points, n_points=n_points)
    start = curve[0]
    end = curve[-1]
    chord_vec = end - start
    chord = float(np.linalg.norm(chord_vec))
    if chord <= 1e-8:
        return curve, np.zeros(n_points, dtype=np.float32), np.zeros(n_points, dtype=np.float32), 0.0

    tangent = chord_vec / chord
    rel = curve - start
    tangential = rel @ tangent
    normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
    signed = rel @ normal
    return curve, tangential, signed, chord


def _trim_slice(n, trim_ratio=0.12):
    trim = int(round(n * trim_ratio))
    trim = min(trim, max(0, n // 3 - 1))
    return slice(trim, max(trim + 1, n - trim))


def _same_sign_penalty(p1, p2):
    prod = p1 * p2
    return float(np.mean(np.maximum(prod, 0.0)))


def _profile_feature_scale(profile1, profile2):
    s1 = float(np.max(np.abs(profile1))) if len(profile1) else 0.0
    s2 = float(np.max(np.abs(profile2))) if len(profile2) else 0.0
    return max((s1 + s2) * 0.5, 1.0)


def _compatible_length(e1, e2):
    pts1 = np.asarray(e1.shape, dtype=np.float32)
    pts2 = np.asarray(e2.shape, dtype=np.float32)
    if len(pts1) < 2 or len(pts2) < 2:
        return False

    c1 = float(np.linalg.norm(pts1[-1] - pts1[0]))
    c2 = float(np.linalg.norm(pts2[-1] - pts2[0]))
    a1 = _polyline_length(pts1)
    a2 = _polyline_length(pts2)

    chord_ratio = min(c1, c2) / max(max(c1, c2), 1.0)
    arc_ratio = min(a1, a2) / max(max(a1, a2), 1.0)
    return chord_ratio >= 0.55 and arc_ratio >= 0.45


def _edge_match_score(e1, e2, n_points=128):
    """
    Robust edge score for already aligned edges.

    `stick_pieces(...)` has already translated/rotated the candidate piece, so this
    function must *not* re-center or freely re-rotate edges. We compare the two
    curves as they are and prefer complementary profiles (tab vs hole).
    """
    if not _compatible_length(e1, e2):
        return float("inf")

    pts1 = np.asarray(e1.shape, dtype=np.float32)
    pts2 = np.asarray(e2.shape, dtype=np.float32)
    if len(pts1) < 2 or len(pts2) < 2:
        return float("inf")

    c1, t1, p1, chord1 = _signed_profile(pts1, n_points=n_points)
    c2, t2, p2, chord2 = _signed_profile(pts2[::-1], n_points=n_points)

    keep = _trim_slice(n_points, trim_ratio=0.14)
    p1k = p1[keep]
    p2k = p2[keep]
    t1k = t1[keep]
    t2k = t2[keep]
    c1k = c1[keep]
    c2k = c2[keep]

    scale = max((chord1 + chord2) * 0.5, 1.0)
    feature_scale = _profile_feature_scale(p1k, p2k)

    # Complementary edges should mirror around the baseline.
    complement_err = float(np.mean(np.abs(p1k + p2k))) / feature_scale

    # Same-sign shapes (hole-hole / head-head) create a large positive product.
    same_sign_penalty = _same_sign_penalty(p1k, p2k) / (feature_scale ** 2)

    # Small tangential mismatch is tolerated, but strong drift means bad corner split.
    tangent_err = float(np.mean(np.abs(t1k - t2k))) / scale

    # Absolute aligned fit still matters, but only on the trimmed interior.
    fit_err = float(np.mean(np.linalg.norm(c1k - c2k, axis=1))) / scale

    endpoint_err = 0.5 * (
        np.linalg.norm(c1[0] - c2[0]) + np.linalg.norm(c1[-1] - c2[-1])
    ) / scale

    color_err = 0.0
    try:
        if getattr(e1, "color", None) is not None and getattr(e2, "color", None) is not None:
            if len(e1.color) > 0 and len(e2.color) > 0:
                color_err = euclidean_distance(get_colors(e1), get_colors(e2)) / max(len(e1.color), len(e2.color), 1)
    except Exception:
        color_err = 0.0

    return (
        2.8 * complement_err
        + 4.5 * same_sign_penalty
        + 0.9 * fit_err
        + 0.35 * tangent_err
        + 0.25 * endpoint_err
        + 0.01 * color_err
    )
