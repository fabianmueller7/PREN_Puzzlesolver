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

        # Pad the shortest with 0
        e2 = np.lib.pad(
            e2, ((pad_left, pad_right), (0, 0)), "constant", constant_values=(0, 0)
        )
    else:
        # No padding just cut longest to match shortest length
        e1 = e1[: e2.shape[0]]

    #if reverse:
        #e2 = np.flip(e2, 0)
    #d = np.linalg.norm(e1 - e2, axis=1)
    #return np.sum(d > thres) / e1.shape[0]

    if reverse:
        e2 = np.flip(e2, 0)

    # 🔁 Kontinuierliche 360°-Rotationsprüfung
    best_score = float("inf")
    best_angle = 0

    # Prüfe alle Winkel von 0° bis 359° (1°-Schritte)
    for angle in range(0, 360):
        # Rotationsmatrix um den Ursprung (0,0)
        M = cv2.getRotationMatrix2D((0, 0), angle, 1.0)
        e2_rot = cv2.transform(np.array([e2]), M)[0]

        # Euklidische Distanz zwischen den Kantenpunkten berechnen
        d = np.linalg.norm(e1 - e2_rot, axis=1)
        score = np.sum(d > thres) / e1.shape[0]

        # Besten Score und zugehörigen Winkel merken
        if score < best_score:
            best_score = score
            best_angle = angle

    # 🔹 optional: besten Winkel im Edge-Objekt speichern
    try:
        e2.rotation_angle = best_angle
    except Exception:
        pass

    return best_score, best_angle


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
    """
    Geometry-only distance between two edges (for 'real' puzzle).
    Uses only shape, ignores colors completely.
    """

    # Optional quick check: reject edges whose lengths differ too much
    if not have_edges_similar_length(
            e1.shape[0], e1.shape[-1], e2.shape[0], e2.shape[-1], 0.20
    ):
        return float("inf")

    return _edge_geom_distance(e1, e2)


def generated_edge_compute(e1, e2):
    """
    Geometry-only distance between two edges (for generated puzzles).
    Same logic as real_edge_compute so behaviour is consistent.
    """

    if not have_edges_similar_length(
            e1.shape[0], e1.shape[-1], e2.shape[0], e2.shape[-1], 0.20
    ):
        return float("inf")

    return _edge_geom_distance(e1, e2)

def _resample_curve(points, n_points=100):
    """
    Resample a sequence of 2D points to exactly n_points along its arc length.
    points: (N, 2) array-like
    """
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) < 2:
        return np.repeat(pts[:1], n_points, axis=0)

    # cumulative arc length
    deltas = np.diff(pts, axis=0)
    seg_len = np.linalg.norm(deltas, axis=1)
    s = np.concatenate([[0], np.cumsum(seg_len)])
    total = s[-1]
    if total == 0:
        return np.repeat(pts[:1], n_points, axis=0)

    # target positions along the curve
    t = np.linspace(0, total, n_points)

    resampled = []
    j = 0
    for tt in t:
        while j + 1 < len(s) and s[j + 1] < tt:
            j += 1
        if j + 1 >= len(pts):
            resampled.append(pts[-1])
        else:
            ratio = (tt - s[j]) / (s[j + 1] - s[j] + 1e-8)
            resampled.append(pts[j] + ratio * (pts[j + 1] - pts[j]))
    return np.asarray(resampled, dtype=np.float32)


def _edge_geom_distance(e1, e2, n_points=100):
    """
    Purely geometric distance between two Edge objects.
    Uses only edge.shape, ignores any color.

    Returns a single float: smaller = more similar.
    """
    pts1 = np.asarray(e1.shape, dtype=np.float32)
    pts2 = np.asarray(e2.shape, dtype=np.float32)

    if len(pts1) < 2 or len(pts2) < 2:
        return float("inf")

    # Make shapes translation-invariant
    pts1 = pts1 - pts1.mean(axis=0, keepdims=True)
    pts2 = pts2 - pts2.mean(axis=0, keepdims=True)

    # Resample both curves to same length
    c1 = _resample_curve(pts1, n_points=n_points)
    c2 = _resample_curve(pts2, n_points=n_points)

    # Direct direction
    d1 = np.linalg.norm(c1 - c2, axis=1).mean()

    # Flipped direction (reverse one edge)
    d2 = np.linalg.norm(c1 - c2[::-1], axis=1).mean()

    return min(d1, d2)



#def generated_edge_compute(e1, e2):
    """
    Return the distance between colors of two edges for generated puzzle.

    :param e1: Edge object
    :param e2: Edge object
    :return: distance Float
    """
    # edge size
    shapevalue, distvalue = dist_edge(
        e1.shape[0], e1.shape[-1], e2.shape[0], e2.shape[-1]
    )

    # edges diff
    edge_shape_score = diff_match_edges2(np.array(e1.shape), np.array(e2.shape))
    # Sigmoid
    K = -1.05
    edge_shape_score = (K * edge_shape_score) / (K - edge_shape_score + 1)

    # colors
    e1_lab_colors = get_colors(e1)
    e2_lab_colors = get_colors(e2)
    val = min(
        euclidean_distance(e1_lab_colors, e2_lab_colors),
        euclidean_distance(e1_lab_colors, e2_lab_colors[::-1]),
    )
    return val * (1.0 + math.sqrt(shapevalue) * 0.3) * (1.0 + edge_shape_score * 0.001)
