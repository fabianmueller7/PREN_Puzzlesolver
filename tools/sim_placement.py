"""Offline simulation of robot piece placement — no robot, no physical run.

Renders what the robot WILL produce, in robot-mm coordinates, by replaying the
exact transform move_pieces() applies to each piece:

    final_shape = rotate(piece_contour, ROTATION_SIGN * rotation_deg)
                  translated so its centroid lands on end_center_robot_mm

The piece contours are taken from the warped border-detected capture and mapped
into robot mm via config.pixel_to_robot, so the picture is faithful to the real
table. Compare the output to the solver's stick.png (the correct solution):
if the config is right, the pieces tessellate the same way.

Usage:
    python tools/sim_placement.py \
        --debug "/path/to/debug_output" \
        --out   predicted_placement.png

It reads piece_centers.json + capture.jpg from --debug. ROTATION_SIGN and all
grid/rotation config come from solver.config, so editing config.py and re-running
this script shows the effect immediately.
"""
import sys
import os
import json
import argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from solver import config  # noqa: E402

try:
    ROTATION_SIGN = __import__("main").ROTATION_SIGN  # reuse the real sign
except Exception:
    ROTATION_SIGN = +1

# PUZZLE_TARGET value that was active when piece_centers.json was generated.
# Used to back out the base Kabsch angle so the sim can apply the CURRENT config.
# Defaults to the current config (correct for freshly-generated jsons); set
# SIM_JSON_PT to replay an older json (e.g. =90 for pre-placement-fix logs).
JSON_PT = float(os.environ.get("SIM_JSON_PT", str(config.PUZZLE_TARGET_ROTATION_DEG)))


def _extract_contours(capture_path):
    """Return list of (centroid_px, contour_px) for the dark pieces on white."""
    img = cv2.imread(capture_path)
    if img is None:
        raise RuntimeError(f"Could not read {capture_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # pieces are dark on a light field
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    h, w = gray.shape
    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.01 * h * w:        # drop specks
            continue
        m = cv2.moments(c)
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        out.append((np.array([cx, cy]), c.reshape(-1, 2).astype(float)))
    return out


def _to_robot(pts_px):
    """Map an (N,2) array of warped-image px to robot mm via config.pixel_to_robot."""
    return np.array([config.pixel_to_robot(x, y) for x, y in pts_px], dtype=float)


def _rotate(pts, deg, about):
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    R = np.array([[c, -s], [s, c]])
    return (pts - about) @ R.T + about


def simulate(debug_dir, out_path):
    pcs = json.load(open(os.path.join(debug_dir, "piece_centers.json")))
    contours = _extract_contours(os.path.join(debug_dir, "capture.jpg"))

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#e6194B", "#3cb44b", "#4363d8", "#f58231"]

    # End positions via the same rigid map the solver/robot use (solved centroid →
    # robot mm, assembly recentred). Indexed by piece order, matching pcs.
    robot_ends = config.assembly_to_robot([p.get("end_center_px") for p in pcs])

    for p in pcs:
        idx = p["piece_index"]
        start_px = np.array(p["start_center_px"], dtype=float)
        gn, ge = p["grid_coord"]
        end_mm = np.array(robot_ends[idx], dtype=float)
        base = p["rotation_deg"] - JSON_PT
        deg = base + config.PUZZLE_TARGET_ROTATION_DEG \
            + config.COLUMN_ROTATION_CORRECTIONS.get(ge, 0.0) \
            + config.ROW_ROTATION_CORRECTIONS.get(gn, 0.0)
        angle = ROTATION_SIGN * deg

        # nearest extracted contour to this piece's start centre
        cont = min(contours, key=lambda cc: np.hypot(*(cc[0] - start_px)))[1]

        cont_mm = _to_robot(cont)
        centroid = cont_mm.mean(axis=0)
        placed = _rotate(cont_mm, angle, centroid)         # rotate about own centre
        placed = placed - centroid + end_mm                # translate centroid -> target

        poly = plt.Polygon(placed, closed=True, alpha=0.55,
                           facecolor=colors[idx % 4], edgecolor="black", linewidth=1.2)
        ax.add_patch(poly)
        gc = p.get("grid_coord")
        ax.text(*end_mm, f"{idx}\n{gc}\n{p['rotation_deg']:.0f}°",
                ha="center", va="center", fontsize=8, weight="bold")

    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.invert_yaxis()   # robot Y increases downward
    ax.set_title("Predicted placement (robot mm) — compare to stick.png")
    ax.set_xlabel("robot X [mm] (increases LEFT)")
    ax.set_ylabel("robot Y [mm] (increases DOWN)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    print(f"ROTATION_SIGN={ROTATION_SIGN}  PUZZLE_TARGET={config.PUZZLE_TARGET_ROTATION_DEG}"
          f"  ROW_corr={config.ROW_ROTATION_CORRECTIONS}  COL_corr={config.COLUMN_ROTATION_CORRECTIONS}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", default="debug_output", help="debug_output dir with piece_centers.json + capture.jpg")
    ap.add_argument("--out", default="debug_output/predicted_placement.png")
    args = ap.parse_args()
    simulate(args.debug, args.out)
