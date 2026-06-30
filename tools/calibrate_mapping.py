"""Fit & diagnose the pixel→robot-mm mapping from measured grid correspondences.

Goal: improve pickup precision. The current mapping (solver/config.py CAL_M) is a
6-DOF AFFINE fit from 4 corners — it cannot represent perspective or the wide-angle
Pi camera's lens distortion, so its error grows toward the field edges.

Feed this tool a set of (pixel, robot_mm) correspondences measured across the field
and it will:
  * fit AFFINE (current model), HOMOGRAPHY (8-DOF, exact for a flat plane), and a
    2nd-order POLYNOMIAL (captures residual lens distortion empirically),
  * report each model's residual (RMS / max, and center vs edge) — this reveals the
    error SIGNATURE: if affine max >> RMS and the worst points are at the edges, it's
    distortion and the polynomial wins,
  * print the winning model's coefficients ready to paste into config.py.

Input file (CSV, one correspondence per line, '#' comments allowed):
    pixel_x, pixel_y, robot_x_mm, robot_y_mm

Usage:
    python tools/calibrate_mapping.py corr.csv
    python tools/calibrate_mapping.py corr.csv --image-w 906 --image-h 648

How to collect corr.csv: see tools/calibrate_grid_collect.py — it drives the robot
to a grid of known robot-mm points (dropping a marker piece), so you only need to
read off each marker's pixel centre from the warped capture.
"""
import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def load_csv(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = [p for p in line.replace(",", " ").split() if p]
            if len(parts) < 4:
                continue
            rows.append([float(p) for p in parts[:4]])
    a = np.asarray(rows, dtype=float)
    if len(a) < 4:
        raise SystemExit(f"Need >=4 correspondences, got {len(a)} from {path}")
    return a[:, :2], a[:, 2:]          # pixels (N,2), robot mm (N,2)


# ---- models: each returns a predict(px) -> mm function ----------------------

def fit_affine(px, mm):
    A = np.c_[px, np.ones(len(px))]                 # [x, y, 1]
    coef, *_ = np.linalg.lstsq(A, mm, rcond=None)   # (3,2)
    return lambda P: np.c_[P, np.ones(len(P))] @ coef, ("affine", coef)


def fit_poly2(px, mm):
    def feats(P):
        x, y = P[:, 0], P[:, 1]
        return np.c_[np.ones(len(P)), x, y, x * x, x * y, y * y]
    A = feats(px)
    coef, *_ = np.linalg.lstsq(A, mm, rcond=None)   # (6,2)
    return lambda P: feats(P) @ coef, ("poly2", coef)


def fit_homography(px, mm):
    # DLT: solve for H (3x3) mapping pixel->mm in homogeneous coords.
    x, y = px[:, 0], px[:, 1]
    u, v = mm[:, 0], mm[:, 1]
    z = np.ones(len(px))
    O = np.zeros(len(px))
    rows_u = np.c_[x, y, z, O, O, O, -u * x, -u * y, -u]
    rows_v = np.c_[O, O, O, x, y, z, -v * x, -v * y, -v]
    M = np.vstack([rows_u, rows_v])
    _, _, Vt = np.linalg.svd(M)
    H = Vt[-1].reshape(3, 3)

    def predict(P):
        Ph = np.c_[P, np.ones(len(P))] @ H.T
        return Ph[:, :2] / Ph[:, 2:3]
    return predict, ("homography", H)


def residuals(predict, px, mm):
    err = np.linalg.norm(predict(px) - mm, axis=1)   # mm per point
    return err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="correspondences: pixel_x,pixel_y,robot_x_mm,robot_y_mm")
    ap.add_argument("--image-w", type=float, default=906.0)
    ap.add_argument("--image-h", type=float, default=648.0)
    args = ap.parse_args()

    px, mm = load_csv(args.csv)
    cx, cy = args.image_w / 2, args.image_h / 2
    # "edge" = outer third of the frame by radius from image centre
    r = np.hypot(px[:, 0] - cx, px[:, 1] - cy)
    rmax = np.hypot(cx, cy)
    edge = r > 0.66 * rmax

    print(f"Loaded {len(px)} correspondences ({edge.sum()} edge / {(~edge).sum()} center)\n")
    print(f"{'model':<12}{'RMS mm':>9}{'max mm':>9}{'center RMS':>12}{'edge RMS':>11}")
    results = []
    for fit in (fit_affine, fit_homography, fit_poly2):
        predict, (name, coef) = fit(px, mm)
        e = residuals(predict, px, mm)
        rms = float(np.sqrt(np.mean(e**2)))
        cen = float(np.sqrt(np.mean(e[~edge]**2))) if (~edge).any() else float("nan")
        edg = float(np.sqrt(np.mean(e[edge]**2))) if edge.any() else float("nan")
        results.append((name, coef, rms, e.max(), cen, edg))
        print(f"{name:<12}{rms:9.2f}{e.max():9.2f}{cen:12.2f}{edg:11.2f}")

    # signature read-out
    aff = results[0]
    print("\n--- diagnosis ---")
    if aff[5] > 2 * aff[4] + 0.3:
        print(f"Affine error is much worse at the edges ({aff[5]:.2f} vs {aff[4]:.2f} mm RMS)")
        print("  => lens distortion / perspective. The polynomial (or homography) is the fix.")
    elif aff[2] < 0.6:
        print(f"Affine is already accurate (RMS {aff[2]:.2f} mm). Remaining error is likely")
        print("  mechanical (backlash/homing) or detection noise, not the mapping.")
    else:
        print(f"Affine RMS {aff[2]:.2f} mm, fairly uniform. A homography/poly may still help a bit.")

    best = min(results, key=lambda t: t[2])
    print(f"\nBest model: {best[0]} (RMS {best[2]:.2f} mm, max {best[3]:.2f} mm)")
    if best[0] == "poly2":
        c = best[1]
        fx = ", ".join(f"{float(v):.8g}" for v in c[:, 0])
        fy = ", ".join(f"{float(v):.8g}" for v in c[:, 1])
        print("\nPaste into solver/config.py:")
        print("CAL_POLY = [")
        print(f"    [{fx}],  # robot_x coeffs")
        print(f"    [{fy}],  # robot_y coeffs")
        print("]   # features: [1, x, y, x^2, x*y, y^2]")
    elif best[0] == "affine":
        c = best[1]
        print("\nCAL_M (affine) update for solver/config.py:")
        print("CAL_M = [")
        print(f"    [{c[0,0]:.6f}, {c[1,0]:.6f}, {c[2,0]:.6f}],")
        print(f"    [{c[0,1]:.6f}, {c[1,1]:.6f}, {c[2,1]:.6f}],")
        print("]")


if __name__ == "__main__":
    main()
