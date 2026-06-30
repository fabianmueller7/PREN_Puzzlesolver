"""Collect pixel↔robot-mm correspondences on a grid, for tools/calibrate_mapping.py.

Procedure (improves pickup precision):
  1. Attach a marker (pen/sticker) to the gripper, or grip a small marker piece.
  2. --plan        : print the grid of robot (x,y) points the run will visit.
  3. --run         : drive the robot to each grid point and tap down to leave a
                     mark (Pi + robot). Marks accumulate on the field.
  4. Capture the warped playfield (the normal pipeline capture, e.g. debug_output/
     capture.jpg) so every mark is visible.
  5. --detect IMG  : find the marks and AUTO-PAIR each to its grid point using the
                     inverse of the current affine CAL_M (approximately right, so
                     nearest-match is unambiguous). Writes corr.csv.
  6. python tools/calibrate_mapping.py corr.csv  -> fit + diagnose + new mapping.

Grid covers the pickup zone in robot mm; tune with --x0/--x1/--y0/--y1/--nx/--ny.
"""
import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from solver import config  # noqa: E402


def grid_points(a):
    xs = np.linspace(a.x0, a.x1, a.nx)
    ys = np.linspace(a.y0, a.y1, a.ny)
    return [(round(float(x), 1), round(float(y), 1)) for y in ys for x in xs]


def robot_to_pixel_affine(rx, ry):
    """Invert the affine CAL_M (px->mm) to get mm->px, for auto-pairing detections."""
    M = np.array(config.CAL_M, dtype=float)          # 2x3
    A, t = M[:, :2], M[:, 2]
    return tuple(np.linalg.solve(A, np.array([rx, ry]) - t))


def detect_marks(image_path, n_expected):
    import cv2
    img = cv2.imread(image_path)
    if img is None:
        raise SystemExit(f"could not read {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    pts = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 4 or a > 0.02 * h * w:     # dots: not specks, not pieces
            continue
        M = cv2.moments(c)
        pts.append((M["m10"] / M["m00"], M["m01"] / M["m00"]))
    print(f"detected {len(pts)} marks (expected {n_expected})")
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true", help="print grid points and exit")
    ap.add_argument("--run", action="store_true", help="drive robot, tap a mark at each point")
    ap.add_argument("--detect", metavar="IMG", help="detect marks in IMG and write corr.csv")
    ap.add_argument("--out", default="corr.csv")
    ap.add_argument("--port", default="/dev/ttyACM0")
    # grid extent (robot mm) — default covers the observed pickup zone
    ap.add_argument("--x0", type=float, default=70);  ap.add_argument("--x1", type=float, default=250)
    ap.add_argument("--y0", type=float, default=210); ap.add_argument("--y1", type=float, default=350)
    ap.add_argument("--nx", type=int, default=5);     ap.add_argument("--ny", type=int, default=4)
    a = ap.parse_args()

    pts = grid_points(a)

    if a.plan or not (a.run or a.detect):
        print(f"{len(pts)} grid points (robot mm):")
        for i, (x, y) in enumerate(pts):
            print(f"  {i:2d}  X={x:6.1f}  Y={y:6.1f}")
        return

    if a.run:
        from robot.pico_interface import PicoInterface
        robot = PicoInterface(port=a.port)
        try:
            robot.motors_enable(); robot.home_z(); robot.gripper_up()
            robot.home_x(); robot.home_y(); robot.gripper_up()
            for i, (x, y) in enumerate(pts):
                print(f"-> mark {i}: X={x} Y={y}")
                robot.go_to(x, y)
                robot.gripper_down(); robot.gripper_up()
            print("Done. Now capture the warped field, then run --detect.")
        finally:
            robot.motors_disable(); robot.close()
        return

    if a.detect:
        marks = detect_marks(a.detect, len(pts))
        # auto-pair: each grid point -> nearest detected mark (by expected pixel)
        used, rows = set(), []
        for (x, y) in pts:
            ex, ey = robot_to_pixel_affine(x, y)
            best, bi = None, None
            for j, (px, py) in enumerate(marks):
                if j in used:
                    continue
                d = (px - ex) ** 2 + (py - ey) ** 2
                if best is None or d < best:
                    best, bi = d, j
            if bi is None:
                print(f"  WARN no mark for grid ({x},{y})"); continue
            used.add(bi)
            px, py = marks[bi]
            rows.append((px, py, x, y))
        with open(a.out, "w") as f:
            f.write("# pixel_x, pixel_y, robot_x_mm, robot_y_mm\n")
            for px, py, x, y in rows:
                f.write(f"{px:.2f}, {py:.2f}, {x}, {y}\n")
        print(f"wrote {len(rows)} correspondences -> {a.out}")
        print(f"next: python tools/calibrate_mapping.py {a.out}")


if __name__ == "__main__":
    main()
