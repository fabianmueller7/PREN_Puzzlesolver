"""Headless harness for the dedicated <=6-piece solver (SmallSolver).

Run:  python3 tools/run_smallsolver.py [image_path]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

DEFAULT_IMG = "resources/must works/NEUE-Teile.jpeg"
OUT_DIR = "geo_output"

from solver.Puzzle.Enums import TypeEdge

_COLOR = {
    TypeEdge.HOLE:      (255, 178, 102),
    TypeEdge.HEAD:      (102, 255, 255),
    TypeEdge.BORDER:    (0, 255, 0),
    TypeEdge.UNDEFINED: (0, 0, 255),
}


def render(pieces, path, pad=40):
    all_pts = np.concatenate(
        [np.asarray(e.shape, dtype=float) for p in pieces for e in p.edges_ if len(e.shape)]
    )
    mn = all_pts.min(0)
    span = (all_pts.max(0) - mn).astype(int) + 2 * pad
    canvas = np.zeros((span[1], span[0], 3), dtype=np.uint8)
    for p in pieces:
        for e in p.edges_:
            pts = np.asarray(e.shape, dtype=float)
            if len(pts) < 2:
                continue
            xy = (pts - mn + pad).astype(np.int32)
            cv2.polylines(canvas, [xy.reshape(-1, 1, 2)], False,
                          _COLOR.get(e.type, (200, 200, 200)), 1, cv2.LINE_AA)
            cv2.circle(canvas, tuple(xy[0]), 3, (160, 0, 160), -1)
    cv2.imwrite(path, canvas)
    print(f"[render] wrote {path}  ({canvas.shape[1]}x{canvas.shape[0]})")


def main():
    img = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMG
    os.makedirs(OUT_DIR, exist_ok=True)
    os.environ.setdefault("ZOLVER_TEMP_DIR", OUT_DIR)

    from solver.Puzzle.Puzzle import Puzzle
    from solver.Puzzle.SmallSolver import solve_small

    print(f"[small] extracting pieces from {img}")
    puzzle = Puzzle(img, viewer=None, green_screen=False)
    pieces = puzzle.pieces_
    print(f"[small] {len(pieces)} pieces extracted")

    ok = solve_small(pieces, green=False, log=print)
    print(f"[small] {'SOLVED' if ok else 'FAILED'}")
    if ok:
        coords = {id(p): getattr(p, 'coord', None) for p in pieces}
        for i, p in enumerate(pieces):
            print(f"  piece {i}: coord(row,col)={coords[id(p)]}")
        render(pieces, os.path.join(OUT_DIR, "small_result.png"))


if __name__ == "__main__":
    main()
