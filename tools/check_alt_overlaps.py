import os
import sys
import time
import queue

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Puzzle.Puzzle import Puzzle
from Puzzle.alternative_solver import AlternativeSolver

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--image', required=True)
parser.add_argument('--wait', type=float, default=30.0)
args = parser.parse_args()

img_path = os.path.abspath(args.image)
if not os.path.exists(img_path):
    print('Image not found:', img_path)
    sys.exit(1)

# use a temp dir
import time as _t
outdir = os.path.join(REPO_ROOT, 'debug_output', f'check_alt_{_t.strftime("%Y%m%d_%H%M%S")}')
os.makedirs(outdir, exist_ok=True)
os.environ['ZOLVER_TEMP_DIR'] = outdir

puzzle = Puzzle(img_path, green_screen=False)
alt_q = queue.Queue()
alt = AlternativeSolver(puzzle, alt_q)
alt.start()

waited = 0.0
interval = 0.5
while waited < args.wait:
    if hasattr(puzzle, 'alt_results'):
        break
    time.sleep(interval)
    waited += interval

if not hasattr(puzzle, 'alt_results'):
    print('Alternative solver did not finish in time')
    sys.exit(2)

# Collect per-piece pixel sets
piece_pixel_sets = []
for i, p in enumerate(puzzle.pieces_):
    pts = set((int(x), int(y)) for (x, y) in p.pixels.keys())
    piece_pixel_sets.append(pts)

# Check overlaps
n = len(piece_pixel_sets)
overlaps = {}
for i in range(n):
    for j in range(i+1, n):
        inter = piece_pixel_sets[i].intersection(piece_pixel_sets[j])
        if inter:
            overlaps[(i,j)] = len(inter)

print('pieces:', n)
print('total overlapping pairs:', len(overlaps))
for (i,j), cnt in overlaps.items():
    print(f'Pair {i}-{j}: overlapping pixels = {cnt}')

# exit code 0 if minimal overlaps (<=10 pixels per pair), else 3
max_overlap = max(overlaps.values()) if overlaps else 0
print('max_overlap_pixels:', max_overlap)
if max_overlap <= 10:
    sys.exit(0)
else:
    sys.exit(3)
