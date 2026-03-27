import argparse
import glob
import multiprocessing as mp
import os

import matplotlib.pyplot as plt

from Puzzle.Puzzle import Puzzle

# Use persistent debug output directory
debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_output")
os.makedirs(debug_dir, exist_ok=True)
for f in glob.glob(os.path.join(debug_dir, "*")):
    os.remove(f)
os.environ["ZOLVER_TEMP_DIR"] = debug_dir
print(f"Debug output directory: {debug_dir}")

# Parse arguments
parser = argparse.ArgumentParser(description="Solve Puzzles!")
parser.add_argument(
    "-g", "--green_screen", help="enable green background removing", action="store_true"
)
parser.add_argument("-p", "--profile", help="enable profiling", action="store_true")
parser.add_argument("file", type=str, help="input_file")
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = fig_size[0] * 2
plt.rcParams["figure.figsize"] = fig_size

args = parser.parse_args()

mp.set_start_method("fork")

if args.profile:
    import cProfile
    import pstats
    import io
    from pstats import SortKey

    with cProfile.Profile() as pr:
        puzzle = Puzzle(args.file, green_screen=args.green_screen)
        puzzle.solve_puzzle()

        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats(SortKey.CUMULATIVE)
        ps.print_stats(50)
        print(s.getvalue())
else:
    Puzzle(args.file, green_screen=args.green_screen)
