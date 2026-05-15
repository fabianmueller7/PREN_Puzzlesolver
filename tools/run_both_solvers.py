"""
Run the original PREN_Puzzlesolver and the concurrent alternative solver
on a given image, then display/save their outputs.

Usage:
  python tools/run_both_solvers.py --image resources/jigsaw-samples/Test-erweitert.png

This script creates a temporary working directory (used by the solver),
invokes the Puzzle solver (which starts the alternative solver in background),
waits for both to finish (polling for `puzzle.alt_results`), and then
prints and optionally opens the generated images and alt-results JSON.
"""

import argparse
import json
import os
import queue
import sys
import time
import subprocess

# Make the package importable when running this script from the repository root.
# Ensure the PREN_Puzzlesolver package folder (the parent of this tools/ folder)
# is on sys.path so `from Puzzle.Puzzle import Puzzle` works regardless of cwd.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Puzzle.alternative_solver import AlternativeSolver
from Puzzle.Puzzle import Puzzle


def main():
    parser = argparse.ArgumentParser(description='Run main and alternative solver on one image')
    parser.add_argument(
        '--image',
        required=False,
        default=os.path.join('PREN_Puzzlesolver', 'resources', 'pren-samples', 'PREN-Samples_2.jpg'),
        help='Path to puzzle image (relative to PREN_Puzzlesolver root). Note: The default image path may not exist in all setups.'
    )
    parser.add_argument('--wait-alt-seconds', type=float, default=30.0, help='Seconds to wait for alternative solver results after main solver completes')
    parser.add_argument('--open', action='store_true', help='Open output folder with the OS default application')
    args = parser.parse_args()

    img_path = os.path.abspath(args.image)
    if not os.path.exists(img_path):
        print('Image not found:', img_path)
        print('Please provide a valid image path using the --image argument.')
        print("Example: --image resources/jigsaw-samples/Test-erweitert.png")
        return

    # Create timestamped debug output directory (same as alternative solver)
    debug_base = os.path.join(REPO_ROOT, 'debug_output')
    os.makedirs(debug_base, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    debug_dir = os.path.join(debug_base, f'alt_solver_{timestamp}')
    os.makedirs(debug_dir, exist_ok=True)
    
    os.environ['ZOLVER_TEMP_DIR'] = debug_dir

    print('Debug output folder:', debug_dir)
    print('Starting Puzzle on image:', img_path)

    # Construct puzzle and run solver (this will run the alternative solver in background)
    puzzle = Puzzle(img_path, green_screen=False)
    # Provide a minimal viewer so the solver's export_pieces() will write images.
    class DummyViewer:
        def addImage(self, *args, **kwargs):
            return

        def addLog(self, *args, **kwargs):
            return

    puzzle.viewer = DummyViewer()
    puzzle.solve_puzzle()

    print('Main solver finished. Checking for alternative solver results...')
    
    # Start the alternative solver.
    alt_result_queue = queue.Queue()
    alternative = AlternativeSolver(puzzle, alt_result_queue)
    alternative.start()  # Start the alternative solver thread (if not already started by Puzzle)

    # Wait for alt_results to appear or for the alternative solver to return a result.
    waited = 0.0
    interval = 0.5
    alt = None
    while waited < args.wait_alt_seconds:
        if hasattr(puzzle, 'alt_results'):
            alt = puzzle.alt_results
            break
        try:
            alt = alt_result_queue.get(timeout=interval)
            break
        except queue.Empty:
            pass
        waited += interval

    if alt is None and hasattr(puzzle, 'alt_results'):
        alt = puzzle.alt_results

    if alt is None:
        # Try one final join if the thread is still alive
        if alternative.is_alive():
            alternative.join(timeout=2.0)
        if hasattr(puzzle, 'alt_results'):
            alt = puzzle.alt_results

    if alt is not None:
        out_json = os.path.join(debug_dir, 'alt_results.json')
        try:
            with open(out_json, 'w', encoding='utf-8') as f:
                json.dump(alt, f, indent=2, default=str)
            print('Alternative solver results written to', out_json)
        except Exception as e:
            print('Failed to write alt results:', e)
        # Print brief summary
        if isinstance(alt, dict):
            print('\nAlternative solver summary:')
            for k in ['total_pieces', 'pieces_placed', 'success']:
                if k in alt:
                    print(f' - {k}:', alt[k])
        else:
            print('Alternative solver result (non-dict):', alt)
    else:
        print(f'No alt_results after {args.wait_alt_seconds} seconds.')

    # Optionally open debug output folder if requested
    if args.open:
        try:
            if sys.platform.startswith('win'):
                os.startfile(debug_dir)
            elif sys.platform.startswith('darwin'):
                subprocess.run(['open', debug_dir])
            else:
                subprocess.run(['xdg-open', debug_dir])
        except Exception as e:
            print(f'Could not open folder: {e}')


if __name__ == '__main__':
    main()
