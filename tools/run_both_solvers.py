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
import sys
import tempfile
import time
import atexit

# Make the package importable when running this script from the repository root.
# Ensure the PREN_Puzzlesolver package folder (the parent of this tools/ folder)
# is on sys.path so `from Puzzle.Puzzle import Puzzle` works regardless of cwd.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Puzzle.Puzzle import Puzzle


def main():
    parser = argparse.ArgumentParser(description='Run main and alternative solver on one image')
    parser.add_argument('--image', required=False, default=os.path.join('resources', 'jigsaw-samples', 'Test-erweitert.png'), help='Path to puzzle image (relative to PREN_Puzzlesolver root)')
    parser.add_argument('--wait-alt-seconds', type=float, default=30.0, help='Seconds to wait for alternative solver results after main solver completes')
    parser.add_argument('--open', action='store_true', help='Open output images and JSON with the OS default application')
    args = parser.parse_args()

    img_path = os.path.abspath(args.image)
    if not os.path.exists(img_path):
        print('Image not found:', img_path)
        return

    # Prepare temporary directory and env var used by the solver
    temp_dir = tempfile.TemporaryDirectory()
    os.environ['ZOLVER_TEMP_DIR'] = temp_dir.name
    atexit.register(temp_dir.cleanup)

    print('Temporary folder:', temp_dir.name)
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

    # Wait for alt_results to appear (poll)
    waited = 0.0
    interval = 0.5
    while waited < args.wait_alt_seconds:
        if hasattr(puzzle, 'alt_results'):
            break
        time.sleep(interval)
        waited += interval

    if hasattr(puzzle, 'alt_results'):
        alt = puzzle.alt_results
        out_json = os.path.join(temp_dir.name, 'alt_results.json')
        try:
            with open(out_json, 'w', encoding='utf-8') as f:
                json.dump(alt, f, indent=2, default=str)
            print('Alternative solver results written to', out_json)
        except Exception as e:
            print('Failed to write alt results:', e)
        # Print brief summary
        if isinstance(alt, dict):
            print('\nAlternative solver summary:')
            for k in ['num_pieces', 'corner_candidates', 'possible_corners']:
                if k in alt:
                    print(f' - {k}:', alt[k])
            if 'groups' in alt:
                print(' - groups (count, avg_length):')
                for g in alt['groups']:
                    print(f"    count={g['count']}, avg_length={g['avg_length']:.1f}")
        else:
            print('Alternative solver result (non-dict):', alt)
    else:
        print(f'No alt_results after {args.wait_alt_seconds} seconds.')

    # Look for final solver output images in temp dir
    stick_path = os.path.join(temp_dir.name, 'stick.png')
    colored_path = os.path.join(temp_dir.name, 'colored.png')
    print('\nMain solver exported images (if available):')
    for p in (stick_path, colored_path):
        if os.path.exists(p):
            print(' -', p)
            if args.open:
                try:
                    os.startfile(p)
                except Exception:
                    pass
        else:
            print(' - missing:', p)

    if hasattr(puzzle, 'alt_results') and args.open:
        try:
            os.startfile(out_json)
        except Exception:
            pass


if __name__ == '__main__':
    main()
