#!/usr/bin/env python3
"""
Run only the alternative solver on a puzzle image.

Usage:
  python tools/run_alternative_only.py --image "resources/must works/Puzzle_Variante6_6Teile.png"
"""

import argparse
import json
import os
import queue
import sys
import tempfile
import time
import atexit
from datetime import datetime

# Make the package importable
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Puzzle.alternative_solver import AlternativeSolver
from Puzzle.Puzzle import Puzzle


def main():
    parser = argparse.ArgumentParser(description='Run alternative solver only on one image')
    parser.add_argument(
        '--image',
        required=False,
        default=os.path.join('resources', 'pren-samples', 'PREN-Samples_2.jpg'),
        help='Path to puzzle image'
    )
    parser.add_argument('--wait-seconds', type=float, default=60.0, help='Seconds to wait for solver to complete')
    parser.add_argument('--open', action='store_true', help='Open output images and JSON with the OS default application')
    args = parser.parse_args()

    img_path = os.path.abspath(args.image)
    if not os.path.exists(img_path):
        print('Image not found:', img_path)
        return

    # Create timestamped debug output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_dir = os.path.join(REPO_ROOT, "debug_output", f"alt_solver_{timestamp}")
    os.makedirs(debug_dir, exist_ok=True)
    os.environ['ZOLVER_TEMP_DIR'] = debug_dir

    print('Output folder:', debug_dir)
    print('Starting extraction on image:', img_path)

    # Extract pieces with Puzzle (no solving)
    puzzle = Puzzle(img_path, green_screen=False)
    
    # Provide a minimal viewer so the extractor will write images
    class DummyViewer:
        def addImage(self, *args, **kwargs):
            return
        def addLog(self, *args, **kwargs):
            return

    puzzle.viewer = DummyViewer()
    
    # Just extract pieces, don't solve
    print(f'\nExtracted {len(puzzle.pieces_)} pieces')

    print('\nStarting alternative solver...')
    
    # Start the alternative solver
    result_q = queue.Queue()
    alternative = AlternativeSolver(puzzle, result_q)
    alternative.start()

    # Wait for completion
    waited = 0.0
    interval = 0.5
    while waited < args.wait_seconds:
        if not alternative.is_alive():
            break
        time.sleep(interval)
        waited += interval

    alternative.join(timeout=5)

    # Check if alt_results was set
    if hasattr(puzzle, 'alt_results'):
        alt = puzzle.alt_results
        out_json = os.path.join(debug_dir, 'alt_results.json')
        try:
            with open(out_json, 'w', encoding='utf-8') as f:
                json.dump(alt, f, indent=2, default=str)
            print(f'\n✓ Alternative solver results written to: {out_json}')
        except Exception as e:
            print(f'Failed to write alt results: {e}')

        # Print summary
        if isinstance(alt, dict):
            print('\n' + '='*50)
            print('ALTERNATIVE SOLVER SUMMARY')
            print('='*50)
            for k in ['num_pieces', 'corner_candidates', 'possible_corners', 'pieces_placed']:
                if k in alt:
                    print(f'  {k}: {alt[k]}')
            if 'groups' in alt:
                print('  groups:')
                for g in alt['groups']:
                    print(f"    - count={g['count']}, avg_length={g['avg_length']:.1f}")
            print('='*50 + '\n')
        else:
            print('Result:', alt)
    else:
        print(f'\n✗ Solver did not complete within {args.wait_seconds} seconds')

    # Show debug output location
    print(f'Debug output: {debug_dir}')

    # Optionally open files
    if args.open and hasattr(puzzle, 'alt_results'):
        import subprocess
        try:
            # Open the debug directory
            if sys.platform == 'win32':
                os.startfile(debug_dir)
            elif sys.platform == 'darwin':
                subprocess.run(['open', debug_dir])
            else:
                subprocess.run(['xdg-open', debug_dir])
        except Exception as e:
            print(f'Could not open folder: {e}')


if __name__ == '__main__':
    main()
