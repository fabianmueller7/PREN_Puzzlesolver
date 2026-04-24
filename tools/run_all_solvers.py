#!/usr/bin/env python3
"""
Run all three solvers (Main, Alternative, and LEGO) on a puzzle image.

Usage:
  python tools/run_all_solvers.py --image "resources/pren-samples/PREN-Samples_2.jpg"
  python tools/run_all_solvers.py --image "resources/pren-samples/PREN-Samples_2.jpg" --open

This script:
1. Extracts pieces from the image
2. Starts the Main Solver (primary algorithm with BORDER + FILL strategies)
3. Starts the Alternative Solver (grid-based with heuristics)
4. Starts the LEGO Solver (spline-based edge comparison)
5. Waits for all solvers to complete
6. Compares and displays results from all three approaches
"""

import argparse
import json
import os
import sys
import time
import subprocess
from datetime import datetime

# Make the package importable
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Puzzle.Puzzle import Puzzle


def main():
    parser = argparse.ArgumentParser(description='Run all three solvers on one image')
    parser.add_argument(
        '--image',
        required=False,
        default=os.path.join('resources', 'pren-samples', 'PREN-Samples_2.jpg'),
        help='Path to puzzle image'
    )
    parser.add_argument('--wait-alt-seconds', type=float, default=60.0, 
                       help='Seconds to wait for alternative/LEGO solver results')
    parser.add_argument('--open', action='store_true', 
                       help='Open output folder with the OS default application')
    args = parser.parse_args()

    img_path = os.path.abspath(args.image)
    if not os.path.exists(img_path):
        print('❌ Image not found:', img_path)
        print('Please provide a valid image path using the --image argument.')
        return

    # Create timestamped debug output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_base = os.path.join(REPO_ROOT, 'debug_output')
    debug_dir = os.path.join(debug_base, f'all_solvers_{timestamp}')
    os.makedirs(debug_dir, exist_ok=True)
    os.environ['ZOLVER_TEMP_DIR'] = debug_dir

    print('=' * 70)
    print('PREN_Puzzlesolver - ALL THREE SOLVERS TEST')
    print('=' * 70)
    print(f'Image: {img_path}')
    print(f'Debug output: {debug_dir}')
    print()

    # Create minimal viewer
    class DummyViewer:
        def addImage(self, *args, **kwargs):
            return
        def addLog(self, *args, **kwargs):
            return

    # Extract and solve
    print('[1/3] Creating Puzzle object and extracting pieces...')
    puzzle = Puzzle(img_path, viewer=DummyViewer(), green_screen=False)
    print(f'✓ Extracted {len(puzzle.pieces_)} pieces')
    print()

    print('[2/3] Starting all three solvers...')
    print('  - Main Solver (BORDER + FILL strategy)')
    print('  - Alternative Solver (grid-based heuristics)')
    print('  - LEGO Solver (spline-based comparison)')
    print()
    
    puzzle.viewer = DummyViewer()
    start_time = time.time()
    puzzle.solve_puzzle()
    elapsed = time.time() - start_time
    
    print(f'Main Solver finished in {elapsed:.2f} seconds')
    print()

    # Wait for background solvers
    print('[3/3] Waiting for background solvers to complete...')
    wait_start = time.time()
    waited = 0.0
    interval = 1.0
    
    alt_done = False
    lego_done = False
    
    while waited < args.wait_alt_seconds:
        if hasattr(puzzle, 'alt_results') and not alt_done:
            print(f'✓ Alternative Solver results received ({waited:.1f}s)')
            alt_done = True
        
        if hasattr(puzzle, 'lego_results') and not lego_done:
            print(f'✓ LEGO Solver results received ({waited:.1f}s)')
            lego_done = True
        
        if alt_done and lego_done:
            break
        
        time.sleep(interval)
        waited = time.time() - wait_start

    print()
    print('=' * 70)
    print('RESULTS SUMMARY')
    print('=' * 70)
    print()

    # Main solver results
    print('📊 MAIN SOLVER:')
    print(f'  Pieces placed: {len(puzzle.connected_directions)}')
    print(f'  Success: {"YES ✓" if len(puzzle.connected_directions) == len(puzzle.pieces_) else "NO ✗"}')
    print()

    # Alternative solver results
    if hasattr(puzzle, 'alt_results'):
        alt = puzzle.alt_results
        if isinstance(alt, dict):
            print('📊 ALTERNATIVE SOLVER:')
            print(f'  Pieces placed: {alt.get("pieces_placed", "?")}')
            print(f'  Success: {"YES ✓" if alt.get("success", False) else "NO ✗"}')
            if 'error' in alt:
                print(f'  Error: {alt["error"]}')
            print()
    else:
        print('📊 ALTERNATIVE SOLVER: No results yet')
        print()

    # LEGO solver results
    if hasattr(puzzle, 'lego_results'):
        lego = puzzle.lego_results
        if isinstance(lego, dict):
            print('📊 LEGO SOLVER:')
            print(f'  Pieces placed: {lego.get("pieces_placed", "?")} / {lego.get("total_pieces", "?")}')
            print(f'  Success: {"YES ✓" if lego.get("success", False) else "NO ✗"}')
            if 'error' in lego:
                print(f'  Error: {lego["error"]}')
            print()
    else:
        print('📊 LEGO SOLVER: No results yet')
        print()

    # Save combined results
    combined_results = {
        'timestamp': timestamp,
        'image': img_path,
        'total_pieces': len(puzzle.pieces_),
        'main_solver': {
            'pieces_placed': len(puzzle.connected_directions),
            'success': len(puzzle.connected_directions) == len(puzzle.pieces_),
        },
        'alternative_solver': puzzle.alt_results if hasattr(puzzle, 'alt_results') else None,
        'lego_solver': puzzle.lego_results if hasattr(puzzle, 'lego_results') else None,
    }
    
    results_file = os.path.join(debug_dir, 'all_solvers_results.json')
    with open(results_file, 'w') as f:
        json.dump(combined_results, f, indent=2, default=str)
    
    print(f'📁 Results saved to: {results_file}')
    print()

    # Open folder if requested
    if args.open:
        try:
            if sys.platform == 'win32':
                os.startfile(debug_dir)
            elif sys.platform == 'darwin':
                subprocess.run(['open', debug_dir])
            else:
                subprocess.run(['xdg-open', debug_dir])
            print(f'📂 Opened folder: {debug_dir}')
        except Exception as e:
            print(f'Could not open folder: {e}')

    print('=' * 70)


if __name__ == '__main__':
    main()
