#!/usr/bin/env python3
"""
Simple test script to extract pieces from an image without GUI.
"""
import os
import sys
import glob

# Setup temp dir
debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_output")
os.makedirs(debug_dir, exist_ok=True)

# Clean up old files
for f in glob.glob(os.path.join(debug_dir, "*")):
    try:
        if os.path.isdir(f):
            import shutil
            shutil.rmtree(f)
        else:
            os.remove(f)
    except:
        pass

os.environ["ZOLVER_TEMP_DIR"] = debug_dir
print(f"Debug output directory: {debug_dir}")

# Now import after env setup
from Puzzle.Extractor import Extractor

if len(sys.argv) < 2:
    print("Usage: python test_extraction.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]
print(f"\nTesting extraction on: {image_path}")

# Run extraction
extractor = Extractor(image_path, viewer=None, green_screen=False)
pieces = extractor.extract()

if pieces is None:
    print("\n❌ NO PIECES FOUND!")
    sys.exit(1)

print(f"\n✓ Successfully found {len(pieces)} pieces!")
for i, piece in enumerate(pieces):
    print(f"  Piece {i+1}: {piece}")
