"""
Deep debug for piece #1 extraction failure
Shows angle curves, corner detection, and edge classification
"""

import argparse
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from Puzzle.Extractor import Extractor
from Img.filters import my_find_corner_signature


def debug_piece_extraction_detailed(image_path):
    """Detailed analysis of why piece #1 fails to extract."""
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    print(f"🔍 Deep Analysis: {image_path}")
    print("=" * 80)
    
    temp_dir = os.path.join(os.path.dirname(__file__), "debug_extraction_deep")
    os.makedirs(temp_dir, exist_ok=True)
    os.environ["ZOLVER_TEMP_DIR"] = temp_dir
    
    # Extract
    extractor = Extractor(image_path, green_screen=False, factor=0.84)
    bw = extractor.img_bw
    img = extractor.img
    
    # Find contours from simple_piece_threshold
    print("\n[STEP 1] Finding contours...")
    contours_all, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print(f"✓ Found {len(contours_all)} contours")
    
    # Try corner signature detection on each piece
    print("\n[STEP 2] Analyzing corner/edge detection for each piece...")
    print("-" * 80)
    
    for piece_idx in range(len(contours_all)):
        cnt = contours_all[piece_idx]
        print(f"\n📍 PIECE #{piece_idx}:")
        print(f"   Contour length: {len(cnt)} points")
        
        # Get signature
        corners, edges_shape, types_edges = my_find_corner_signature(cnt, green=False)
        
        if corners is None:
            print(f"   ❌ FAILED: Corner detection returned None")
            print(f"   ⚠️  This piece will be SKIPPED during extraction")
            
            # Try to understand why
            print(f"\n   Analyzing why corner detection failed...")
            
            # Check contour properties
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = (hull_area - area) / hull_area if hull_area > 0 else 0
            
            print(f"   - Contour area: {area:.0f} px²")
            print(f"   - Perimeter: {perimeter:.1f} px")
            print(f"   - Hull area: {hull_area:.0f} px²")
            print(f"   - Solidity (inverse gap): {solidity:.3f}")
            print(f"   - Roughness: {(perimeter**2 / (4 * np.pi * area)):.3f}")
            
            if piece_idx == 1:
                print(f"\n   🎯 THIS IS THE MISSING PIECE!")
                print(f"   Saving detailed visualization...")
                
                # Draw the problematic contour
                debug_img = np.zeros_like(bw)
                cv2.drawContours(debug_img, [cnt], 0, 255, 2)
                cv2.imwrite(os.path.join(temp_dir, "piece_1_contour.png"), debug_img)
                
                # Draw on original
                debug_img_orig = img.copy()
                cv2.drawContours(debug_img_orig, [cnt], 0, (0, 0, 255), 3)  # Red
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(debug_img_orig, "PIECE #1 (MISSING)", (cx-50, cy-20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imwrite(os.path.join(temp_dir, "piece_1_on_original.png"), debug_img_orig)
                
        else:
            print(f"   ✓ SUCCESS: Corner detection succeeded")
            print(f"   - Corners found at indices: {corners}")
            print(f"   - Edge types: {[str(t) for t in types_edges]}")
            
            # Check if any UNDEFINED
            undefined_count = sum(1 for t in types_edges if "UNDEFINED" in str(t))
            if undefined_count > 0:
                print(f"   ⚠️  WARNING: {undefined_count} UNDEFINED edge(s) detected!")
    
    print("\n" + "=" * 80)
    print("📂 Debug files saved to:", temp_dir)
    print("\nKey files:")
    print("  - piece_1_contour.png: The problematic contour in isolation")
    print("  - piece_1_on_original.png: The piece highlighted on original image")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep debug of piece extraction")
    parser.add_argument("--image", required=True, help="Path to puzzle image")
    args = parser.parse_args()
    
    debug_piece_extraction_detailed(args.image)
