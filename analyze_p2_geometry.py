"""
Analyze the specific geometry of piece P2 that causes corner detection to fail
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
from Img.filters import my_find_corner_signature, get_relative_angles
from Img.peak_detect import detect_peaks


def analyze_piece_p2_geometry(image_path):
    """Detailed geometric analysis of why P2 fails corner detection."""
    
    print(f"🔍 Analyzing P2 Geometry: {image_path}")
    print("=" * 80)
    
    temp_dir = os.path.join(os.path.dirname(__file__), "debug_p2_geometry")
    os.makedirs(temp_dir, exist_ok=True)
    os.environ["ZOLVER_TEMP_DIR"] = temp_dir
    
    # Extract
    extractor = Extractor(image_path, green_screen=False, factor=0.84)
    bw = extractor.img_bw
    img = extractor.img
    
    # Get the 6 contours from simple_piece_threshold
    # We need to rerun it to get the contours
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    high = np.percentile(gray, 98)
    gray = np.clip(gray, 0, high).astype(np.uint8)
    gray = np.power(gray / 255.0, 0.8)
    gray = (gray * 255).astype(np.uint8)
    
    _, bw_test = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bw_test = cv2.morphologyEx(bw_test, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    bw_test = cv2.morphologyEx(bw_test, cv2.MORPH_OPEN, kernel_open, iterations=1)
    bw_test = cv2.GaussianBlur(bw_test, (5, 5), 0)
    _, bw_test = cv2.threshold(bw_test, 128, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(bw_test, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    print(f"✓ Found {len(contours)} contours using simple_piece_threshold")
    
    # All pieces sorted by area
    areas = [cv2.contourArea(c) for c in contours]
    for i, (cnt, area) in enumerate(zip(contours, areas)):
        print(f"  Piece {i}: area={area:.0f}")
    
    # Analyze piece P2 (which one is it in the list?)
    # Based on the corners_vis.png output, P2 seems to be around the bottom
    # Let's analyze each one to find which has corner detection issues
    
    print("\n[ANALYSIS] Testing corner detection on each piece:")
    print("-" * 80)
    
    for piece_idx, cnt in enumerate(contours):
        print(f"\n📍 Contour #{piece_idx} (area={cv2.contourArea(cnt):.0f}):")
        
        corners, edges_shape, types_edges = my_find_corner_signature(cnt, green=False)
        
        if corners is None:
            print(f"   ❌ FAILED - This is the missing piece!")
            
            # Analyze its geometry
            cnt_convert = [c[0] for c in cnt]
            
            # Check contour properties
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            
            # Get angles
            relative_angles = get_relative_angles(np.array(cnt_convert), export=False, sigma=5)
            relative_angles = np.array(relative_angles)
            
            print(f"\n   📊 Geometric Properties:")
            print(f"   - Area: {area:.0f} px²")
            print(f"   - Perimeter: {perimeter:.1f} px")
            print(f"   - Contour points: {len(cnt)}")
            
            # Analyze angle peaks
            print(f"\n   🔍 Angle Analysis (sigma=5):")
            extr = detect_peaks(relative_angles, mph=0.3 * np.max(relative_angles))
            print(f"   - Max angle: {np.max(relative_angles):.4f}")
            print(f"   - Min angle: {np.min(relative_angles):.4f}")
            print(f"   - Detected peaks: {len(extr)}")
            print(f"   - Expected peaks: 4 (for 4 corners)")
            
            if len(extr) != 4:
                print(f"   ⚠️  Peak detection found {len(extr)} peaks instead of 4!")
                print(f"       This is likely why corner detection failed.")
                
                # Visualize
                fig, axes = plt.subplots(2, 1, figsize=(12, 8))
                
                axes[0].plot(relative_angles)
                axes[0].plot(extr, relative_angles[extr], "ro", label="Detected peaks")
                axes[0].axhline(y=0.3 * np.max(relative_angles), color='r', linestyle='--', alpha=0.3)
                axes[0].set_title(f"Relative Angles for Piece #{piece_idx} (Missing Piece)")
                axes[0].set_ylabel("Angle (radians)")
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                
                # Draw contour
                ix = axes[1]
                debug_img = np.zeros((cnt.max(axis=0)[0][0] + 100, cnt.max(axis=0)[0][1] + 100, 3), dtype=np.uint8)
                cv2.drawContours(debug_img, [cnt], 0, (255, 255, 255), 2)
                
                # Mark peaks
                for peak_idx in extr:
                    pt = cnt[peak_idx % len(cnt)][0]
                    cv2.circle(debug_img, tuple(pt.astype(int)), 10, (0, 255, 0), -1)
                
                axes[1].imshow(debug_img)
                axes[1].set_title(f"Contour #{piece_idx} with {len(extr)} Detected Peaks")
                axes[1].axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(temp_dir, "missing_piece_analysis.png"), dpi=150)
                plt.close()
                
                print(f"\n   ✓ Visualization saved: missing_piece_analysis.png")
        else:
            print(f"   ✓ SUCCESS - Corners detected at indices: {corners}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze P2 geometry")
    parser.add_argument("--image", required=True, help="Path to puzzle image")
    args = parser.parse_args()
    
    analyze_piece_p2_geometry(args.image)
