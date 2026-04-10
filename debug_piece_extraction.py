"""
Debug script to track missing puzzle pieces in extraction
Usage: python debug_piece_extraction.py --image "resources\must works\Puzzle_Variante6_6Teile.png"
"""

import argparse
import os
import sys
import cv2
import numpy as np

# Make the package importable
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from Puzzle.Extractor import Extractor


def analyze_extraction(image_path):
    """Analyze the piece extraction process and identify missing pieces."""
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    print(f"🔍 Analyzing: {image_path}")
    print("=" * 70)
    
    # Create temp dir
    temp_dir = os.path.join(os.path.dirname(__file__), "debug_extraction")
    os.makedirs(temp_dir, exist_ok=True)
    os.environ["ZOLVER_TEMP_DIR"] = temp_dir
    
    # Read image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"❌ Could not read image: {image_path}")
        return
    
    print(f"✓ Image shape: {img.shape}")
    print()
    
    # Step 1: Run extractor preprocessing
    extractor = Extractor(image_path, green_screen=False, factor=0.84)
    
    # Step 2: Get the binary image before contour extraction
    bw = extractor.img_bw
    print(f"✓ Binary image created: {bw.shape}")
    
    # Step 3: Find ALL contours before filtering
    contours_all, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print(f"✓ Found {len(contours_all)} contours (before any filtering)")
    
    # Step 4: Analyze areas
    areas = [cv2.contourArea(c) for c in contours_all]
    areas_sorted = sorted(enumerate(areas), key=lambda x: x[1], reverse=True)
    
    print("\n📊 CONTOUR AREAS (sorted by size):")
    print("-" * 70)
    print(f"{'#':<3} {'Index':<6} {'Area':<15} {'% of Max':<12} {'Status':<15}")
    print("-" * 70)
    
    max_area = max(areas) if areas else 1
    threshold = max_area * 0.33  # This is the filtering threshold
    
    for rank, (idx, area) in enumerate(areas_sorted, 1):
        percent = (area / max_area * 100) if max_area > 0 else 0
        status = "✓ KEPT" if area >= threshold else "❌ FILTERED OUT"
        print(f"{rank:<3} {idx:<6} {area:<15.0f} {percent:<11.1f}% {status:<15}")
    
    print("-" * 70)
    print(f"Filtering threshold: {threshold:.0f} (min_ratio=0.33 × max_area)")
    print()
    
    # Step 5: Apply filtering like extractor does
    filtered_contours = extractor._filter_contours_by_area(contours_all, min_ratio=0.33)
    print(f"📉 After filtering: {len(filtered_contours)} contours remain")
    print(f"📍 Pieces filtered out: {len(contours_all) - len(filtered_contours)}")
    
    if len(contours_all) != len(filtered_contours):
        print("\n⚠️  PIECES FILTERED OUT:")
        for idx, area in areas_sorted:
            if area < threshold:
                percent = (area / max_area * 100) if max_area > 0 else 0
                print(f"   - Contour #{idx}: {area:.0f} px² ({percent:.1f}% of max)")
    
    # Step 6: Try extraction
    print("\n🔄 Running full extraction...")
    pieces = extractor.extract()
    
    if pieces:
        print(f"✓ Successfully extracted {len(pieces)} pieces")
    else:
        print("❌ Extraction returned None")
    
    # Step 7: Create visualization
    print("\n🎨 Creating debug visualizations...")
    
    # Draw all contours
    debug_all = np.zeros_like(bw)
    cv2.drawContours(debug_all, contours_all, -1, 255, 2)
    cv2.imwrite(os.path.join(temp_dir, "all_contours.png"), debug_all)
    
    # Draw filtered contours
    debug_filtered = np.zeros_like(bw)
    cv2.drawContours(debug_filtered, filtered_contours, -1, 255, 2)
    cv2.imwrite(os.path.join(temp_dir, "filtered_contours.png"), debug_filtered)
    
    # Draw contours with labels and areas
    debug_labeled = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    for idx, (cont_idx, area) in enumerate(areas_sorted, 1):
        c = contours_all[cont_idx]
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Draw contour
            color = (0, 255, 0) if area >= threshold else (0, 0, 255)  # Green if kept, Red if filtered
            cv2.drawContours(debug_labeled, [c], 0, color, 2)
            
            # Add label with area
            cv2.putText(debug_labeled, f"#{idx} {area:.0f}", (cx-30, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    cv2.imwrite(os.path.join(temp_dir, "labeled_contours.png"), debug_labeled)
    
    print(f"\n✓ Debug images saved to: {temp_dir}/")
    print("   - all_contours.png (all detected contours)")
    print("   - filtered_contours.png (after size filtering)")
    print("   - labeled_contours.png (with areas and indices)")
    print()
    print(f"📝 Summary:")
    print(f"   Expected: 6 pieces")
    print(f"   Detected: {len(contours_all)} contours")
    print(f"   Extracted: {len(pieces) if pieces else 0} pieces")
    print(f"   Missing: {6 - (len(pieces) if pieces else 0)} piece(s)")
    
    if len(contours_all) > len(filtered_contours):
        missing_count = len(contours_all) - len(filtered_contours)
        print(f"\n🔴 ROOT CAUSE: {missing_count} piece(s) filtered out due to small size!")
        print("   Recommendation: Lower min_ratio parameter or check image quality")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug piece extraction")
    parser.add_argument("--image", required=True, help="Path to puzzle image")
    args = parser.parse_args()
    
    analyze_extraction(args.image)
