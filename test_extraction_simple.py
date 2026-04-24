#!/usr/bin/env python3
"""
Simple test script to check piece extraction stages.
"""
import os
import sys
import glob
import cv2
import numpy as np

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

# Import extraction module
import config

if len(sys.argv) < 2:
    print("Usage: python test_extraction_simple.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]
print(f"\nTesting extraction on: {image_path}\n")

# Test 1: Load and check image
img = cv2.imread(image_path, cv2.IMREAD_COLOR)
if img is None:
    print("❌ Could not load image!")
    sys.exit(1)

print(f"✓ Image loaded: {img.shape}")

# Test 2: Simulate simple_piece_threshold logic
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(f"✓ Converted to grayscale")

# Reflection suppression
img_gray = cv2.medianBlur(img_gray, 5)
img_gray = cv2.bilateralFilter(img_gray, d=9, sigmaColor=75, sigmaSpace=75)

# Clip top highlights
high = np.percentile(img_gray, 98)
img_gray = np.clip(img_gray, 0, high).astype(np.uint8)

# Gamma compression
img_gray = np.power(img_gray / 255.0, 0.8)
img_gray = (img_gray * 255).astype(np.uint8)

# Threshold
_, bw = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
print(f"✓ Applied Otsu threshold")

# Morphological smoothing
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel_close, iterations=2)
bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel_open, iterations=1)
print(f"✓ Applied morphological smoothing")

# Gaussian blur + rethreshold
bw = cv2.GaussianBlur(bw, (7, 7), 0)
_, bw = cv2.threshold(bw, 128, 255, cv2.THRESH_BINARY)
print(f"✓ Applied Gaussian blur + rethreshold")

# Find contours (raw)
contours_raw, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"\n✓ Found {len(contours_raw)} initial contours")

# Filter by area
contours_sorted = sorted(contours_raw, key=cv2.contourArea, reverse=True)
max_area = cv2.contourArea(contours_sorted[0]) if contours_sorted else 0
area_thr = max_area * 0.05

good_contours = [c for c in contours_raw if cv2.contourArea(c) > area_thr]
print(f"✓ After area filtering (>{area_thr:.0f}px): {len(good_contours)} contours")

# Fill and smooth
filled = np.zeros_like(bw)
cv2.drawContours(filled, good_contours, -1, 255, thickness=cv2.FILLED)
filled = cv2.GaussianBlur(filled, (5, 5), 0)
_, filled = cv2.threshold(filled, 128, 255, cv2.THRESH_BINARY)

# Find final contours
contours_final, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"✓ Final contours after smoothing: {len(contours_final)}")

# Save debug images
cv2.imwrite(os.path.join(debug_dir, "01_grayscale.png"), img_gray)
cv2.imwrite(os.path.join(debug_dir, "02_binary.png"), bw)
cv2.imwrite(os.path.join(debug_dir, "03_filled.png"), filled)

# Draw contours on image
debug_img = img.copy()
cv2.drawContours(debug_img, good_contours, -1, (0, 255, 0), 2)
cv2.imwrite(os.path.join(debug_dir, "04_contours.png"), debug_img)

print(f"\n{'='*50}")
print(f"RESULT: {len(good_contours)} PUZZLE PIECES DETECTED")
print(f"{'='*50}\n")

# Print summary
for i, cnt in enumerate(good_contours):
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    print(f"  Piece {i+1}: area={area:.0f}px, perimeter={perimeter:.0f}px")
