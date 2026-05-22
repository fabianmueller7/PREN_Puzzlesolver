"""Generate a printable sheet of the 4 ArUco corner markers.

Layout (2×2 grid):
  [ID 0 — Top Left]    [ID 1 — Top Right]
  [ID 3 — Bot Left]    [ID 2 — Bot Right]

Output: resources/aruco_tags.png
At 300 DPI each 200 px tag ≈ 17 mm. Scale the PNG when printing to the desired size.
"""
import os
import sys

import cv2
import cv2.aruco as aruco
import numpy as np

TAG_SIZE_PX = 200
MARGIN = 20
LABEL_H = 28
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.45
FONT_THICKNESS = 1

LABELS = {
    0: "ID 0 - Top Left",
    1: "ID 1 - Top Right",
    2: "ID 2 - Bot Right",
    3: "ID 3 - Bot Left",
}

# Layout order: [TL, TR, BL, BR] → row 0: IDs 0,1  row 1: IDs 3,2
GRID = [[0, 1], [3, 2]]

dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

cell_w = TAG_SIZE_PX + 2 * MARGIN
cell_h = TAG_SIZE_PX + 2 * MARGIN + LABEL_H
sheet_h = cell_h * 2
sheet_w = cell_w * 2

sheet = np.zeros((sheet_h, sheet_w), dtype=np.uint8)  # black background for inverted tags

for row_idx, row in enumerate(GRID):
    for col_idx, tag_id in enumerate(row):
        tag_img = np.zeros((TAG_SIZE_PX, TAG_SIZE_PX), dtype=np.uint8)
        aruco.generateImageMarker(dictionary, tag_id, TAG_SIZE_PX, tag_img)
        tag_img = cv2.bitwise_not(tag_img)  # inverted: white modules on black background

        x0 = col_idx * cell_w + MARGIN
        y0 = row_idx * cell_h + MARGIN
        sheet[y0:y0 + TAG_SIZE_PX, x0:x0 + TAG_SIZE_PX] = tag_img

        label = LABELS[tag_id]
        text_x = col_idx * cell_w + MARGIN
        text_y = row_idx * cell_h + MARGIN + TAG_SIZE_PX + LABEL_H - 8
        cv2.putText(sheet, label, (text_x, text_y),
                    FONT, FONT_SCALE, 255, FONT_THICKNESS, cv2.LINE_AA)

out_path = os.path.join(os.path.dirname(__file__), "..", "resources", "aruco_tags.png")
out_path = os.path.normpath(out_path)
cv2.imwrite(out_path, sheet)
print(f"Saved: {out_path}  ({sheet_w}×{sheet_h} px)")
print("Tip: at 300 DPI, 200 px ≈ 17 mm per tag. Scale when printing.")
