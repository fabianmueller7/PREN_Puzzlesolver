import sys
import os

# zum Projekt-Root navigieren (eine Ebene höher als /Robot)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import json
import numpy as np

from Puzzle.Extractor import Extractor


def compute_piece_pose_from_pixels(pixels):
    """
    Berechnet Schwerpunkt (cx, cy) und Orientierung (theta_deg)
    aus einer Liste von Pixelkoordinaten (x, y).
    """

    if not pixels:
        return None

    # Falls pixels ein Dict ist: nur die Keys verwenden
    if isinstance(pixels, dict):
        pts = np.array(list(pixels.keys()), dtype=np.float32)
    else:
        pts = np.array(pixels, dtype=np.float32)

    # Schwerpunkt
    cx = float(np.mean(pts[:, 0]))
    cy = float(np.mean(pts[:, 1]))

    # Orientierung über minAreaRect
    # Erwartet Form (N,1,2)
    contour = pts.reshape(-1, 1, 2)
    (rect_center, (w, h), angle) = cv2.minAreaRect(contour)

    # Winkel normalisieren (OpenCV gibt -90..0)
    if angle < -45:
        angle += 90.0

    return cx, cy, angle


def detect_piece_positions(
    image_path: str,
    output_json: str = "positions_a4.json",
    a4_width_mm: float = 297.0,
    a4_height_mm: float = 210.0,
    green_screen: bool = False,
):
    """
    Erkennt Puzzleteile auf der A4 Flaeche und exportiert:
      - Schwerpunkt in Pixel
      - Schwerpunkt in mm (relativ zur A4 Flaeche)
      - Orientierung in Grad

    Annahmen:
      - Kamera schaut moeglichst senkrecht auf A4.
      - A4 ist voll im Bild (oben/links = (0,0)).
    """

    # 1) Teile mit bestehendem Extractor holen
    extractor = Extractor(image_path, viewer=None, green_screen=green_screen)
    pieces = extractor.extract()

    if not pieces:
        raise RuntimeError("Keine Puzzleteile gefunden.")

    # 2) Bildgroesse fuer Skalierung Pixel -> mm
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Konnte Bild nicht laden: {image_path}")

    img_h, img_w = img.shape[:2]
    scale_x = a4_width_mm / float(img_w)
    scale_y = a4_height_mm / float(img_h)

    results = []

    # 3) Fuer jedes Piece Pose bestimmen
    for idx, piece in enumerate(pieces, start=1):
        pixels = getattr(piece, "pixels", None)
        if not pixels:
            continue

        pose = compute_piece_pose_from_pixels(pixels)
        if pose is None:
            continue

        cx_px, cy_px, theta_deg = pose

        results.append(
            {
                "id": idx,
                "x_px": round(cx_px, 2),
                "y_px": round(cy_px, 2),
                "x_mm": round(cx_px * scale_x, 2),
                "y_mm": round(cy_px * scale_y, 2),
                "theta_deg": round(theta_deg, 2),
            }
        )

    # 4) JSON schreiben
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"{len(results)} Teile erkannt → gespeichert in {output_json}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Erkennt Puzzleteil-Positionen (x, y, theta) auf A4-Flaeche."
    )
    parser.add_argument("image", help="Pfad zum A4-Bild mit Puzzleteilen")
    parser.add_argument(
        "--out",
        default="positions_a4.json",
        help="Pfad fuer JSON-Output (Default: positions_a4.json)",
    )
    parser.add_argument(
        "--green",
        action="store_true",
        help="Gruenscreen-Entfernung aktivieren, falls verwendet.",
    )
    args = parser.parse_args()

    detect_piece_positions(args.image, args.out, green_screen=args.green)
