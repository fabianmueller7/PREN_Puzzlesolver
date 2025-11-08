import json
import math

# A5-Rahmengrösse in mm (Innenfläche)
A5_WIDTH_MM = 210.0
A5_HEIGHT_MM = 148.0


def compute_target_positions(input_json="positions_a4.json",
                             output_json="positions_a5.json",
                             rows=None,
                             cols=None):
    """
    Berechnet Zielkoordinaten der Puzzleteile im A5-Rahmen.

    Parameter:
        input_json  : Datei mit Ist-Koordinaten (positions_a4.json)
        output_json : Datei für Soll-Koordinaten (positions_a5.json)
        rows, cols  : Optional feste Anzahl Reihen/Spalten
                      (wird sonst automatisch geschätzt)
    """

    # 1) JSON-Daten einlesen
    with open(input_json, "r", encoding="utf-8") as f:
        pieces = json.load(f)

    n = len(pieces)
    if n == 0:
        raise ValueError("Keine Teile im JSON gefunden!")

    # 2) Raster automatisch bestimmen, falls nicht gesetzt
    if rows is None or cols is None:
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)

    # 3) Zellenbreite und -höhe berechnen
    cell_w = A5_WIDTH_MM / cols
    cell_h = A5_HEIGHT_MM / rows

    # 4) Zielkoordinaten berechnen
    targets = []
    for i, piece in enumerate(pieces):
        r = i // cols
        c = i % cols

        # Mittelpunkt jeder Rasterzelle (oben-links = 0,0)
        x_target = (c + 0.5) * cell_w
        y_target = (r + 0.5) * cell_h

        targets.append({
            "id": piece["id"],
            "x_target_mm": round(x_target, 2),
            "y_target_mm": round(y_target, 2),
            "theta_target_deg": 0.0
        })

    # 5) JSON-Datei speichern
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(targets, f, indent=2)

    print(f"{len(targets)} Zielkoordinaten → gespeichert in {output_json}")
    print(f"Raster: {rows} Zeilen × {cols} Spalten")
    return targets


if __name__ == "__main__":
    compute_target_positions()
