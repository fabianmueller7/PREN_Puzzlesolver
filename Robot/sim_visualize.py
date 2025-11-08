import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# A5-Rahmengrösse in mm
A5_WIDTH_MM = 210.0
A5_HEIGHT_MM = 148.0


def visualize_positions(input_json="positions_a5.json",
                        show_ids=True,
                        save_path=None):
    """
    Zeigt den A5-Rahmen und die Zielkoordinaten der Puzzleteile an.

    Parameter:
        input_json : Datei mit Zielkoordinaten (positions_a5.json)
        show_ids   : Wenn True, werden die IDs der Teile angezeigt
        save_path  : Optionaler Pfad zum Speichern als PNG
    """

    # 1) Daten laden
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2) Plot vorbereiten
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.set_xlim(0, A5_WIDTH_MM)
    ax.set_ylim(A5_HEIGHT_MM, 0)  # y-Achse oben->unten
    ax.set_aspect("equal")
    ax.set_title("A5 Puzzle-Zielpositionen")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")

    # 3) Rahmen zeichnen
    rect = patches.Rectangle(
        (0, 0),
        A5_WIDTH_MM,
        A5_HEIGHT_MM,
        linewidth=2,
        edgecolor="navy",
        facecolor="lightgray",
        alpha=0.3
    )
    ax.add_patch(rect)

    # 4) Punkte einzeichnen
    for piece in data:
        x = piece["x_target_mm"]
        y = piece["y_target_mm"]
        ax.plot(x, y, "ro", markersize=8)
        if show_ids:
            ax.text(x + 3, y - 3, f"ID {piece['id']}",
                    color="black", fontsize=9, weight="bold")

    # 5) Plot anzeigen oder speichern
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Visualisierung gespeichert als: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    visualize_positions()
