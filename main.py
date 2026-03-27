"""
Hauptprogramm (Einstiegspunkt) des Puzzlesolver-GUI.

Aufgaben:
- Erstellt ein temporäres Arbeitsverzeichnis und macht es über
  die Umgebungsvariable SOLVER_TEMP_DIR für das Programm verfügbar.
- Sorgt dafür, dass dieses Verzeichnis beim Beenden automatisch gelöscht wird.
- Startet die PyQt5-Anwendung und zeigt das Hauptfenster (Viewer) an.
"""


import glob
import os           # Für den Zugriff auf Umgebungsvariablen
import sys          # Für Kommandozeilenargumente, die an Qt weitergegeben werden

from PyQt5.QtWidgets import QApplication    # Hauptklasse der Qt-GUI-Anwendung
from GUI.Viewer import Viewer               # Import des Hauptfensters (eigene Klasse)


if __name__ == "__main__":
    debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_output")
    os.makedirs(debug_dir, exist_ok=True)
    for f in glob.glob(os.path.join(debug_dir, "*")):
        os.remove(f)
    os.environ["ZOLVER_TEMP_DIR"] = debug_dir
    print(f"Debug output directory: {debug_dir}")

    # Display GUI and exit
    app = QApplication(sys.argv)
    imageViewer = Viewer()
    imageViewer.show()
    sys.exit(app.exec_())
