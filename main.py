"""
Hauptprogramm (Einstiegspunkt) der Zolver-GUI.

Aufgaben:
- Erstellt ein temporäres Arbeitsverzeichnis und macht es über
  die Umgebungsvariable ZOLVER_TEMP_DIR für das Programm verfügbar.
- Sorgt dafür, dass dieses Verzeichnis beim Beenden automatisch gelöscht wird.
- Startet die PyQt5-Anwendung und zeigt das Hauptfenster (Viewer) an.
Moin
"""


import atexit       # Zum Registrieren einer Aufräumfunktion beim Programmende
import os           # Für den Zugriff auf Umgebungsvariablen
import sys          # Für Kommandozeilenargumente, die an Qt weitergegeben werden
import tempfile     # Zum Erstellen eines temporären Verzeichnisses

from PyQt5.QtWidgets import QApplication    # Hauptklasse der Qt-GUI-Anwendung
from GUI.Viewer import Viewer               # Import des Hauptfensters (eigene Klasse)


if __name__ == "__main__":
    # Create and use temporary directory
    temp_dir = tempfile.TemporaryDirectory()
    os.environ["ZOLVER_TEMP_DIR"] = temp_dir.name
    atexit.register(temp_dir.cleanup)

    # Display GUI and exit
    app = QApplication(sys.argv)
    imageViewer = Viewer()
    imageViewer.show()
    sys.exit(app.exec_())
