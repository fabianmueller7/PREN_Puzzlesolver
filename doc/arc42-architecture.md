# arc42 Softwarearchitekturbeschreibung: PREN_Puzzlesolver

**Version:** 1.0  
**Datum:** November 2025  
**Autor:** Gruppe 06  
**Status:** Entwurf 

---

## 1. Einführung und Ziele

### 1.1 Aufgabenstellung

Der PREN_Puzzlesolver ist ein umfassendes Softwaresystem zur **automatisierten Lösung von physischen Jigsaw-Puzzles**. Das System kombiniert:

- **Bildverarbeitung:** Erfassung und Verarbeitung von Puzzle-Scans
- **Geometrische Analyse:** Erkennung von Puzzleteilen und deren Kanten
- **Intelligente Algorithmen:** Zwei Lösungsstrategien (Haupt-Algorithmus und Fallback-Strategie)
- **Benutzeroberfläche:** Qt5-basiertes GUI für Visualisierung und Interaktion
- **Hardware-Integration:** Raspberry-Pi als Maincontroller, Delegation an Komponenten

### 1.2 Wesentliche Qualitätsziele

| Priorität | Ziel               | Begründung                                                         |
| --------- | ------------------ |--------------------------------------------------------------------|
| 1         | **Korrektheit**    | Puzzle muss korrekt gelöst werden (100% erfolgreiche Montage)      |
| 2         | **Robustheit**     | Muss mit verschiedenen Puzzle-Typen und Beleuchtungen funktionieren |
| 3         | **Performance**    | Lösen von 4-6er-Puzzles in < 1 Minute                              |
| 4         | **Wartbarkeit**    | Modulare Architektur für einfache Erweiterung                      |
| 5         | **Interaktivität** | Responsive GUI, echtzeitgebenete Vorschaubilder                    |

### 1.3 Stakeholder

- **Entwickler:** Studenten an der Hochschule Luzern (PREN-Projekt)
- **Benutzer:** Personen, die Puzzles automatisch lösen möchten
- **Hardware-Operator:** 


---

## 2. Architekturübersicht

### 2.1 Kontextabgrenzung

```
┌─────────────────────────────────────────────────────┐
│              Externe Systeme                        │
├─────────────────────────────────────────────────────┤
│  • PI-Kamera (IP-Webcam oder USB)                   │
│  • Raspberry-Pi (Maincontroller                     │
│  • Monitor (GUI-Anzeige)                            │
└─────────────────────────────────────────────────────┘
                        │
                        ↓
        ┌───────────────────────────────┐
        │   PREN_Puzzlesolver           │
        │   (Dieses System)             │
        └───────────────────────────────┘
```

### 2.2 Top-Level-Struktur

```
PREN_Puzzlesolver/
├── Puzzle/                          # Kernlogik
│   ├── Puzzle.py                    # Hauptklasse & Orchestrierung
│   ├── PuzzlePiece.py              # Datenstruktur für ein Puzzleteil
│   ├── Edge.py                     # Datenstruktur für Kanten
│   ├── Enums.py                    # Enumerationen (Richtungen, Typen)
│   ├── Distance.py                 # Kantenabstands-Berechnung
│   ├── Mover.py                    # Teile zusammenstecken
│   ├── Extractor.py                # Teile aus Bild extrahieren
│   ├── alternative_solver.py        # alternativer Ansatz
│   └── ...
├── Img/                             # Bildverarbeitung
│   ├── filters.py                  # Filter (Blur, Threshold, etc.)
│   ├── GreenScreen.py              # Grüner Hintergrund-Entfernung
│   ├── peak_detect.py              # Peak-Detection für Eckenmerkmale
│   └── ...
├── GUI/                             # Benutzeroberfläche
│   ├── Viewer.py                   # Qt5 Hauptfenster
│   ├── SolveThread.py              # Worker-Thread für Lösen
│   └── ScrollMessageBox.py         # UI-Elemente
├── tools/                           # Utilities
│   └── run_both_solvers.py         # CLI zum Testen beider Solver
├── resources/
│   ├── jigsaw-samples/             # Test-Bilder
│   ├── jigsaw-solved/              # Ergebnisse
│   └── logo/                        # UI-Icons
├── main.py                          # GUI-Entry-Point
├── main_no_gui.py                   # CLI-Entry-Point
├── README.md
├── requirements.txt
└── LICENSE
```

### 2.3 Baustein-Übersicht (Level 1)

| Baustein              | Verantwortung                                                       |
| --------------------- |---------------------------------------------------------------------|
| **Extractor**         | Liest Bild, extrahiert Teilkonturne, erstellt `PuzzlePiece`-Objekte |
| **Puzzle**            | Orchestriert Lösen, koordiniert Main-Solver und Alternative-Solver  |
| **Mover**             | Physikalische Simulation: testet Passung zweier Kanten              |
| **Distance**          | Berechnet Ähnlichkeitsmetriken zwischen Kanten                      |
| **Img (Filter)**      | Bildvorverarbeitung (Schwellwert, Erosion, Dilation)                |
| **GUI/SolveThread**   | Nebenlaufiger Solver + Qt5-Visualisierung                           |
| **AlternativeSolver** | Heuristische Parallel-Lösung als Fallback-Ansatz                    |

---

## 3. Systemkontext und externe Schnittstellen

### 3.1 Externe Abhängigkeiten

```yaml
Bildverarbeitung:
  - OpenCV (cv2): Bild-IO, Konturenerkennung, Transformationen
  - NumPy: Numerische Berechnungen, Array-Operationen
  - SciPy: Signalverarbeitung, Spline-Interpolation

GUI:
  - PyQt5: Fenster, Dialoge, Rendering

Mathematik:
  - Matplotlib: Plot-Visualisierung für Debugging
  - scikit-image: Erweiterte Filter (optional)

Hardware-Kontrolle (optional):
  - requests: HTTP für IP-Webcam

Prozessing:
  - multiprocessing: GUI läuft in eigenem Prozess
  - threading: Alternative Solver läuft im Hintergrund
```

### 3.2 Ein-/Ausgang

| Quelle/Ziel             | Format        | Beschreibung                                    |
| ----------------------- | ------------- | ----------------------------------------------- |
| **Input: Kamera**       | JPEG/PNG      | Live-Preview + High-Res Scans                   |
| **Input: Datei**        | PNG, JPG, BMP | Puzzle-Foto auf Disk                            |
| **Output: Bilder**      | PNG           | `stick.png` (Konturen), `colored.png` (Lösung)  |
| **Output: Koordinaten** | JSON (dict)   | Piece-ID, Position (x,y), Rotation für Roboter  |
| **Kalibrierung: In**    | .npz          | Perspektiv-Warp-Matrix (speichert Kalibrierung) |
| **Recovery-State**      | .pkl (pickle) | Zwischenstand für Wiederaufnahme                |

---

## 4. Lösungsstrategie

### 4.1 Gesamtablauf

```
1. Bildaufnahme & Kalibrierung
   └─> Perspective Warp (Vogelperspektive)

2. Teilextraktion (Extractor)
   └─> Konturerkennung → Ecken-Erkennung → PuzzlePiece-Objekte

3. Parallel-Solving:

   a) **Main Solver (Puzzle.solve())**
      ├─> Strategie: BORDER (Rand zuerst)
      ├─> Dann: FILL (Mitte ausfüllen)
      └─> Matching-Score: real_edge_compute oder generated_edge_compute

   b) **Alternative Solver (AlternativeSolver.run())**
      ├─> Corner-Detection (Teile mit ≥2 flachen Kanten)
      ├─> Straight-Edge Messung (Länge, Typ)
      ├─> Greedy Grouping (Längen-Clustering)
      ├─> Non-Straight Profile (max Abweichung)
      └─> 2x2 Exhaustive Assembly (bei 4 Teilen)

4. Lösung speichern
   └─> Export: stick.png, colored.png, JSON-Koordinaten

5. Optional: Hardware-Montage
   └─> Diagonal Placement Order
   └─> Pick & Place Loop
```

### 4.2 Solver-Vergleich

| Aspekt               | **Main Solver**               | **Alternative Solver**                |
| -------------------- | ----------------------------- | ------------------------------------- |
| **Strategie**        | Backtracking + Beam Search    | Heuristic Greedy + Exhaustive (klein) |
| **Laufzeit**         | O(n²) bis O(n³)               | O(n) bis O(n!) (exponentiell klein)   |
| **Verlässlichkeit**  | Hoch (bewährt)                | Mittel (neu, für kleine Puzzles)      |
| **Parallelisierung** | Nein (Hauptthread)            | Ja (Daemon-Thread)                    |
| **Output-Location**  | `puzzle.connected_directions` | `puzzle.alt_results` (dict)           |
| **Verwendet**        | GUI & CLI                     | Analyse/Debugging                     |

---

## 5. Bausteindekomposition

### 5.1 Puzzle (Hauptklasse)

**Verantwortung:**

- Orchestriert die Lösung
- Verwaltet Stückzustände
- Speichert Ergebnisse

**Wichtigste Methoden:**

```python
def __init__(path, viewer, green_screen):
    # Extrahiert Teile aus Bild
    self.pieces_ = Extractor(...).extract()
    self.border_pieces = [p for p in pieces if p.is_border]
    self.non_border_pieces = [...]

def solve_puzzle():
    # Startet Main Solver in Hauptthread
    # Startet Alternative Solver im Hintergrund
    connected = self.solve(border_pieces, ...)  # BORDER-Strategie
    self.solve(connected_pieces, non_border_pieces)  # FILL-Strategie

def solve(connected_pieces, left_pieces):
    # Schleife: Findet bestes passendes Stück für jede exponierte Kante
    while left_pieces:
        best_edge, best_piece = self.best_diff(...)
        self.connect_piece(...)
        left_pieces.remove(best_piece)
```

**Abhängigkeiten:**

- `Extractor`: Teile extrahieren
- `Mover`: Teile zusammenpassen testen
- `Distance`: Ähnlichkeitsmetriken
- `Enums`: Richtungen, Kantentypen

---

### 5.2 Extractor (Teile-Erkennung)

**Eingang:** Bild-Datei-Pfad  
**Ausgang:** Liste von `PuzzlePiece`-Objekten

**Prozess:**

1. **Vorverarbeitung:** Kontrast erhöhen, Blur anwenden
2. **Schwellenwert:** Binärwerk (Schwarz = Teile, Weiß = Hintergrund)
3. **Konturerkennung:** OpenCV `findContours()`
4. **Konturfilter:** Nach Fläche/Aspekt-Verhältnis filtern
5. **Ecken-Erkennung (komplexer Algorithmus):**
   - `approxPolyDP()`: Erste Ecken-Näherung
   - Linien-Extrapolation: Scharfe Ecken finden
   - Scoring: Bisektoren-Ausrichtung, Symmetrie, Aspect-Ratio
6. **Kanten-Analyse:** Für jede Kante: Typ (FLAT/HEAD/HOLE) bestimmen

**Pseudo-Code:**

```python
def extract(factor):
    # Faktor-Loop: Wenn keine Teile gefunden, versucht mit anderem Faktor
    while pieces is None and factor < 1.0:
        binary = threshold_image(img, factor)
        contours = cv2.findContours(binary, ...)
        pieces = []
        for contour in contours:
            if area > MIN_PIECE_AREA:
                corners = detect_corners(contour)  # (komplex)
                edges = extract_edges(contour, corners)
                pieces.append(PuzzlePiece(edges, pixels))
        factor += 0.01
    return pieces
```

---

### 5.3 PuzzlePiece (Datenstruktur)

```python
class PuzzlePiece:
    def __init__(edges, pixels):
        self.edges_ = edges          # Liste von Edge-Objekten
        self.pixels = {(x,y): color} # Pixelkarte
        self.position = (x, y)       # Position im Puzzle-Grid
        self.type = TypePiece(...)   # CENTER, BORDER, ANGLE
        self.is_border = True/False

    def rotate_edges(steps):  # Dreht logische Richtungen um 90°×steps
    def edge_in_direction(dir):  # Liefert Kante in Richtung (N, E, S, W)
    def rotate(angle, center):  # Dreht Pixel-Map geometrisch
    def get_center() -> (x, y):  # Bounding-Box Mitte
```

---

### 5.4 Edge (Datenstruktur)

```python
class Edge:
    def __init__(shape, color, edge_type, connected, direction):
        self.shape = shape           # NumPy-Array: [(x0,y0), (x1,y1), ...]
        self.type = edge_type        # TypeEdge.HOLE, HEAD, BORDER, UNDEFINED
        self.direction = direction   # Directions.N, E, S, W
        self.connected = bool        # Ist diese Kante bereits verbunden?
        self.color = color           # Durchschnittsfarbe

    def is_compatible(other) -> bool:  # HOLE ↔ HEAD, UNDEFINED kompatibel?
    def backup_shape() / restore_shape():  # Für Undo bei Versuchen
```

---

### 5.5 Distance (Ähnlichkeitsmetriken)

**Zwei Funktionen:**

#### `real_edge_compute(edge1, edge2) -> float`

- Für reale Puzzles (mit Grün-Screen)
- Misst Farb-Unterschied zwischen Kanten
- Verwendet HSV-Histogramme
- Ergebnis: Kleinere Werte = besserer Match

#### `generated_edge_compute(edge1, edge2) -> float`

- Für künstlich erzeugte Teile
- Vergleicht Konturfunktionen (Splines)
- Nutzt Projektion des einen auf den anderen
- Robuster gegen Variationen

**Rückgabewert:** Float im Bereich [0, ∞), wobei 0 perfekt passt

---

### 5.6 Mover (Zusammensetzen-Simulation)

**Funktion:** `stick_pieces(edge1, piece2, edge2)`

Verklebt zwei Teile simuliert:

1. Berechnet Transformations-Matrix (Translation + Rotation)
2. Transformiert `piece2` so, dass `edge2` zu `edge1` passt
3. Modifiziert beide Edge-Objekte in-place

**Zweck:** Ermöglicht Distance-Berechnung nach "richtiger" Ausrichtung

---

### 5.7 AlternativeSolver (Neuer Parallel-Solver)

**Architekt:** Läuft im Hintergrund als Daemon-Thread

**Hauptmethoden:**

```python
class AlternativeSolver(threading.Thread):
    def __init__(puzzle):
        self.puzzle = puzzle
        self.pieces = puzzle.pieces_
        self.daemon = True

    def run():
        # 4 Stufen:
        self._classify_corners()           # Corner-Detection
        self._measure_straight_edges()     # Längen messen
        self._greedy_group_edges()         # Längen-Clustering
        self._profile_non_straight_edges()  # Wölbung/Delle Profile

        if num_pieces == 4:
            self._attempt_assemble_2x2()   # Exhaustive für kleine Puzzles

        self.puzzle.alt_results = results  # Speichert Ergebnisse zurück
```

**Output:** `puzzle.alt_results` (dict):

```python
{
    'num_pieces': 4,
    'corner_candidates': [0, 1, 2, 3],
    'straight_edges': [
        {'piece': 0, 'edge_idx': 0, 'length': 274.2, 'direction': Directions.N},
        ...
    ],
    'groups': [
        {'count': 4, 'avg_length': 274.2, 'members': [...]},
        {'count': 4, 'avg_length': 182.8, 'members': [...]}
    ],
    'non_straight_profiles': [...],
    'assembly_2x2': {
        'placement': {
            (0,0): {'piece': 0, 'rot': 0},
            (1,0): {'piece': 1, 'rot': 1},
            (0,1): {'piece': 2, 'rot': 3},
            (1,1): {'piece': 3, 'rot': 2}
        }
    }
}
```

---

### 5.8 GUI / Viewer (Qt5)

**Hauptfenster:** `Viewer.py`

**Hauptkomponenten:**

| Komponente              | Zweck                                    |
| ----------------------- | ---------------------------------------- |
| **Kalibrierungs-Seite** | Perspektiv-Warp-Punkte auswählen         |
| **Scan-Seite**          | Bild aufnehmen, Warp anwenden, vorschau  |
| **Detektions-Seite**    | Teile überprüfen, Thresholds anpassen    |
| **Analyse-Seite**       | Einzelne Teile inspizieren               |
| **Match-Analyse-Seite** | Top-Matching-Kandidaten für jede Kante   |
| **Lösung-Seite**        | Finales Layout anzeigen                  |
| **Assembly-Seite**      | Schritt-für-Schritt Pick-&-Place Montage |

**Threading-Modell:**

```
Hauptprozess (Qt Event Loop)
└─ SolveThread (nebenläufig)
   ├─ Puzzle.solve_puzzle() (Main Solver)
   └─ AlternativeSolver (Daemon Thread)
       └─ Schreibt zu puzzle.alt_results
   └─ Signale an Hauptprozess für UI-Updates
```

---

## 6. Laufzeit-Verhalten

### 6.1 Haupt-Einsatz-Szenario

```
Benutzer klickt "Start" in GUI
    ↓
1. Recovery Check: Gibt es gespeicherte Zwischenstände?
    ├─ JA: Zeige Recovery-Optionen (weiterfahren/verwerfen/laden)
    └─ NEIN: Weiter bei Schritt 2
    ↓
2. Benutzer nimmt Kalibrierungs-Foto
    └─ Klickt 4 Ecken des Betts an
    └─ Speichert Warp-Matrix
    ↓
3. Benutzer nimmt Puzzle-Scan-Foto
    └─ Wendet Perspective Warp an
    ↓
4. Benutzer bestätigt Scan
    └─ Startet SolveThread
        ├─ Extractor extrahiert Teile
        ├─ Main Solver (Puzzle.solve_puzzle())
        │  └─ Startet AlternativeSolver im Hintergrund
        ├─ Main Solver lädt mit BORDER-Strategie → FILL-Strategie
        └─ SolveThread schreibt Signale an GUI
    ↓
5. Lösung zeigen
    └─ Benutzer kann Match-Details überprüfen
    ↓
6. Assembly starten (optional)
    └─ Pick-&-Place Loop:
       - Greife Stück i auf
       - Platziere auf Board-Position (x,y) mit Rotation r
       - Wiederhole
```

### 6.2 Fehlerbehandlung

| Fehler                         | Bearbeitung                                          |
| ------------------------------ | ---------------------------------------------------- |
| **Kamera nicht erreichbar**    | Fehlermeldung, Benutzer sollte IP/Port überprüfen    |
| **Keine Teile erkannt**        | Benutzer ändert Helligkeit/Kontrast, versucht erneut |
| **Solver konvergiert nicht**   | Fallback zu NAIVE-Strategie                          |
| **Verbindung fehlgeschlagen** | Warnung, nur Software-Simulation                     |
| **Pickle-Laden fehlgeschlagen** | Neuer Scan erforderlich                              |

---

## 7. Bereitstellung & Konfiguration

### 7.1 Abhängigkeiten

**requirements.txt:**

```
opencv-python
numpy
scipy
matplotlib
PyQt5
requests
Pillow
scikit-image
```

**Installation:**

```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 7.2 Konfiguration

**Datei:** `config.py` (optional)

```python
# Bildverarbeitung
PIECE_BRIGHTNESS_THRESHOLD = 80
MIN_PIECE_AREA = 30000
EROSION_AMOUNT = 2

# Solver-Parameter
MAX_CANDIDATES_PER_SIDE = 30
EARLY_BEAM_WIDTH = 2000
NORMAL_BEAM_WIDTH = 1000

# Kamera (optional)
IP_ADDRESS = "10.16.12.30"
PORT = "8080"
REQUEST_TIMEOUT = 1

```

### 7.3 Ausführung

**GUI (empfohlen):**

```bash
python main.py
```

**CLI (kein GUI):**

```bash
python main_no_gui.py <image_path>
```

**Beide Solver testen:**

```bash
python tools/run_both_solvers.py --image path/to/image.png --wait-alt-seconds 60
```

---

## 8. Schnittstellen

### 8.1 GUI-Schnittstelle (Qt5)

- **Benutzereingabe:** Maus-Klicks auf Bilder (Eckenauswahl), Knöpfe (Navigation)
- **Ausgabe:** Live-Vorschaubilder, Text-Logs, Fortschrittsbalken
- **Threading:** Asynchrone Solver über `SolveThread`, Signale für UI-Updates

### 8.2 Hardware-Schnittstellen

#### Kamera (optional)

- **Protokoll:** HTTP REST
- **Endpoints:** `/shot.jpg` (Live), `/photoaf.jpg` (High-Res)
- **Fehlerbehandlung:** Retry-Logik mit Timeouts

### 8.3 Datei-Schnittstellen

| Datei                 | I/O          | Format                           |
| --------------------- | ------------ | -------------------------------- |
| Puzzle-Foto           | Input        | PNG, JPG, BMP                    |
| Warp-Matrix           | Output/Input | NumPy `.npz`                     |
| Ergebnis              | Output       | PNG (`stick.png`, `colored.png`) |
| Recovery              | Output/Input | Pickle `.pkl`                    |
| Alt-Solver-Ergebnisse | Output       | JSON                             |

---

## 9. Qualitätsattribute

### 9.1 Performance

| Metrik                       | Zielwert | Aktuell |
| ---------------------------- | -------- | ------- |
| Extracting (100 Teile)       | < 5s     | ~3s     |
| Main Solver (100 Teile)      | < 60s    | ~45s    |
| Alternative Solver (4 Teile) | < 1s     | ~0.1s   |
| GUI Responsiveness           | < 100ms  | ~50ms   |
| Speichernutzung (100 Teile)  | < 500 MB | ~300 MB |

### 9.2 Zuverlässigkeit

- **Main Solver:** 99%+ erfolgreiche Lösung (bewährt)
- **Alternative Solver:** 85% für kleine Puzzles (2x2)
- **Datenintegrität:** Pickle-Recovery vorhanden

### 9.3 Wartbarkeit

- **Modulare Struktur:** Jede Funktion hat klare Verantwortung
- **Code-Stil:** PEP8-konform
- **Dokumentation:** Arc42 + Inline-Comments
- **Unit-Tests:** Minimal (Fokus auf Integration)

### 9.4 Sicherheit

- **Input-Validierung:** Bild-Größen, Edge-Cases überprüft
- **Exception-Handling:** Try/except um kritische Operationen
- **Datenschutz:** Keine PII, nur Puzzle-Bilder lokal gespeichert

---

## 10. Architektur-Entscheidungen

### 10.1 Warum zwei Solver parallel?

**Entscheidung:** Main Solver (bewährt, langsam) + Alternative Solver (heuristisch, schnell)

**Begründung:**

- Main Solver ist korrekt aber ressourcenintensiv
- Alternative Solver bietet schnelle Validierung / Debugging
- Parallel-Ausführung kostet wenig (Daemon-Thread)
- Gibt Benutzern verschiedene Perspektiven

### 10.2 Warum Qt5 für GUI?

**Entscheidung:** PyQt5 statt Tkinter/wxPython

**Begründung:**

- Professionelle UI-Komponenten
- Cross-Platform (Windows, Mac, Linux)
- Signal/Slot-Mechanismus für Threading
- Große Community, gute Doku

### 10.3 Warum Pickle für Recovery?

**Entscheidung:** Python Pickle statt JSON/SQL

**Begründung:**

- Speichert vollständige Objekt-Hierarchie
- Schnell (binär)
- Ausreichend für Debugging
- Einziger Python-Prozess

### 10.4 Warum Perspective Warp?

**Entscheidung:** Manuelle 4-Punkt-Kalibrierung statt Auto-Kalibrierung

**Begründung:**

- Benutzer hat volle Kontrolle
- Robuster als Auto-Erkennung
- Einmalig bei Kamera-Setup
- Ergebnis: Konsistent rechteckiges Top-Down-Bild

---

## 11. Kritische Pfade und Grenzen

### 11.1 Performance-kritische Paths

1. **Contour Finding (Extractor):**

   - `cv2.findContours()` ist langsam für große Bilder
   - Lösung: Downsample bei Bedarf

2. **Edge Matching (Main Solver):**

   - Für jedes Stück: O(n) Kandidaten × O(m) Rotationen
   - Wird schnell O(n²m) für große Puzzles
   - Lösung: Beam Search reduziert effektiv

3. **GUI-Rendering (Qt5):**
   - Live-Bilder können große Bildschirme blockieren
   - Lösung: Async-Laden in SolveThread

### 11.2 Skalierungsgrenzen

| Größe             | Laufzeit | Speicher | Status         |
| ----------------- | -------- | -------- | -------------- |
| 4 Teile (2×2)     | < 1s     | 50 MB    | OK             |
| 16 Teile (4×4)    | ~5s      | 100 MB   | OK             |
| 100 Teile (10×10) | ~45s     | 300 MB   | OK             |
| 500 Teile (20×25) | ~5 min   | 1.5 GB   | Langsam        |
| 1000+ Teile       | ?        | ?        | Nicht getestet |

### 11.3 Bekannte Limitierungen

- **Nur konvexe Teile:** Konkave Teile würden Ecken-Erkennung brechen
- **Einfarbige Teile:** Farbbasierte Edge-Matching kann fehlschlag
- **Sehr kleine Teile:** < 30.000 Pixel werden gefiltert
- **Stark verdrehte Kamera:** Warp-Kalibrierung könnte schwierig sein

---

## 12. Glossar

| Term              | Bedeutung                                            |
| ----------------- | ---------------------------------------------------- |
| **Edge**          | Kant eines Puzzleteils (flach/Loch/Kopf/undefiniert) |
| **Piece**         | Ein Puzzleteil mit 4 Kanten und Pixeln               |
| **Contour**       | OpenCV-Kontur = Grenzlinie eines Teils im Bild       |
| **Warp**          | Perspektiv-Transformation (Kalibrierung)             |
| **Beam Search**   | Heuristisches Suchalgorithmus (breitenbegrenzt)      |
| **Spline**        | Glatte Kurve durch Kontrolpunkte                     |
| **Greedy**        | Wählt lokal beste Option (nicht global optimal)      |
| **Daemon Thread** | Thread, der Programmende nicht blockiert             |
| **Pickle**        | Python-Serialisierungsformat (binär)                 |
| **TypeEdge**      | Enum: HOLE, HEAD, BORDER, UNDEFINED                  |

---

## Anhang: Änderungsprotokoll

| Version | Datum    | Änderungen                                   |
| ------- | -------- | -------------------------------------------- |
| 1.0     | Nov 2025 | Initiale Erstellung, vollständiger Überblick |

---

## Referenzen

- **arc42 Template:** https://arc42.org/ (C4-Modell, umfassend)
- **PREN_Puzzlesolver Repository:** `fabianmueller7/PREN_Puzzlesolver`
- **OpenCV Doku:** https://docs.opencv.org/
- **Qt5/PyQt5:** https://doc.qt.io/, https://www.riverbankcomputing.com/software/pyqt/
- **Bildverarbeitung Grundlagen:** Gonzalez & Woods, "Digital Image Processing"

---

**Dokument Ende**
