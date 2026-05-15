import os
import sys
import queue
import time
import glob

# Pfad-Setup
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Debug-Verzeichnis vorbereiten
debug_dir = os.path.join(REPO_ROOT, "debug_output_lego")
os.makedirs(debug_dir, exist_ok=True)
for f in glob.glob(os.path.join(debug_dir, "*")):
    try: os.remove(f)
    except: pass
os.environ["ZOLVER_TEMP_DIR"] = debug_dir

from Puzzle.Puzzle import Puzzle
from Puzzle.lego_solver import LegoSolver

def run_standalone_lego(image_path):
    print(f"--- LEGO Solver Standalone ---")
    print(f"Bild: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"Fehler: Datei nicht gefunden: {image_path}")
        return

    # 1. Puzzle initialisieren (extrahiert automatisch die Teile)
    print("Extrahiere Teile...")
    puzzle = Puzzle(image_path)
    
    # 2. Lego Solver vorbereiten
    res_queue = queue.Queue()
    solver = LegoSolver(puzzle, res_queue)
    
    # 3. Solver starten
    print("Starte Lego Solver...")
    start_time = time.time()
    solver.start()
    
    # 4. Auf Ergebnis warten
    try:
        result = res_queue.get(timeout=60) # 60 Sekunden Timeout
        duration = time.time() - start_time
        print(f"\nErgebnis nach {duration:.2f}s:")
        print(f"Erfolg: {result.get('success')}")
        print(f"Platzierte Teile: {result.get('pieces_placed')}/{result.get('total_pieces')}")
        if result.get('success'):
            print(f"Dimension: {result.get('dimension')}")
            print(f"Placements: {result.get('placements')}")
    except queue.Empty:
        print("\nTimeout: Der Solver hat zu lange gebraucht.")

if __name__ == "__main__":
    img = "resources/must works/PREN-Samples.png"
    if len(sys.argv) > 1:
        img = sys.argv[1]
    run_standalone_lego(img)