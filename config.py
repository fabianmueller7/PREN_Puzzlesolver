DEBUG_FILE_OUTPUT = 1    # Saves debug images/data to debug_output/ and runs processing sequentially
DEBUG_SHOW_DIAGRAMS = 1  # Shows matplotlib diagrams interactively (requires DEBUG_FILE_OUTPUT = 1)
DEBUG_ALT_SOLVER = 1     # Saves debug images for the alternative solver to debug_output/

EDGE_OFFSET = 11  # pixels (≈ 2 mm per edge side) — shifts each edge outward to show the manufacturing tolerance band in debug output

# Schwellenwert für Kanten-Matches. 
# Werte unter 1000 stellen sicher, dass nur HEAD-HOLE Paarungen akzeptiert werden.
# Ein niedrigerer Wert (z.B. 600) erhöht die Strenge bei der geometrischen Passform.
MATCH_THRESHOLD = 2.6
