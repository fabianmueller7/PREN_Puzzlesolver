"""
LEGO Puzzle Solver - Third Solver Method for PREN_Puzzlesolver
"""

import threading
import queue
import os
from typing import List, Optional, Tuple, Dict, Set
import numpy as np
from .Enums import TypeEdge, Directions, get_opposite_direction, directions
from .Puzzle import Puzzle
from .PuzzlePiece import PuzzlePiece as Piece
from .Mover import stick_pieces
from .Distance import generated_edge_compute, real_edge_compute
from .tuple_helper import display_dim


class LegoSolver(threading.Thread):
    """
    Ein Tree-Search Backtracking Solver, der alle Teile im Grid anordnet.
    """

    def __init__(self, puzzle: Puzzle, result_queue: queue.Queue):
        super().__init__()
        self.puzzle = puzzle
        self.result_queue = result_queue
        self.daemon = True
        self.grid = {}
        self.used_indices = set()

    def log(self, text):
        print(f"[LEGO_SOLVER] {text}")

    def get_corner_pieces(self):
        """Findet alle Stücke mit 2 BORDER-Kanten."""
        return [i for i, p in enumerate(self.puzzle.pieces_) if p.number_of_border() >= 2]

    def check_border_constraint(self, piece, x, y, w, h):
        """Prüft, ob die Außenkanten des Teils flach sind."""
        # North (y=0)
        if y == 0 and piece.edge_in_direction(Directions.N).type != TypeEdge.BORDER: return False
        if y > 0 and piece.edge_in_direction(Directions.N).type == TypeEdge.BORDER: return False
        # South (y=h-1)
        if y == h - 1 and piece.edge_in_direction(Directions.S).type != TypeEdge.BORDER: return False
        if y < h - 1 and piece.edge_in_direction(Directions.S).type == TypeEdge.BORDER: return False
        # West (x=0)
        if x == 0 and piece.edge_in_direction(Directions.W).type != TypeEdge.BORDER: return False
        if x > 0 and piece.edge_in_direction(Directions.W).type == TypeEdge.BORDER: return False
        # East (x=w-1)
        if x == w - 1 and piece.edge_in_direction(Directions.E).type != TypeEdge.BORDER: return False
        if x < w - 1 and piece.edge_in_direction(Directions.E).type == TypeEdge.BORDER: return False
        return True

    def get_match_score(self, piece, x, y):
        """Berechnet Nose-Fit und Corner-Gap Score zu bestehenden Nachbarn."""
        total_score = 0
        # Check West neighbor
        if x > 0:
            neighbor = self.grid[(x - 1, y)]
            e_neighbor = neighbor.edge_in_direction(Directions.E)
            e_piece = piece.edge_in_direction(Directions.W)
            if not e_neighbor.is_compatible(e_piece): return float('inf')
            
            stick_pieces(e_neighbor, piece, e_piece)
            gap = np.linalg.norm(np.array(e_neighbor.shape[-1]) - np.array(e_piece.shape[0]))
            if self.puzzle.green_:
                nose_score = real_edge_compute(e_neighbor, e_piece)
            else:
                nose_score = generated_edge_compute(e_neighbor, e_piece)
            total_score += nose_score + (gap * 2.0)

        # Check North neighbor
        if y > 0:
            neighbor = self.grid[(x, y - 1)]
            e_neighbor = neighbor.edge_in_direction(Directions.S)
            e_piece = piece.edge_in_direction(Directions.N)
            if not e_neighbor.is_compatible(e_piece): return float('inf')
            
            stick_pieces(e_neighbor, piece, e_piece)
            gap = np.linalg.norm(np.array(e_neighbor.shape[-1]) - np.array(e_piece.shape[0]))
            if self.puzzle.green_:
                nose_score = real_edge_compute(e_neighbor, e_piece)
            else:
                nose_score = generated_edge_compute(e_neighbor, e_piece)
            total_score += nose_score + (gap * 2.0)
            
        return total_score

    def align_all_pieces(self, w, h):
        """Aligns all pieces physically in the grid using stick_pieces."""
        for idx in range(1, len(self.puzzle.pieces_)):
            x = idx % w
            y = idx // w
            piece = self.grid.get((x, y))
            if not piece: continue
            
            # Process in order: stick to West or North neighbor
            if x > 0:
                neighbor = self.grid.get((x - 1, y))
                if neighbor:
                    e_neighbor = neighbor.edge_in_direction(Directions.E)
                    e_piece = piece.edge_in_direction(Directions.W)
                    # Ensure float shapes to avoid NumPy casting errors in Mover.py (float += translation)
                    for edge in piece.edges_:
                        if edge.shape.dtype != np.float64:
                            edge.shape = edge.shape.astype(np.float64)
                    stick_pieces(e_neighbor, piece, e_piece, final_stick=True)
                    continue # Already stuck to West
            
            if y > 0:
                neighbor = self.grid.get((x, y - 1))
                if neighbor:
                    e_neighbor = neighbor.edge_in_direction(Directions.S)
                    e_piece = piece.edge_in_direction(Directions.N)
                    # Ensure float shapes to avoid NumPy casting errors in Mover.py (float += translation)
                    for edge in piece.edges_:
                        if edge.shape.dtype != np.float64:
                            edge.shape = edge.shape.astype(np.float64)
                    stick_pieces(e_neighbor, piece, e_piece, final_stick=True)

    def solve_recursive(self, idx, w, h):
        """Backtracking Kern: Versucht das nächste Grid-Feld zu füllen."""
        if idx == len(self.puzzle.pieces_):
            return True

        x = idx % w
        y = idx // w

        candidates = []
        for i, piece in enumerate(self.puzzle.pieces_):
            if i in self.used_indices: continue
            
            # Optimierung: Da (0,0) immer eine Ecke ist, muss das erste Teil 
            # mindestens 2 Border-Kanten haben.
            if idx == 0 and piece.number_of_border() < 2: continue

            # Save state locally for rotations and backtracking to avoid corrupting global backup
            local_state = piece.get_current_state()
            
            for rot in range(4):
                piece.set_state(local_state)
                # Ensure float shapes to avoid NumPy casting errors in Mover.py
                for edge in piece.edges_:
                    if edge.shape.dtype != np.float64:
                        edge.shape = edge.shape.astype(np.float64)
                        
                piece.rotate_edges(rot)
                if self.check_border_constraint(piece, x, y, w, h):
                    score = self.get_match_score(piece, x, y)
                    if score < 1000: # Threshold for basic compatibility
                        candidates.append((score, i, rot))

        # Sortiere nach bestem Score (Nase + Corner Gap)
        candidates.sort(key=lambda x: x[0])

        for score, i, rot in candidates:
            piece = self.puzzle.pieces_[i]
            piece.rotate_edges(rot)
            self.grid[(x, y)] = piece
            self.used_indices.add(i)
            
            if self.solve_recursive(idx + 1, w, h):
                return True
                
            self.used_indices.remove(i)
            del self.grid[(x, y)]
            piece.restore_initial_state()
            
        return False

    def run(self):
        try:
            self.log("Starting Tree-Search Solver (Lego Replacement)")
            
            # Reset all pieces to their 'clean' state before starting LEGO solver
            for piece in self.puzzle.pieces_:
                piece.restore_initial_state()

            possible_dims = self.puzzle.possible_dim
            # Wir wissen: Es kommen nur 4 Teile (2x2) oder 6 Teile (2x3/3x2) vor.
            # Alle Teile liegen am Rand.
            valid_dims = []
            for w_m1, h_m1 in possible_dims:
                w, h = w_m1 + 1, h_m1 + 1
                if (w == 2 and h == 2) or (w == 2 and h == 3) or (w == 3 and h == 2):
                    valid_dims.append((w, h))

            if not valid_dims:
                error_msg = f"No valid dimensions (2x2, 2x3, 3x2) found in {display_dim(possible_dims)}. Aborting."
                self.log(error_msg)
                result = {
                    'success': False,
                    'error': error_msg
                }
                self.puzzle.lego_results = result
                self.result_queue.put(result)
                return

            success = False
            final_dim = (0, 0)

            for w, h in valid_dims:
                self.log(f"Attempting dimensions: {w}x{h}")
                # Reset state for each dimension attempt
                for piece in self.puzzle.pieces_:
                    piece.restore_initial_state()
                    for edge in piece.edges_:
                        edge.shape = edge.shape.astype(np.float64)
                self.grid = {}
                self.used_indices = set()
                
                # Start search
                if self.solve_recursive(0, w, h):
                    success = True
                    final_dim = (w, h)
                    break
            
            if success:
                self.log(f"Success! Puzzle solved with dimensions {display_dim([final_dim])}")
                
                # Align pieces physically for the final image
                self.align_all_pieces(final_dim[0], final_dim[1])
                self.puzzle.translate_puzzle()
                
                # Export images
                temp_dir = os.environ.get("ZOLVER_TEMP_DIR", ".")
                self.puzzle.export_pieces(
                    os.path.join(temp_dir, "lego_stick.png"),
                    os.path.join(temp_dir, "lego_colored.png"),
                    display=False
                )

                placements = {}
                for (x, y), piece in self.grid.items():
                    p_idx = self.puzzle.pieces_.index(piece)
                    placements[p_idx] = (x, y)
                
                result = {
                    'success': True,
                    'pieces_placed': len(self.grid),
                    'total_pieces': len(self.puzzle.pieces_),
                    'placements': placements,
                    'dimension': final_dim
                }
            else:
                self.log("Failed to find a valid arrangement for any dimension.")
                result = {'success': False, 'pieces_placed': 0}

            self.puzzle.lego_results = result
            self.result_queue.put(result)
            
        except Exception as e:
            self.log(f"Error in Solver: {e}")
            import traceback
            traceback.print_exc()
            result = {
                'success': False,
                'error': str(e)
            }
            self.puzzle.lego_results = result
            self.result_queue.put(result)
