import threading
import queue
import math
from typing import List, Optional

from enumerations import TypeEdge
from geometry import angle_between
from Puzzle import Puzzle, Piece


class AlternativeSolver(threading.Thread):
    """
    A more robust secondary solver that tries alternative placement heuristics.
    It does NOT change the original solver's logic, but extends it with:
    - robust corner detection
    - relaxed edge matching
    - improved fallback behaviour
    - rotation synchronization with main solver
    """

    def __init__(self, puzzle: Puzzle, result_queue: queue.Queue):
        super().__init__()
        self.puzzle = puzzle
        self.result_queue = result_queue

        # ensures this thread does not block program exit
        self.daemon = True

        # synchronise rotations from main solver if available
        self._sync_rotations()

    # ----------------------------------------------------------------------
    # Rotation Synchronisation
    # ----------------------------------------------------------------------

    def _sync_rotations(self):
        """Synchronises rotations of pieces with the main solver."""
        for piece in self.puzzle.pieces_:
            if hasattr(piece, "rotation_angle"):  # main solver sets this
                rotations = (piece.rotation_angle // 90) % 4
                for _ in range(rotations):
                    piece.rotate_edges(1)

    # ----------------------------------------------------------------------
    # Corner Detection
    # ----------------------------------------------------------------------

    def detect_corners(self) -> List[Piece]:
        """
        More tolerant corner detection.
        A corner is a piece with >= 2 border-like edges.
        """
        possible_corners = []

        for piece in self.puzzle.pieces_:
            flat_edges = 0

            for e in piece.edges_:
                if e.type == TypeEdge.FLAT:
                    flat_edges += 1
                elif getattr(e, "curvature", 0) < 0.15:
                    # allow "almost flat" edges
                    flat_edges += 1

            if flat_edges >= 2:
                possible_corners.append(piece)

        # strict mode fallback:
        if len(possible_corners) == 0:
            # old strict logic
            for piece in self.puzzle.pieces_:
                count = sum(1 for e in piece.edges_ if e.type == TypeEdge.FLAT)
                if count == 2:
                    possible_corners.append(piece)

        # final fallback: if puzzle is 2x2, allow all candidates
        if len(possible_corners) == 0 and len(self.puzzle.pieces_) == 4:
            return list(self.puzzle.pieces_)

        return possible_corners

    # ----------------------------------------------------------------------
    # Edge Matching Heuristic
    # ----------------------------------------------------------------------

    def edge_diff(self, e1, e2) -> float:
        """
        Computes similarity between two edges.
        Smaller = better.
        """
        # classic diff
        diff = abs(e1.length - e2.length)

        # use angle if present
        if hasattr(e1, "angle") and hasattr(e2, "angle"):
            diff += abs(e1.angle - e2.angle) * 0.5

        return diff

    def find_best_match(self, piece: Piece, used_pieces: set) -> Optional[Piece]:
        """Finds the best-fitting neighbour based on relaxed thresholds."""

        best_piece = None
        best_score = math.inf

        for candidate in self.puzzle.pieces_:
            if candidate.number in used_pieces:
                continue
            if candidate == piece:
                continue

            # relaxed matching
            for edge1 in piece.edges_:
                for edge2 in candidate.edges_:
                    # border edges never match each other
                    if edge1.type == TypeEdge.FLAT and edge2.type == TypeEdge.FLAT:
                        continue

                    score = self.edge_diff(edge1, edge2)
                    if score < best_score:
                        best_score = score
                        best_piece = candidate

        # additional safety: requires a somewhat reasonable score
        if best_score < 300:   # typical threshold for your extracted pieces
            return best_piece

        return None

    # ----------------------------------------------------------------------
    # Placement
    # ----------------------------------------------------------------------

    def solve(self):
        """Tries to solve by placing pieces outward from a detected corner."""
        corners = self.detect_corners()

        if len(corners) == 0:
            print("[AlternativeSolver] No corners detected → abort.")
            return

        print(f"[AlternativeSolver] corners detected: {[c.number for c in corners]}")

        used = set()
        solution = {}

        start_piece = corners[0]
        used.add(start_piece.number)
        solution[0] = start_piece.number

        current = start_piece

        index = 1
        while len(used) < len(self.puzzle.pieces_):
            match = self.find_best_match(current, used)

            if match is None:
                # try random corner fallback if stuck
                for p in self.puzzle.pieces_:
                    if p.number not in used:
                        match = p
                        break

            used.add(match.number)
            solution[index] = match.number
            current = match
            index += 1

        # return solution (a simple ordering of piece numbers)
        self.result_queue.put(solution)

    # ----------------------------------------------------------------------
    # Thread run()
    # ----------------------------------------------------------------------

    def run(self):
        try:
            self.solve()
        except Exception as e:
            print("[AlternativeSolver] failed:", e)
