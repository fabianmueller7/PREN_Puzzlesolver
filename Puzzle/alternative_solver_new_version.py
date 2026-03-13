import threading
import queue
import math
from typing import List, Optional, Tuple
import numpy as np
from .Enums import TypeEdge
from .Puzzle import Puzzle
from .PuzzlePiece import PuzzlePiece as Piece


def edge_length(shape: np.ndarray) -> float:
    """Compute the arc length of an edge shape (sequence of (x,y))."""
    if shape is None or len(shape) < 2:
        return 0.0
    diffs = np.diff(np.asarray(shape, dtype=float), axis=0)
    dists = np.hypot(diffs[:, 0], diffs[:, 1])
    return float(dists.sum())


def max_deviation_from_line(shape: np.ndarray) -> Tuple[float, int]:
    """Return (max_deviation, index_of_max) from straight line between ends.

    Uses perpendicular distance from the baseline to each intermediate point.
    """
    pts = np.asarray(shape, dtype=float)
    if pts.shape[0] < 3:
        return 0.0, 0
    p0 = pts[0]
    p1 = pts[-1]
    baseline = p1 - p0
    baseline_len = np.hypot(baseline[0], baseline[1])
    if baseline_len == 0:
        return 0.0, 0
    unit_baseline = baseline / baseline_len
    
    max_dev = 0.0
    max_idx = 0
    for i in range(1, len(pts) - 1):
        vec = pts[i] - p0
        proj = np.dot(vec, unit_baseline)
        proj_point = p0 + proj * unit_baseline
        dev = np.hypot(pts[i][0] - proj_point[0], pts[i][1] - proj_point[1])
        if dev > max_dev:
            max_dev = dev
            max_idx = i
    return max_dev, max_idx


def straight_length(shape: np.ndarray, threshold: float = 1.0) -> float:
    """Compute the length of the straight part of the edge until deviation exceeds threshold."""
    if shape is None or len(shape) < 3:
        return edge_length(shape)
    max_dev, idx = max_deviation_from_line(shape)
    if max_dev <= threshold:
        return edge_length(shape)
    # Length up to idx
    if idx < 2:
        return 0.0
    straight_shape = shape[:idx+1]
    return edge_length(straight_shape)


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
        Detect corners by finding pieces with 2 adjacent straight (BORDER) edges.
        """
        possible_corners = []

        for piece in self.puzzle.pieces_:
            # Check for two consecutive BORDER edges
            for i in range(4):
                if (piece.edges_[i].type == TypeEdge.BORDER and 
                    piece.edges_[(i+1) % 4].type == TypeEdge.BORDER):
                    possible_corners.append(piece)
                    break

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
        diff = abs(edge_length(e1.shape) - edge_length(e2.shape))

        # use angle if present
        if hasattr(e1, "angle") and hasattr(e2, "angle"):
            diff += abs(e1.angle - e2.angle) * 0.5

        return diff

    def find_best_match(self, piece: Piece, used_pieces: set) -> Optional[Piece]:
        """Finds the best-fitting neighbour based on relaxed thresholds."""

        best_piece = None
        best_score = math.inf

        for candidate in self.puzzle.pieces_:
            if self.puzzle.pieces_.index(candidate) in used_pieces:
                continue
            if candidate == piece:
                continue

            # relaxed matching
            for edge1 in piece.edges_:
                for edge2 in candidate.edges_:
                    # border edges never match each other
                    if edge1.type == TypeEdge.BORDER and edge2.type == TypeEdge.BORDER:
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

        print(f"[AlternativeSolver] corners detected: {[self.puzzle.pieces_.index(c) for c in corners]}")

        used = set()
        solution = {}

        start_piece = corners[0]
        used.add(self.puzzle.pieces_.index(start_piece))
        solution[0] = self.puzzle.pieces_.index(start_piece)

        current = start_piece

        index = 1
        while len(used) < len(self.puzzle.pieces_):
            match = self.find_best_match(current, used)

            if match is None:
                # try random corner fallback if stuck
                for p in self.puzzle.pieces_:
                    if self.puzzle.pieces_.index(p) not in used:
                        match = p
                        break

            used.add(self.puzzle.pieces_.index(match))
            solution[index] = self.puzzle.pieces_.index(match)
            current = match
            index += 1

        # return solution (a simple ordering of piece numbers)
        self.result_queue.put(solution)
        self.puzzle.alt_results = {
            'solution': solution,
            'corner_candidates': [self.puzzle.pieces_.index(c) for c in corners],
            'num_pieces': len(self.puzzle.pieces_)
        }

    # ----------------------------------------------------------------------
    # Thread run()
    # ----------------------------------------------------------------------

    def run(self):
        try:
            self.solve()
        except Exception as e:
            print("[AlternativeSolver] failed:", e)
