import threading
import queue
import math
import os
from typing import List, Optional, Tuple
import numpy as np
import cv2
import config
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

        if config.DEBUG_ALT_SOLVER == 1:
            self._save_debug(corners, solution)

    # ----------------------------------------------------------------------
    # Debug Output
    # ----------------------------------------------------------------------

    def _debug_dir(self) -> str:
        """Return the debug output directory, clearing it before each run."""
        import shutil
        d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "debug_output")
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)
        return d

    def _save_debug(self, corners: List[Piece], solution: dict):
        """
        Save debug visualisations to debug_output/ in the same style as the
        main solver's export_pieces():
          - alt_edges.png   – all pieces with edges coloured by type, corners marked
          - alt_solution.png – pieces labelled with their solution order number
        """
        if not self.puzzle.pieces_:
            return

        pieces = self.puzzle.pieces_
        bboxes = [p.get_bbox() for p in pieces]
        minX = min(b[0] for b in bboxes)
        minY = min(b[1] for b in bboxes)
        maxX = max(b[2] for b in bboxes)
        maxY = max(b[3] for b in bboxes)

        h = maxX - minX + 1
        w = maxY - minY + 1
        colored_img = np.zeros((h, w, 3), dtype=np.uint8)
        edge_img    = np.zeros((h, w, 3), dtype=np.uint8)

        corner_set = set(id(p) for p in corners)

        for piece in pieces:
            # --- coloured pixels ---
            for (px, py), c in piece.pixels.items():
                ix, iy = int(px) - minX, int(py) - minY
                if 0 <= ix < h and 0 <= iy < w:
                    colored_img[ix, iy] = c

            # --- edges coloured by type ---
            for e in piece.edges_:
                for ey, ex in e.shape:
                    ix, iy = int(ex) - minX, int(ey) - minY
                    if 0 <= ix < h and 0 <= iy < w:
                        if e.type == TypeEdge.HOLE:
                            rgb = (255, 178, 102)   # blue-ish (BGR stored as RGB here)
                        elif e.type == TypeEdge.HEAD:
                            rgb = (102, 255, 255)
                        elif e.type == TypeEdge.BORDER:
                            rgb = (0, 255, 0)
                        else:
                            rgb = (0, 0, 255)
                        edge_img[ix, iy] = rgb

            # --- corner dots at edge start points ---
            for e in piece.edges_:
                if len(e.shape) == 0:
                    continue
                ey, ex = e.shape[0]
                ix, iy = int(ex) - minX, int(ey) - minY
                if 0 <= ix < h and 0 <= iy < w:
                    dot_color = (255, 0, 255) if id(piece) in corner_set else (128, 0, 128)
                    cv2.circle(edge_img, (iy, ix), 5, dot_color, -1)

            # --- centre-of-mass marker (from edge contour points) ---
            edge_pts = [(int(ex), int(ey)) for e in piece.edges_ for ey, ex in e.shape]
            if edge_pts:
                com_x = int(np.mean([p[0] for p in edge_pts])) - minX
                com_y = int(np.mean([p[1] for p in edge_pts])) - minY
                if 0 <= com_x < h and 0 <= com_y < w:
                    cv2.drawMarker(edge_img, (com_y, com_x), (0, 255, 255),
                                   cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)

        # --- legend on edge image ---
        legend = [
            ((0, 255, 0),     "BORDER (flat edge)"),
            ((255, 178, 102), "HOLE (indentation)"),
            ((102, 255, 255), "HEAD (protrusion)"),
            ((0, 0, 255),     "UNDEFINED"),
            ((255, 0, 255),   "Corner candidate"),
        ]
        box_size, padding = 16, 6
        font, font_scale, font_thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
        line_h = box_size + padding
        legend_h = len(legend) * line_h + padding
        legend_w = 220
        lx, ly = 10, 10
        cv2.rectangle(edge_img, (lx - 2, ly - 2),
                      (lx + legend_w, ly + legend_h), (40, 40, 40), -1)
        for i, (color, label) in enumerate(legend):
            y0 = ly + padding + i * line_h
            cv2.rectangle(edge_img, (lx + 4, y0),
                          (lx + 4 + box_size, y0 + box_size), color, -1)
            cv2.putText(edge_img, label,
                        (lx + 4 + box_size + 6, y0 + box_size - 3),
                        font, font_scale, (220, 220, 220), font_thickness, cv2.LINE_AA)

        # --- solution-order overlay on coloured image ---
        solution_img = colored_img.copy()

        # Draw internal (non-BORDER) edges as green lines
        for piece in pieces:
            for e in piece.edges_:
                if e.type == TypeEdge.BORDER:
                    continue
                pts = np.array(
                    [[int(ey) - minY, int(ex) - minX] for ey, ex in e.shape],
                    dtype=np.int32
                )
                if len(pts) >= 2:
                    cv2.polylines(solution_img, [pts], isClosed=False,
                                  color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
        idx_to_piece = {self.puzzle.pieces_.index(p): p for p in pieces}
        for order, piece_idx in solution.items():
            piece = idx_to_piece.get(piece_idx)
            if piece is None:
                continue
            edge_pts = [(int(ex), int(ey)) for e in piece.edges_ for ey, ex in e.shape]
            if not edge_pts:
                continue
            com_x = int(np.mean([p[0] for p in edge_pts])) - minX
            com_y = int(np.mean([p[1] for p in edge_pts])) - minY
            label = str(order)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            tx, ty = com_y - tw // 2, com_x + th // 2
            cv2.putText(solution_img, label, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(solution_img, label, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)

        out = self._debug_dir()
        config.save_debug_img(os.path.join(out, "alt_edges.png"), edge_img)
        config.save_debug_img(os.path.join(out, "alt_solution.png"), solution_img)
        print(f"[AlternativeSolver] debug images saved to {out}/alt_edges.png and alt_solution.png")

    # ----------------------------------------------------------------------
    # Thread run()
    # ----------------------------------------------------------------------

    def run(self):
        try:
            self.solve()
        except Exception as e:
            print("[AlternativeSolver] failed:", e)
