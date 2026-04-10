import threading
import queue
import math
import os
from typing import List, Optional, Tuple, Dict, Set
import numpy as np
import cv2
import config
from .Enums import TypeEdge, Directions, get_opposite_direction
from .Puzzle import Puzzle
from .PuzzlePiece import PuzzlePiece as Piece
from .Mover import stick_pieces
from .Distance import generated_edge_compute


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


def edge_curvature(shape: np.ndarray) -> float:
    """Compute normalized curvature (roughness) of an edge.
    0 = perfectly straight, higher values = more curved/complex."""
    if shape is None or len(shape) < 2:
        return 0.0
    max_dev, _ = max_deviation_from_line(shape)
    edge_len = edge_length(shape)
    if edge_len < 1:
        return 0.0
    return max_dev / (edge_len + 1.0)


def edge_straightness(shape: np.ndarray, threshold: float = 2.0) -> float:
    """Return 1.0 if edge is straight (within threshold), 0.0 if curved."""
    curvature = edge_curvature(shape)
    return 1.0 if curvature <= threshold else 0.0


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
    Improved alternative solver that:
    1. Actually places pieces on a grid (like main solver)
    2. Has better corner and edge classification
    3. Provides step-by-step visualization
    4. Uses more robust edge matching with rotation testing
    """

    def __init__(self, puzzle: Puzzle, result_queue: queue.Queue):
        super().__init__()
        self.puzzle = puzzle
        self.result_queue = result_queue
        self.daemon = True
        
        # Grid-based placement tracking
        self.connected_pieces: List[Piece] = []
        self.used_pieces: Set[int] = set()
        self.piece_positions: Dict[int, Tuple[int, int]] = {}  # piece_idx -> (grid_x, grid_y)
        self.grid: Dict[Tuple[int, int], int] = {}  # (grid_x, grid_y) -> piece_idx
        
        # Bounds of current filled grid
        self.min_x = 0
        self.max_x = 0
        self.min_y = 0
        self.max_y = 0
        
        self.step_count = 0

    # ======================================================================
    # EDGE ANALYSIS
    # ======================================================================

    def classify_edge_complexity(self, edge) -> str:
        """Classify edge as STRAIGHT (border), SIMPLE (hole/head), or COMPLEX."""
        if edge.type == TypeEdge.BORDER:
            return "STRAIGHT"
        curvature = edge_curvature(edge.shape)
        if curvature < 1.0:
            return "SIMPLE"
        return "COMPLEX"

    def get_edge_signature(self, edge) -> Tuple[float, float, str]:
        """Return a signature for comparing edges: (length, curvature, complexity)."""
        length = edge_length(edge.shape)
        curvature = edge_curvature(edge.shape)
        complexity = self.classify_edge_complexity(edge)
        return (length, curvature, complexity)

    # ======================================================================
    # CORNER & EDGE DETECTION
    # ======================================================================

    def detect_corners(self) -> List[Piece]:
        """
        Detect corners by finding pieces with:
        - 2 adjacent BORDER edges
        - Both borders are relatively straight
        """
        possible_corners = []

        for piece in self.puzzle.pieces_:
            # Check for two consecutive BORDER edges
            for i in range(4):
                edge1 = piece.edges_[i]
                edge2 = piece.edges_[(i+1) % 4]
                
                if edge1.type == TypeEdge.BORDER and edge2.type == TypeEdge.BORDER:
                    # Check straightness
                    str1 = edge_straightness(edge1.shape)
                    str2 = edge_straightness(edge2.shape)
                    if str1 > 0.5 and str2 > 0.5:
                        possible_corners.append(piece)
                        break

        # Fallback for small puzzles
        if len(possible_corners) == 0 and len(self.puzzle.pieces_) <= 4:
            return list(self.puzzle.pieces_)

        return possible_corners

    def classify_pieces_by_type(self) -> Dict[str, List[Piece]]:
        """Classify pieces as CORNER or BORDER or CENTER based on BORDER edges."""
        result = {"CORNER": [], "BORDER": [], "CENTER": []}
        
        for piece in self.puzzle.pieces_:
            num_borders = sum(1 for e in piece.edges_ if e.type == TypeEdge.BORDER)
            
            if num_borders == 2:
                result["CORNER"].append(piece)
            elif num_borders == 1:
                result["BORDER"].append(piece)
            else:
                result["CENTER"].append(piece)
        
        return result

    # ======================================================================
    # EDGE MATCHING WITH ROTATION
    # ======================================================================

    def edge_diff(self, e1, e2) -> float:
        """Compare two edges. Smaller = more similar.
        
        Considers: length, curvature, type compatibility.
        """
        if e1.type == TypeEdge.BORDER or e2.type == TypeEdge.BORDER:
            return float('inf')
        
        # Must be opposite types (HOLE <-> HEAD)
        if e1.type == e2.type:
            return float('inf')
        
        # Compare lengths
        len1 = edge_length(e1.shape)
        len2 = edge_length(e2.shape)
        length_diff = abs(len1 - len2)
        
        # Compare curvatures
        curv1 = edge_curvature(e1.shape)
        curv2 = edge_curvature(e2.shape)
        curv_diff = abs(curv1 - curv2)
        
        # Weighted sum: prefer length match, then curvature
        return length_diff * 1.0 + curv_diff * 0.5

    def find_best_match(self, edge, available_pieces: List[Piece]) -> Optional[Tuple[Piece, int, float]]:
        """Find best-matching piece for given edge from available pieces.
        
        Returns: (matching_piece, rotation_amount, score) or None
        """
        best_piece = None
        best_rotation = 0
        best_score = float('inf')

        for piece in available_pieces:
            # Try each rotation
            for rotation in range(4):
                for candidate_edge in piece.edges_:
                    score = self.edge_diff(edge, candidate_edge)
                    if score < best_score:
                        best_score = score
                        best_piece = piece
                        best_rotation = rotation

        # Only accept reasonable matches
        if best_score < 100.0:  # threshold
            return (best_piece, best_rotation, best_score)

        return None

    # ======================================================================
    # GRID-BASED PUZZLE SOLVING
    # ======================================================================

    def place_piece_at(self, piece: Piece, x: int, y: int, rotation: int = 0):
        """Place a piece at grid position (x, y) with given rotation."""
        piece_idx = self.puzzle.pieces_.index(piece)
        
        # Apply rotation
        for _ in range(rotation):
            piece.rotate_edges(1)
        
        # Track placement
        self.connected_pieces.append(piece)
        self.used_pieces.add(piece_idx)
        self.piece_positions[piece_idx] = (x, y)
        self.grid[(x, y)] = piece_idx
        
        # Update bounds
        self.min_x = min(self.min_x, x)
        self.max_x = max(self.max_x, x)
        self.min_y = min(self.min_y, y)
        self.max_y = max(self.max_y, y)

    def find_next_placement(self) -> Optional[Tuple[Piece, int, int, int]]:
        """Find the next piece to place and where/how to place it.
        
        Returns: (piece, x, y, rotation) or None
        """
        available = [p for i, p in enumerate(self.puzzle.pieces_) if i not in self.used_pieces]
        
        if not available:
            return None

        # Check expansion positions around current grid
        candidates = []
        directions_map = {
            (-1, 0): Directions.W,
            (1, 0): Directions.E,
            (0, -1): Directions.S,
            (0, 1): Directions.N,
        }
        
        # Try all four directions from each placed piece
        for (gx, gy), piece_idx in self.grid.items():
            neighbor_piece = self.puzzle.pieces_[piece_idx]
            
            for (dx, dy), direction in directions_map.items():
                nx, ny = gx + dx, gy + dy
                
                # Skip if already filled
                if (nx, ny) in self.grid:
                    continue
                
                # Get exposed edge of neighbor
                neighbor_edge = neighbor_piece.edge_in_direction(direction)
                if neighbor_edge is None or neighbor_edge.type == TypeEdge.BORDER:
                    continue
                
                # Try matching available pieces
                for piece in available:
                    opposite_dir = get_opposite_direction(direction)
                    for rotation in range(4):
                        for candidate_edge in piece.edges_:
                            score = self.edge_diff(neighbor_edge, candidate_edge)
                            if score < 100.0:  # Good enough threshold
                                candidates.append((piece, nx, ny, rotation, score))
        
        if candidates:
            candidates.sort(key=lambda x: x[4])  # Sort by score
            return candidates[0][:4]

        return None

    def solve_grid_based(self):
        """Solve puzzle by placing pieces on a grid, like the main solver."""
        print("[AlternativeSolver] Starting grid-based solving")
        
        # Find and place first corner
        corners = self.detect_corners()
        if not corners:
            print("[AlternativeSolver] No corners found - aborting")
            return False
        
        start_piece = corners[0]
        self.place_piece_at(start_piece, 0, 0, 0)
        self._save_progress_image(f"Step 001 - Initial corner")
        
        # Greedily place remaining pieces
        while len(self.used_pieces) < len(self.puzzle.pieces_):
            placement = self.find_next_placement()
            
            if not placement:
                print(f"[AlternativeSolver] Stuck after placing {len(self.used_pieces)} pieces")
                break
            
            piece, x, y, rotation = placement
            self.place_piece_at(piece, x, y, rotation)
            self.step_count += 1
            
            if self.step_count % 5 == 0:  # Save every 5th step to avoid too many files
                self._save_progress_image(f"Step {self.step_count:03d}")
        
        print(f"[AlternativeSolver] Grid-based solving complete: {len(self.used_pieces)} pieces placed")
        self._save_progress_image("Final")
        return len(self.used_pieces) == len(self.puzzle.pieces_)

    # ======================================================================
    # VISUALIZATION & DEBUG OUTPUT
    # ======================================================================

    def _debug_dir(self) -> str:
        """Return the debug output directory.
        
        Instead of deleting the directory (which can fail on OneDrive/locked files),
        we use a timestamped subdirectory to isolate each run's results.
        """
        import time
        base_d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "debug_output")
        
        # Create a timestamped subdirectory for this run
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_d = os.path.join(base_d, f"alt_solver_{timestamp}")
        
        try:
            os.makedirs(run_d, exist_ok=True)
            print(f"[AlternativeSolver] Debug output: {run_d}")
        except (PermissionError, OSError) as e:
            print(f"[AlternativeSolver] Could not create debug dir, falling back to temp: {e}")
            # Fallback to temp directory if debug_output dir is inaccessible
            run_d = os.environ.get("ZOLVER_TEMP_DIR", ".")
        
        return run_d

    def _get_bbox_all_pieces(self) -> Tuple[int, int, int, int]:
        """Get bounding box of all pieces."""
        bboxes = [p.get_bbox() for p in self.puzzle.pieces_]
        return (
            min(b[0] for b in bboxes),
            min(b[1] for b in bboxes),
            max(b[2] for b in bboxes),
            max(b[3] for b in bboxes),
        )

    def _save_progress_image(self, title: str = ""):
        """Save a progress image showing current piece placement with edge classification."""
        if not self.puzzle.pieces_:
            return
        
        minX, minY, maxX, maxY = self._get_bbox_all_pieces()
        h = maxX - minX + 1
        w = maxY - minY + 1
        
        colored_img = np.zeros((h, w, 3), dtype=np.uint8)
        edge_img = np.zeros((h, w, 3), dtype=np.uint8)
        
        used_piece_set = set(id(p) for i, p in enumerate(self.puzzle.pieces_) if i in self.used_pieces)
        corners = set(id(p) for p in self.detect_corners())
        
        # Draw pieces
        for i, piece in enumerate(self.puzzle.pieces_):
            # Draw colored pixels
            for (px, py), c in piece.pixels.items():
                ix, iy = int(px) - minX, int(py) - minY
                if 0 <= ix < h and 0 <= iy < w:
                    colored_img[ix, iy] = c
            
            # Draw edges with color by type
            for e in piece.edges_:
                for ey, ex in e.shape:
                    ix, iy = int(ex) - minX, int(ey) - minY
                    if 0 <= ix < h and 0 <= iy < w:
                        if e.type == TypeEdge.HOLE:
                            rgb = (255, 178, 102)  # Orange - HOLE
                        elif e.type == TypeEdge.HEAD:
                            rgb = (102, 255, 255)  # Cyan - HEAD
                        elif e.type == TypeEdge.BORDER:
                            rgb = (0, 255, 0)      # Green - BORDER
                        else:
                            rgb = (0, 0, 255)      # Red - UNDEFINED
                        edge_img[ix, iy] = rgb
            
            # Mark corners in magenta, other placed in purple
            for e in piece.edges_:
                if len(e.shape) == 0:
                    continue
                ey, ex = e.shape[0]
                ix, iy = int(ex) - minX, int(ey) - minY
                if 0 <= ix < h and 0 <= iy < w:
                    if id(piece) in used_piece_set:
                        dot_color = (255, 0, 255) if id(piece) in corners else (180, 0, 180)
                        cv2.circle(edge_img, (iy, ix), 5, dot_color, -1)
        
        # Draw legend
        legend = [
            ((0, 255, 0),     "BORDER (flat)"),
            ((255, 178, 102), "HOLE (indent)"),
            ((102, 255, 255), "HEAD (protrus)"),
            ((0, 0, 255),     "UNDEFINED"),
            ((255, 0, 255),   "Corner"),
            ((180, 0, 180),   "Placed"),
        ]
        
        box_size, padding = 14, 5
        font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
        line_h = box_size + padding
        legend_h = len(legend) * line_h + padding
        legend_w = 200
        lx, ly = 8, 8
        
        cv2.rectangle(edge_img, (lx - 2, ly - 2), (lx + legend_w, ly + legend_h), (40, 40, 40), -1)
        
        for i, (color, label) in enumerate(legend):
            y0 = ly + padding + i * line_h
            cv2.rectangle(edge_img, (lx + 4, y0), (lx + 4 + box_size, y0 + box_size), color, -1)
            cv2.putText(edge_img, label, (lx + 4 + box_size + 5, y0 + box_size - 2),
                       font, font_scale, (220, 220, 220), thickness, cv2.LINE_AA)
        
        # Draw title
        cv2.putText(edge_img, title, (10, edge_img.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Determine output directory
        if "Final" in title:
            out_dir = self._debug_dir()
        else:
            out_dir = os.environ.get("ZOLVER_TEMP_DIR", ".")
        
        os.makedirs(out_dir, exist_ok=True)
        
        edge_path = os.path.join(out_dir, f"alt_edges_{self.step_count:03d}.png")
        colored_path = os.path.join(out_dir, f"alt_colored_{self.step_count:03d}.png")
        
        cv2.imwrite(edge_path, edge_img)
        cv2.imwrite(colored_path, colored_img)
        
        print(f"[AlternativeSolver] Saved: {edge_path}")

    def _save_final_report(self):
        """Save classification results and final statistics."""
        classifications = self.classify_pieces_by_type()
        
        report = {
            "total_pieces": len(self.puzzle.pieces_),
            "pieces_placed": len(self.used_pieces),
            "success": len(self.used_pieces) == len(self.puzzle.pieces_),
            "piece_classification": {
                "corners": len(classifications["CORNER"]),
                "borders": len(classifications["BORDER"]),
                "centers": len(classifications["CENTER"]),
            },
            "placements": self.piece_positions,
            "grid_bounds": {
                "min_x": self.min_x,
                "max_x": self.max_x,
                "min_y": self.min_y,
                "max_y": self.max_y,
            }
        }
        
        out_json = os.path.join(os.environ.get("ZOLVER_TEMP_DIR", "."), "alt_results.json")
        import json
        try:
            with open(out_json, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"[AlternativeSolver] Report saved: {out_json}")
        except Exception as e:
            print(f"[AlternativeSolver] Failed to save report: {e}")
        
        self.puzzle.alt_results = report

    # ======================================================================
    # MAIN SOLVING ENTRY POINT
    # ======================================================================

    def solve(self):
        """Main entry point for solving."""
        try:
            print("[AlternativeSolver] Starting improved puzzle solver")
            
            # Analyze and classify pieces
            classifications = self.classify_pieces_by_type()
            print(f"[AlternativeSolver] Piece classification: "
                  f"CORNERS={len(classifications['CORNER'])}, "
                  f"BORDERS={len(classifications['BORDER'])}, "
                  f"CENTERS={len(classifications['CENTER'])}")
            
            # Solve using grid-based approach
            success = self.solve_grid_based()
            
            # Save results
            self._save_final_report()
            self.result_queue.put({
                'success': success,
                'pieces_placed': len(self.used_pieces),
                'placements': self.piece_positions,
            })
            
        except Exception as e:
            print(f"[AlternativeSolver] solve() failed: {e}")
            import traceback
            traceback.print_exc()

    def run(self):
        """Thread run method."""
        self.solve()
