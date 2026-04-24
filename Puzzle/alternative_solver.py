import threading
import queue
import math
import os
from typing import List, Optional, Tuple, Dict, Set
import numpy as np
import cv2
import config
from .Enums import TypeEdge, Directions, get_opposite_direction, directions
from .Puzzle import Puzzle
from .PuzzlePiece import PuzzlePiece as Piece
from .Mover import stick_pieces
from .Distance import generated_edge_compute
from .tuple_helper import is_neighbor, sub_tuple, add_tuple


def tuple_to_direction(tup: Tuple[int, int]) -> Optional[Directions]:
    """Convert a tuple like (1, 0) to Directions.E"""
    for d in directions:
        if d.value == tup:
            return d
    return None


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

    def edge_diff(self, edge_connected: 'Edge', candidate_piece: 'Piece', candidate_edge: 'Edge') -> float:
        """Compare edges using the main solver's proven method.
        
        Uses stick_pieces and generated_edge_compute which properly respects
        EDGE_OFFSET from config and provides accurate matching.
        
        Returns: Distance score (smaller = better match)
        """
        # Border edges cannot connect to non-border edges
        if (edge_connected.type == TypeEdge.BORDER) != (candidate_edge.type == TypeEdge.BORDER):
            return float('inf')
        
        # Both border edges: perfect match
        if edge_connected.type == TypeEdge.BORDER and candidate_edge.type == TypeEdge.BORDER:
            return 0.0
        
        # Allow any non-border edges to match (relaxed check)
        # if edge_connected.type == candidate_edge.type:
        #     return float('inf')
        
        try:
            # Backup all edges of candidate piece before alignment
            edge_backups = {}
            for e in candidate_piece.edges_:
                e.backup_shape()
                edge_backups[id(e)] = e.shape.copy() if hasattr(e.shape, 'copy') else e.shape
            
            # Align pieces physically (this is what the main solver does)
            stick_pieces(edge_connected, candidate_piece, candidate_edge)
            
            # Compute the actual distance using generated_edge_compute
            # This respects EDGE_OFFSET from config
            score = generated_edge_compute(edge_connected, candidate_edge)
            
            print(f"[DEBUG] edge_diff: edge types {edge_connected.type} vs {candidate_edge.type}, score {score}")
            
            # Restore all edges to their original state
            for e in candidate_piece.edges_:
                e.restore_backup_shape()
            
            return score
            
        except Exception:
            # Ensure restoration on any error
            try:
                for e in candidate_piece.edges_:
                    e.restore_backup_shape()
            except:
                pass
            return float('inf')

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
                piece.rotate_edges(1)  # Rotate at start of iteration
                
                for candidate_edge in piece.edges_:
                    score = self.edge_diff(edge, piece, candidate_edge)
                    if score < best_score:
                        best_score = score
                        best_piece = piece
                        best_rotation = rotation

            # Reset piece to original rotation (4 rotations = back to original)
            for _ in range(4):
                piece.rotate_edges(1)

        # Accept any match better than infinity (i.e., not rejected for type mismatch)
        if best_piece is not None and best_score < float('inf'):
            return (best_piece, best_rotation, best_score)

        return None

    # ======================================================================
    # NEIGHBOR-BASED PLACEMENT (Main Solver Approach)
    # ======================================================================

    def place_piece_at_grid(self, piece: Piece, coord: Tuple[int, int], rotation: int = 0):
        """Place a piece at grid coordinate."""
        piece_idx = self.puzzle.pieces_.index(piece)
        
        # Apply rotation
        for _ in range(rotation % 4):
            piece.rotate_edges(1)
        
        # Track placement
        self.connected_pieces.append(piece)
        self.used_pieces.add(piece_idx)
        self.piece_positions[piece_idx] = coord
        self.grid[coord] = piece_idx
        
        # Update bounds (expand in open directions)
        x, y = coord
        self.min_x = min(self.min_x, x - 1)
        self.max_x = max(self.max_x, x + 1)
        self.min_y = min(self.min_y, y - 1)
        self.max_y = max(self.max_y, y + 1)
        
        print(f"[AlternativeSolver] Placed piece {piece_idx} at ({x}, {y})")

    def find_neighbors_at_coord(self, coord: Tuple[int, int]) -> List[Tuple[Tuple[int, int], Piece, Directions]]:
        """Find all placed pieces adjacent to a coordinate with their direction.
        Return ALL neighbors - let the matching logic decide which edges are valid."""
        neighbors = []
        # Check if coordinate is adjacent to any placed pieces
        for placed_coord in self.grid.keys():
            # Simple adjacency check: differ by 1 in exactly one dimension
            dx = abs(coord[0] - placed_coord[0])
            dy = abs(coord[1] - placed_coord[1])
            if (dx == 1 and dy == 0) or (dx == 0 and dy == 1):
                piece_idx = self.grid[placed_coord]
                direction_tuple = sub_tuple(coord, placed_coord)
                direction = tuple_to_direction(direction_tuple)
                if direction:
                    neighbors.append((placed_coord, self.puzzle.pieces_[piece_idx], direction))
        return neighbors

    def find_best_piece_for_coord(self, coord: Tuple[int, int], available: List[Piece]) -> Optional[Tuple[Piece, int, float]]:
        """Find the best piece to place at a coordinate given its neighbors."""
        neighbors = self.find_neighbors_at_coord(coord)
        
        if not neighbors:
            return None
        
        best_piece = None
        best_rotation = 0
        best_score = float('inf')
        
        for candidate_piece in available:
            for rotation in range(4):
                candidate_piece.rotate_edges(1)
                
                total_score = 0.0
                valid = True
                
                # Try matching against each neighbor
                for neighbor_coord, neighbor_piece, direction_from_neighbor in neighbors:
                    neighbor_exposed_edge = neighbor_piece.edge_in_direction(direction_from_neighbor)
                    candidate_edge = candidate_piece.edge_in_direction(get_opposite_direction(direction_from_neighbor))
                    
                    # Check compatibility
                    if (neighbor_exposed_edge is None or candidate_edge is None or
                        neighbor_exposed_edge.connected or candidate_edge.connected or
                        not neighbor_exposed_edge.is_compatible(candidate_edge)):
                        valid = False
                        break
                    
                    # Compute matching score
                    score = self.edge_diff(neighbor_exposed_edge, candidate_piece, candidate_edge)
                    if score == float('inf'):
                        valid = False
                        break
                    total_score += score
                
                if valid and total_score < best_score:
                    best_score = total_score
                    best_piece = candidate_piece
                    best_rotation = rotation
            
            # Reset piece to original rotation state (4 rotations = back to start)
            for _ in range(4):
                candidate_piece.rotate_edges(1)
        
        if best_piece is not None:
            return (best_piece, best_rotation, best_score)
        return None

    def solve_grid_based(self):
        """Solve puzzle using simple greedy placement with config.EDGE_OFFSET support.
        
        This is a different approach than the main solver: instead of structure-aware
        (BORDER+FILL), this tries each piece at each position with all rotations,
        using edge matching that respects config.EDGE_OFFSET.
        """
        print("[AlternativeSolver] Starting greedy grid-based solving (different approach)")
        
        # Place first piece at origin
        start_piece = self.puzzle.pieces_[0]
        self.place_piece_at_grid(start_piece, (0, 0), 0)
        self._save_progress_image("Step 001 - Initial piece placed at (0,0)")
        
        # Do not mark border edges as connected - let them connect to other pieces
        # for edge in start_piece.edges_:
        #     if edge.type == TypeEdge.BORDER:
        #         edge.connected = True
        
        # Iteratively place remaining pieces
        step = 2
        max_iterations = len(self.puzzle.pieces_) * len(self.puzzle.pieces_)  # Prevent infinite loops
        iterations = 0
        
        while len(self.used_pieces) < len(self.puzzle.pieces_) and iterations < max_iterations:
            iterations += 1
            available = [p for i, p in enumerate(self.puzzle.pieces_) if i not in self.used_pieces]
            
            best_placement = None
            best_total_score = float('inf')
            
            # Try each available piece
            for candidate_piece in available:
                # Try each rotation
                for rotation in range(4):
                    candidate_piece.rotate_edges(1)
                    
                    # Try each position in expanded grid
                    for x in range(self.min_x - 1, self.max_x + 2):
                        for y in range(self.min_y - 1, self.max_y + 2):
                            if (x, y) in self.grid:
                                continue  # Position already filled
                            
                            # Find neighbors at this position
                            neighbors = self.find_neighbors_at_coord((x, y))
                            if not neighbors:
                                continue  # No neighbors, can't place here
                            
                            # DEBUG: Log neighbor findings
                            print(f"[DEBUG] Position ({x},{y}): found {len(neighbors)} neighbors")
                            
                            print(f"[DEBUG] Trying piece {self.puzzle.pieces_.index(candidate_piece)} at ({x}, {y}) rot {rotation}, neighbors: {len(neighbors)}")
                            
                            # Test compatibility with all neighbors
                            # Key insight: BORDER edges don't need to connect to anything
                            # Only NON-BORDER edges require a valid match
                            total_score = 0.0
                            valid = True
                            
                            for neighbor_coord, neighbor_piece, direction_from_neighbor in neighbors:
                                neighbor_edge = neighbor_piece.edge_in_direction(direction_from_neighbor)
                                candidate_edge = candidate_piece.edge_in_direction(get_opposite_direction(direction_from_neighbor))
                                
                                # Skip if either edge is missing
                                if neighbor_edge is None or candidate_edge is None:
                                    valid = False
                                    break
                                
                                # If neighbor's edge is BORDER, it doesn't need to connect to anything
                                if neighbor_edge.type == TypeEdge.BORDER:
                                    continue
                                
                                # If we reach here, neighbor_edge is NON-BORDER
                                # Both edges must be available (not connected) and compatible
                                if neighbor_edge.connected or candidate_edge.connected:
                                    valid = False
                                    break
                                
                                # If candidate edge is BORDER but neighbor is not: incompatible
                                if candidate_edge.type == TypeEdge.BORDER:
                                    valid = False
                                    break
                                
                                # Both are NON-BORDER: check compatibility and score
                                if not neighbor_edge.is_compatible(candidate_edge):
                                    valid = False
                                    break
                                
                                # Compute edge match score (uses config.EDGE_OFFSET internally)
                                score = self.edge_diff(neighbor_edge, candidate_piece, candidate_edge)
                                if score == float('inf'):
                                    valid = False
                                    break
                                
                                total_score += score
                            
                            # Track best placement found
                            if valid and total_score < best_total_score:
                                best_total_score = total_score
                                best_placement = (candidate_piece, rotation, (x, y), total_score)
                    
                    # Reset to next rotation
                
                # Reset piece to original rotation (4 rotations = back to start)
                for _ in range(4):
                    candidate_piece.rotate_edges(1)
            
            # If we found a valid placement, use it
            if best_placement:
                piece, rotation, coord, score = best_placement
                
                # Apply the rotation that was found to be best
                for _ in range(rotation):
                    piece.rotate_edges(1)
                
                self.place_piece_at_grid(piece, coord, rotation)
                
                # Mark edges as connected
                for neighbor_coord, neighbor_piece, direction_from_neighbor in self.find_neighbors_at_coord(coord):
                    neighbor_edge = neighbor_piece.edge_in_direction(direction_from_neighbor)
                    piece_edge = piece.edge_in_direction(get_opposite_direction(direction_from_neighbor))
                    if neighbor_edge:
                        neighbor_edge.connected = True
                    if piece_edge:
                        piece_edge.connected = True
                
                print(f"[AlternativeSolver] Placed piece at {coord} with score {score:.2f}")
                self._save_progress_image(f"Step {step:03d} - Placed at {coord} (score={score:.2f})")
                step += 1
            else:
                # No valid placement found
                print(f"[AlternativeSolver] No valid placement found. Stuck after {len(self.used_pieces)} pieces.")
                break
        
        success = len(self.used_pieces) == len(self.puzzle.pieces_)
        print(f"[AlternativeSolver] Greedy solving complete: {len(self.used_pieces)}/{len(self.puzzle.pieces_)} pieces placed")
        self._save_progress_image("Final - Greedy solving complete")
        return success

    # ======================================================================
    # GRID-BASED PUZZLE SOLVING (OLD - REPLACED ABOVE)
    # ======================================================================

    # (Removed old solve_grid_based and related methods)

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
            
            # Draw offset contours (tolerance bands) for matching visualization
            if config.EDGE_OFFSET > 0:
                all_edge_pts = np.concatenate(
                    [np.asarray(e.shape, dtype=np.float32)
                     for e in piece.edges_ if len(e.shape) > 0],
                    axis=0,
                ) if any(len(e.shape) > 0 for e in piece.edges_) else None
                
                if all_edge_pts is not None:
                    centroid = tuple(all_edge_pts.mean(axis=0))
                    for e in piece.edges_:
                        if len(e.shape) < 2:
                            continue
                        offset_pts = e.compute_offset_shape(config.EDGE_OFFSET, centroid)
                        for oy, ox in offset_pts:
                            ix, iy = int(ox) - minX, int(oy) - minY
                            if 0 <= ix < h and 0 <= iy < w:
                                edge_img[ix, iy] = (0, 165, 255)  # Orange (BGR) offset contour
            
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
            ((0, 165, 255),   f"OFFSET (+{config.EDGE_OFFSET}px)"),
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

    def _save_final_puzzle_image(self):
        """Generate and save the final assembled puzzle image using grid coordinates."""
        if not self.puzzle.pieces_ or not self.grid:
            print("[AlternativeSolver] No pieces or grid to save")
            return
        
        try:
            # Estimate piece size from a sample piece
            sample_piece = None
            for piece in self.puzzle.pieces_:
                if piece.pixels:
                    sample_piece = piece
                    break
            
            if not sample_piece:
                print("[AlternativeSolver] No piece with pixels found")
                return
            
            px_list = list(sample_piece.pixels.keys())
            px_x = [p[0] for p in px_list]
            px_y = [p[1] for p in px_list]
            piece_height = max(px_x) - min(px_x) + 1
            piece_width = max(px_y) - min(px_y) + 1
            
            # Calculate canvas size
            canvas_height = (self.max_x - self.min_x + 1) * piece_height
            canvas_width = (self.max_y - self.min_y + 1) * piece_width
            
            # Create white canvas
            canvas = np.ones((int(canvas_height), int(canvas_width), 3), dtype=np.uint8) * 255
            
            # Draw each piece at its grid position
            for coord, piece_idx in self.grid.items():
                piece = self.puzzle.pieces_[piece_idx]
                grid_x, grid_y = coord
                
                # Canvas offset for this grid position
                canvas_x_offset = (grid_x - self.min_x) * piece_height
                canvas_y_offset = (grid_y - self.min_y) * piece_width
                
                # Get piece pixel bounds
                if not piece.pixels:
                    continue
                
                piece_px_list = list(piece.pixels.keys())
                piece_min_x = int(min(p[0] for p in piece_px_list))
                piece_min_y = int(min(p[1] for p in piece_px_list))
                
                # Draw piece pixels
                for (px, py), color in piece.pixels.items():
                    canvas_x = canvas_x_offset + int(px) - piece_min_x
                    canvas_y = canvas_y_offset + int(py) - piece_min_y
                    
                    if 0 <= canvas_x < int(canvas_height) and 0 <= canvas_y < int(canvas_width):
                        canvas[int(canvas_x), int(canvas_y)] = color
            
            # Save
            out_dir = os.environ.get("ZOLVER_TEMP_DIR", ".")
            os.makedirs(out_dir, exist_ok=True)
            
            puzzle_path = os.path.join(out_dir, "alt_puzzle_solved.png")
            cv2.imwrite(puzzle_path, canvas)
            print(f"[AlternativeSolver] Final puzzle image saved: {puzzle_path}")
            
        except Exception as e:
            print(f"[AlternativeSolver] Failed to save final puzzle image: {e}")
            import traceback
            traceback.print_exc()

    def _save_final_report(self):
        """Save classification results and final statistics."""
        from .Enums import directions  # Import the directions list
        classifications = self.classify_pieces_by_type()
        
        # Build adjacency information
        adjacencies = {}
        for coord, piece_idx in self.grid.items():
            neighbors = []
            for direction in directions:  # Use actual Directions enum values
                neighbor_coord = add_tuple(coord, direction.value)
                if neighbor_coord in self.grid:
                    neighbor_idx = self.grid[neighbor_coord]
                    neighbors.append({
                        "direction": str(direction),
                        "piece_idx": neighbor_idx,
                        "coordinate": neighbor_coord
                    })
            adjacencies[str(coord)] = {
                "piece_idx": piece_idx,
                "neighbors": neighbors
            }
        
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
            "adjacencies": adjacencies,
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
            
            # Save final puzzle image
            self._save_final_puzzle_image()
            
            # Save results
            self._save_final_report()
            
            # Reset all pieces to neutral rotation state (0) before returning
            for piece in self.puzzle.pieces_:
                for _ in range(4):
                    piece.rotate_edges(1)
            
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
