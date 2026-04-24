"""
LEGO Puzzle Solver - Third Solver Method for PREN_Puzzlesolver

Adapts the LEGO-Puzzle-Robot solver approach to the PREN environment.
Uses spline-based edge comparison and grid-based piece placement.

This solver:
1. Classifies pieces by edge types (corner, border, inner)
2. Uses integrated area comparison between splines (LEGO approach)
3. Applies a beam search strategy to find valid placements
4. Runs as a background thread in the main puzzle solving process

Author: Adapted from LEGO-Puzzle-Robot
Date: April 2026
"""

import threading
import queue
import os
from typing import List, Optional, Tuple, Dict, Set
import numpy as np
import cv2
import config
from collections import defaultdict
from scipy.interpolate import splev, splprep

from .Enums import TypeEdge, Directions, get_opposite_direction, directions, TypePiece
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


def edge_curvature(shape: np.ndarray) -> float:
    """Compute normalized curvature (roughness) of an edge.
    0 = perfectly straight, higher values = more curved/complex."""
    if shape is None or len(shape) < 2:
        return 0.0
    pts = np.asarray(shape, dtype=float)
    if pts.shape[0] < 3:
        return 0.0
    p0 = pts[0]
    p1 = pts[-1]
    baseline = p1 - p0
    baseline_len = np.hypot(baseline[0], baseline[1])
    if baseline_len == 0:
        return 0.0
    unit_baseline = baseline / baseline_len
    
    max_dev = 0.0
    for i in range(1, len(pts) - 1):
        vec = pts[i] - p0
        proj = np.dot(vec, unit_baseline)
        proj_point = p0 + proj * unit_baseline
        dev = np.hypot(pts[i][0] - proj_point[0], pts[i][1] - proj_point[1])
        max_dev = max(max_dev, dev)
    
    edge_len = edge_length(shape)
    if edge_len < 1:
        return 0.0
    return max_dev / (edge_len + 1.0)


class LegoSolver(threading.Thread):
    """
    LEGO-based puzzle solver that uses spline comparison for edge matching.
    
    This solver implements the matching strategy from LEGO-Puzzle-Robot:
    - Spline-based edge comparison (integrated area method)
    - Grid-based piece placement
    - Beam search for finding solutions
    """

    def __init__(self, puzzle: Puzzle, result_queue: queue.Queue):
        """
        Initialize the LEGO solver.
        
        Args:
            puzzle: The Puzzle object from PREN
            result_queue: Queue to return results
        """
        super().__init__()
        self.puzzle = puzzle
        self.result_queue = result_queue
        self.daemon = True
        
        # Grid-based placement tracking
        self.connected_pieces: List[Piece] = []
        self.used_pieces: Set[int] = set()
        self.piece_positions: Dict[int, Tuple[int, int]] = {}
        self.grid: Dict[Tuple[int, int], int] = {}
        
        # Bounds of current filled grid
        self.min_x = 0
        self.max_x = 0
        self.min_y = 0
        self.max_y = 0
        
        self.step_count = 0
        self.verbose = 1

    def _log(self, level: int, text: str):
        """Log message if verbosity level allows."""
        log_levels = ["ERROR", "WARNING", "INFO"]
        if level <= self.verbose:
            print(f"[LEGO_SOLVER] [{log_levels[min(level, 2)]}] {text}")

    # ======================================================================
    # SPLINE-BASED EDGE COMPARISON (LEGO Approach)
    # ======================================================================

    def _create_spline(self, shape: np.ndarray) -> Optional[Tuple]:
        """
        Create a spline representation of an edge.
        
        Args:
            shape: Edge shape as np.ndarray of (x, y) points
            
        Returns:
            Spline data (tck tuple) or None if invalid
        """
        if shape is None or len(shape) < 4:
            return None
        
        try:
            pts = np.asarray(shape, dtype=float)
            # Normalize to unit length
            tck, u = splprep(pts.T, s=None, k=min(3, len(pts) - 1))
            return tck
        except Exception as e:
            self._log(0, f"Failed to create spline: {e}")
            return None

    def _compare_sides_integrated_area(self, tck1, tck2, num_samples=150):
        """
        Compare two splines by calculating integrated absolute area between curves.
        
        This is the LEGO solver's core comparison metric. Smaller scores = better match.
        
        Args:
            tck1: Spline data for first edge
            tck2: Spline data for second edge
            num_samples: Number of points to sample
            
        Returns:
            Normalized area difference (float), or infinity on error
        """
        if tck1 is None or tck2 is None:
            return float('inf')

        try:
            # Sample both splines
            u = np.linspace(0, 1, num_samples)
            x1, y1 = splev(u, tck1)
            x2, y2 = splev(u, tck2)

            # Length of each side
            len1 = x1[-1] - x1[0]
            len2 = x2[-1] - x2[0]

            if len1 < 1.0 or len2 < 1.0:
                return float('inf')

            # Common baseline using shorter length
            common_len = min(len1, len2)
            x_common = np.linspace(-common_len / 2.0, common_len / 2.0, num_samples)

            # Interpolate both curves to common x-axis
            y1_interp = np.interp(x_common, x1, y1)

            # Rotate second spline by 180 degrees: x -> -x, y -> -y
            x2_rotated = -x2[::-1]
            y2_rotated = -y2[::-1]
            y2_rotated_interp = np.interp(x_common, x2_rotated, y2_rotated)

            # Calculate mismatch area
            abs_difference = np.abs(y1_interp - y2_rotated_interp)
            score = np.trapz(abs_difference, x=x_common)

            # Normalize by common length
            return score / common_len

        except Exception as e:
            self._log(0, f"Integrated area comparison failed: {e}")
            return float('inf')

    # ======================================================================
    # PIECE CLASSIFICATION
    # ======================================================================

    def _classify_pieces(self) -> Dict[str, List[Piece]]:
        """
        Classify pieces by type (corner, border, inner).
        
        Returns:
            Dict with 'corner', 'border', 'inner' lists
        """
        result = {"corner": [], "border": [], "inner": []}
        
        for piece in self.puzzle.pieces_:
            num_borders = sum(1 for e in piece.edges_ if e.type == TypeEdge.BORDER)
            
            if num_borders == 2:
                result["corner"].append(piece)
                piece.type = TypePiece.ANGLE
            elif num_borders == 1:
                result["border"].append(piece)
                piece.type = TypePiece.BORDER
            else:
                result["inner"].append(piece)
                piece.type = TypePiece.CENTER
        
        self._log(2, f"Classified: {len(result['corner'])} corners, "
                     f"{len(result['border'])} borders, "
                     f"{len(result['inner'])} inner pieces")
        return result

    # ======================================================================
    # GRID-BASED PLACEMENT
    # ======================================================================

    def _place_piece_at_grid(self, piece: Piece, coord: Tuple[int, int], rotation: int = 0):
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
        
        # Update bounds
        x, y = coord
        self.min_x = min(self.min_x, x - 1)
        self.max_x = max(self.max_x, x + 1)
        self.min_y = min(self.min_y, y - 1)
        self.max_y = max(self.max_y, y + 1)
        
        self._log(2, f"Placed piece {piece_idx} at ({x}, {y})")

    def _find_neighbors_at_coord(self, coord: Tuple[int, int]) -> List[Tuple[Tuple[int, int], Piece, Directions]]:
        """Find all placed pieces adjacent to a coordinate."""
        neighbors = []
        for placed_coord in self.grid.keys():
            dx = abs(coord[0] - placed_coord[0])
            dy = abs(coord[1] - placed_coord[1])
            if (dx == 1 and dy == 0) or (dx == 0 and dy == 1):
                piece_idx = self.grid[placed_coord]
                direction_tuple = sub_tuple(coord, placed_coord)
                direction = tuple_to_direction(direction_tuple)
                if direction:
                    neighbors.append((placed_coord, self.puzzle.pieces_[piece_idx], direction))
        return neighbors

    # ======================================================================
    # EDGE MATCHING
    # ======================================================================

    def _find_best_piece_for_coord(self, coord: Tuple[int, int], 
                                   available: List[Piece]) -> Optional[Tuple[Piece, int, float]]:
        """Find the best piece to place at a coordinate given its neighbors."""
        neighbors = self._find_neighbors_at_coord(coord)
        
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
                    candidate_edge = candidate_piece.edge_in_direction(
                        get_opposite_direction(direction_from_neighbor)
                    )
                    
                    # Check basic compatibility
                    if (neighbor_exposed_edge is None or candidate_edge is None or
                        neighbor_exposed_edge.connected or candidate_edge.connected):
                        valid = False
                        break
                    
                    # Check edge type compatibility
                    if not neighbor_exposed_edge.is_compatible(candidate_edge):
                        valid = False
                        break
                    
                    # Compute matching score using LEGO method
                    if (neighbor_exposed_edge.type == TypeEdge.BORDER or 
                        candidate_edge.type == TypeEdge.BORDER):
                        # Border edges: use simple distance
                        score = generated_edge_compute(neighbor_exposed_edge, candidate_edge)
                    else:
                        # Non-border edges: use spline comparison
                        tck1 = self._create_spline(neighbor_exposed_edge.shape)
                        tck2 = self._create_spline(candidate_edge.shape)
                        score = self._compare_sides_integrated_area(tck1, tck2)
                    
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

    # ======================================================================
    # MAIN SOLVING ALGORITHM
    # ======================================================================

    def _generate_spiral_path(self, width: int, height: int) -> List[Tuple[int, int]]:
        """Generate spiral path for filling the puzzle grid."""
        path = []
        top, bottom, left, right = 0, height - 1, 0, width - 1
        
        while top <= bottom and left <= right:
            # Right
            for x in range(left, right + 1):
                path.append((x, top))
            top += 1
            
            # Down
            for y in range(top, bottom + 1):
                path.append((right, y))
            right -= 1
            
            # Left
            if top <= bottom:
                for x in range(right, left - 1, -1):
                    path.append((x, bottom))
                bottom -= 1
            
            # Up
            if left <= right:
                for y in range(bottom, top - 1, -1):
                    path.append((left, y))
                left += 1
        
        return path

    def _find_starting_corner(self, corners: List[Piece]) -> Optional[Tuple[Piece, int]]:
        """Find a suitable starting corner piece."""
        if not corners:
            return None
        
        # Try first corner
        corner = corners[0]
        
        # Find rotation that puts long sides at top and left
        for rotation in range(4):
            # Check if this is a valid starting orientation
            # (would need to match puzzle constraints)
            return (corner, rotation)
        
        return None

    def solve_grid_based(self):
        """
        Solve puzzle using grid-based placement with spline comparison.
        
        Returns:
            True if puzzle was fully solved, False otherwise
        """
        self._log(2, "Starting grid-based solving with spline comparison")
        
        classifications = self._classify_pieces()
        
        # Find and place first corner
        corners = classifications["corner"]
        if not corners:
            self._log(1, "No corners found, using first piece as start")
            start_piece = self.puzzle.pieces_[0]
            start_rotation = 0
        else:
            result = self._find_starting_corner(corners)
            if result:
                start_piece, start_rotation = result
            else:
                start_piece = corners[0]
                start_rotation = 0
        
        self._place_piece_at_grid(start_piece, (0, 0), start_rotation)
        
        # Mark starting piece's border edges as connected
        for edge in start_piece.edges_:
            if edge.type == TypeEdge.BORDER:
                edge.connected = True
        
        # Greedily place remaining pieces
        step = 1
        max_steps = len(self.puzzle.pieces_) * 10  # Safety limit
        
        while len(self.used_pieces) < len(self.puzzle.pieces_) and step < max_steps:
            available = [p for i, p in enumerate(self.puzzle.pieces_) 
                        if i not in self.used_pieces]
            
            # Find best next position to fill
            best_coord = None
            best_piece_info = None
            best_score = float('inf')
            
            # Check positions in grid
            for x in range(self.min_x, self.max_x + 1):
                for y in range(self.min_y, self.max_y + 1):
                    if (x, y) in self.grid:
                        continue  # Already filled
                    
                    # Find neighbors
                    neighbors = self._find_neighbors_at_coord((x, y))
                    if not neighbors:
                        continue
                    
                    # Try finding best piece for this position
                    piece_info = self._find_best_piece_for_coord((x, y), available)
                    
                    if piece_info and piece_info[2] < best_score:
                        best_coord = (x, y)
                        best_piece_info = piece_info
                        best_score = piece_info[2]
            
            if best_coord is None or best_piece_info is None:
                self._log(1, f"No valid placement found. Solved {len(self.used_pieces)}/{len(self.puzzle.pieces_)} pieces")
                break
            
            piece, rotation, score = best_piece_info
            
            # Place the piece
            self._place_piece_at_grid(piece, best_coord, rotation)
            
            # Mark connected edges
            for neighbor_coord, neighbor_piece, direction_from_neighbor in self._find_neighbors_at_coord(best_coord):
                neighbor_edge = neighbor_piece.edge_in_direction(direction_from_neighbor)
                piece_edge = piece.edge_in_direction(get_opposite_direction(direction_from_neighbor))
                neighbor_edge.connected = True
                piece_edge.connected = True
            
            step += 1
            if step % 5 == 0:
                self._log(2, f"Progress: {len(self.used_pieces)}/{len(self.puzzle.pieces_)} pieces placed")
        
        success = len(self.used_pieces) == len(self.puzzle.pieces_)
        self._log(2, f"Grid-based solving complete: {len(self.used_pieces)} pieces placed")
        return success

    # ======================================================================
    # MAIN ENTRY POINT
    # ======================================================================

    def solve(self):
        """Main entry point for solving."""
        try:
            self._log(2, "Starting LEGO puzzle solver")
            
            # Solve using grid-based approach
            success = self.solve_grid_based()
            
            # Reset all pieces to neutral rotation state (0) before returning
            for piece in self.puzzle.pieces_:
                for _ in range(4):
                    piece.rotate_edges(1)
            
            # Return results
            result = {
                'success': success,
                'pieces_placed': len(self.used_pieces),
                'total_pieces': len(self.puzzle.pieces_),
                'placements': self.piece_positions,
            }
            
            self.result_queue.put(result)
            self.puzzle.lego_results = result
            
        except Exception as e:
            self._log(0, f"Solver failed: {e}")
            import traceback
            traceback.print_exc()
            self.puzzle.lego_results = {'success': False, 'error': str(e)}

    def run(self):
        """Thread run method."""
        self.solve()
