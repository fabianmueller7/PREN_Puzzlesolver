"""
Alternative solver running concurrently with the main solver.

This module implements a lightweight, best-effort algorithm that:
- classifies corner pieces by counting adjacent straight (BORDER) edges,
- measures the length of straight edges,
- attempts a greedy grouping of straight edges by length to discover
  likely side assemblies,
- profiles non-straight edges by computing a simple bulge-distance metric
  relative to the adjacent straight edge.

Results are stored in the parent Puzzle instance under `alt_results` so the
rest of the application can inspect them.

This is intentionally conservative and non-destructive: it reads existing
piece and edge data but does not modify shapes or directions.
"""

from __future__ import annotations


from typing import List, Dict, Tuple

import numpy as np

from .Enums import TypeEdge, rotate_direction, get_opposite_direction, directions
from .tuple_helper import equals_tuple

from itertools import permutations, product


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
    base = p1 - p0
    base_len = np.hypot(base[0], base[1])
    if base_len == 0:
        return 0.0, 0
    # compute perpendicular distances
    t = ((pts - p0) @ base) / (base_len ** 2)
    proj = p0 + np.outer(t, base)
    dev = np.hypot((pts - proj)[:, 0], (pts - proj)[:, 1])
    idx = int(np.argmax(dev))
    return float(dev[idx]), idx


class AlternativeSolver:
    """Runs an alternative heuristic solver that analyzes pieces and edges.

    The instance stores results in `puzzle.alt_results`.
    """

    def __init__(self, puzzle):
        self.puzzle = puzzle
        self.pieces = puzzle.pieces_
        self.results: Dict = {}

    def run(self):
        try:
            self.results['num_pieces'] = len(self.pieces)
            self._classify_corners()
            self._measure_straight_edges()
            self._greedy_group_edges()
            self._profile_non_straight_edges()
            # Attempt a small exhaustive assembly for very small puzzles (e.g., 4 pieces -> 2x2)
            if self.results['num_pieces'] == 4:
                assembled = self._attempt_assemble_2x2()
                self.results['assembly_2x2'] = assembled
            # store results back on puzzle for inspection
            self.puzzle.alt_results = self.results
        except Exception as e:
            # store error for debugging
            self.puzzle.alt_results = {'error': str(e)}

    # ------------------------------------------------------------------
    # Step 1: classify corners
    def _classify_corners(self):
        corners = []
        possible_corners = []
        for idx, p in enumerate(self.pieces):
            border_edges = [e for e in p.edges_ if e.type == TypeEdge.BORDER]
            # count how many adjacent border edges exist
            dirs = [e.direction for e in border_edges]
            # convert directions to indices 0..3 using puzzle Enums ordering
            dir_indices = [self.puzzle.enums_index(d) if hasattr(self.puzzle,'enums_index') else None for d in dirs]
            # fallback: we can compute adjacency by checking direction values
            adj_count = 0
            for e in border_edges:
                # check if there exists another border edge with adjacent direction
                for f in border_edges:
                    if e is f:
                        continue
                    # directions are Enums with values like (0,1); adjacency if not opposite
                    if e.direction != f.direction:
                        # directions orthogonal -> dot == 0
                        dot = e.direction.value[0] * f.direction.value[0] + e.direction.value[1] * f.direction.value[1]
                        if dot == 0:
                            adj_count += 1
                            break
            # each adjacency counted once per edge; corner pieces have at least two adjacent border edges
            if len(border_edges) >= 2 and adj_count >= 2:
                corners.append(idx)
            elif len(border_edges) >= 2:
                possible_corners.append(idx)

        self.results['corner_candidates'] = corners
        self.results['possible_corners'] = possible_corners

    # ------------------------------------------------------------------
    # Step 2: measure straight edges
    def _measure_straight_edges(self):
        straight_edges = []
        for p_idx, p in enumerate(self.pieces):
            for e_idx, e in enumerate(p.edges_):
                if e.type == TypeEdge.BORDER:
                    L = edge_length(e.shape)
                    straight_edges.append({'piece': p_idx, 'edge_idx': e_idx, 'length': L, 'direction': e.direction})
        # sort by length descending
        straight_edges.sort(key=lambda x: x['length'], reverse=True)
        self.results['straight_edges'] = straight_edges

    # ------------------------------------------------------------------
    # Step 3: simple greedy grouping of straight edges by similar length
    def _greedy_group_edges(self):
        edges = self.results.get('straight_edges', [])
        lengths = [e['length'] for e in edges]
        if not lengths:
            self.results['groups'] = []
            return
        max_len = max(lengths)
        tol = max(5.0, max_len * 0.05)  # 5 pixels or 5% tolerance

        groups: List[List[dict]] = []
        used = [False] * len(edges)
        for i, e in enumerate(edges):
            if used[i]:
                continue
            group = [e]
            used[i] = True
            for j in range(i + 1, len(edges)):
                if used[j]:
                    continue
                if abs(edges[j]['length'] - e['length']) <= tol:
                    group.append(edges[j])
                    used[j] = True
            groups.append(group)

        # compute group summaries
        group_summaries = []
        for g in groups:
            avg = sum(x['length'] for x in g) / len(g)
            group_summaries.append({'count': len(g), 'avg_length': avg, 'members': g})

        self.results['groups'] = group_summaries

    # ------------------------------------------------------------------
    # Step 4: profile non-straight edges
    def _profile_non_straight_edges(self):
        profiles = []
        for p_idx, p in enumerate(self.pieces):
            for e_idx, e in enumerate(p.edges_):
                if e.type == TypeEdge.BORDER:
                    continue
                # try to find adjacent straight edge (orthogonal direction)
                adj = None
                for adj_edge in p.edges_:
                    if adj_edge is e:
                        continue
                    if adj_edge.type == TypeEdge.BORDER:
                        # orthogonal check
                        dot = e.direction.value[0] * adj_edge.direction.value[0] + e.direction.value[1] * adj_edge.direction.value[1]
                        if dot == 0:
                            adj = adj_edge
                            break

                # measure max deviation along this edge
                dev, idx_dev = max_deviation_from_line(e.shape)
                dist_from_start = 0.0
                if idx_dev > 0:
                    # compute distance along contour from start to idx_dev
                    pts = np.asarray(e.shape, dtype=float)
                    d = np.hypot(np.diff(pts[: idx_dev + 1, 0]), np.diff(pts[: idx_dev + 1, 1]))
                    dist_from_start = float(d.sum())

                profiles.append(
                    {
                        'piece': p_idx,
                        'edge_idx': e_idx,
                        'max_dev': dev,
                        'idx_dev': idx_dev,
                        'dist_from_start': dist_from_start,
                        'has_adjacent_border': adj is not None,
                        'adjacent_border_dir': getattr(adj, 'direction', None),
                    }
                )

        self.results['non_straight_profiles'] = profiles

    # ------------------------------------------------------------------
    # Small exhaustive assembly for 2x2 puzzles (num_pieces == 4)
    def _attempt_assemble_2x2(self):
        """Try all permutations and rotations to place 4 pieces into a 2x2 grid.

        Returns a dict with placement mapping or None if no valid assembly found.
        Placement format: { (x,y): {'piece': idx, 'rotation': r_steps} }
        rotation steps are integers 0..3 (clockwise 90° per step)
        """
        pieces = self.pieces
        tol_len = 10.0  # pixel tolerance for matching inner edge lengths

        # Helper: get edge for a piece at global direction after rotating piece by r steps
        def edge_after_rotation(piece, r_steps, global_dir):
            # Find which original edge aligns to global_dir after rotating piece clockwise r_steps
            orig_dir = rotate_direction(global_dir, -r_steps)
            return piece.edge_in_direction(orig_dir)

        positions = [(0, 0), (1, 0), (0, 1), (1, 1)]  # TL, TR, BL, BR

        for perm in permutations(range(len(pieces))):
            # perm assigns pieces to positions in order
            for rots in product(range(4), repeat=4):
                placement = {}
                ok = True
                for pos_idx, pos in enumerate(positions):
                    p_idx = perm[pos_idx]
                    r = rots[pos_idx]
                    placement[pos] = {'piece': p_idx, 'rot': r}

                # Check border constraints: corners must have BORDER edges outward
                # define required outward border directions per position
                req = {
                    (0, 0): [directions[0], directions[3]],  # N, W
                    (1, 0): [directions[0], directions[1]],  # N, E
                    (0, 1): [directions[2], directions[3]],  # S, W
                    (1, 1): [directions[2], directions[1]],  # S, E
                }
                for pos, info in placement.items():
                    piece = pieces[info['piece']]
                    r = info['rot']
                    for outward_dir in req[pos]:
                        e = edge_after_rotation(piece, r, outward_dir)
                        if e is None or e.type != TypeEdge.BORDER:
                            ok = False
                            break
                    if not ok:
                        break

                if not ok:
                    continue

                # Check inner adjacency compatibility and lengths
                # Positions adjacency pairs: (0,0)-(1,0) horizontal top, (0,1)-(1,1) bottom, (0,0)-(0,1) left vertical, (1,0)-(1,1) right vertical
                adj_checks = [
                    ((0, 0), (1, 0), directions[1]),
                    ((0, 1), (1, 1), directions[1]),
                    ((0, 0), (0, 1), directions[2]),
                    ((1, 0), (1, 1), directions[2]),
                ]
                for a, b, dir_from_a_to_b in adj_checks:
                    pa = placement[a]
                    pb = placement[b]
                    piece_a = pieces[pa['piece']]
                    piece_b = pieces[pb['piece']]
                    ra = pa['rot']
                    rb = pb['rot']
                    edge_a = edge_after_rotation(piece_a, ra, dir_from_a_to_b)
                    edge_b = edge_after_rotation(piece_b, rb, get_opposite_direction(dir_from_a_to_b))
                    if edge_a is None or edge_b is None:
                        ok = False
                        break
                    # Types must be compatible (hole/head)
                    if not edge_a.is_compatible(edge_b):
                        ok = False
                        break
                    # Lengths should be similar
                    la = edge_length(edge_a.shape)
                    lb = edge_length(edge_b.shape)
                    if abs(la - lb) > tol_len:
                        ok = False
                        break
                if not ok:
                    continue

                # If reached here, we found a consistent assembly
                return {'placement': placement}

        return None