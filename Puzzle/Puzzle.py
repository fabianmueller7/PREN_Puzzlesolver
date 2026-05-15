import cv2
import numpy as np
import os
import sys
import config
import math
import threading
from .Distance import real_edge_compute, generated_edge_compute
from .Extractor import Extractor, show_image
from .Mover import stick_pieces
from .utils import rotate

from .Enums import (
    Directions,
    Strategy,
    TypePiece,
    TypeEdge,
    get_opposite_direction,
    step_direction,
    rotate_direction,
    directions,
)

from .tuple_helper import (
    equals_tuple,
    add_tuple,
    sub_tuple,
    is_neighbor,
    corner_puzzle_alignment,
    display_dim,
)


class Puzzle:
    """
    Class used to store all informations about the puzzle
    """

    def log(self, *args):
        print(" ".join(map(str, args)))
        if self.viewer:
            self.viewer.addLog(args)   # goes to ViewerProxy.addLog

    def __init__(self, path, viewer=None, green_screen=False):
        """Extract information of pieces in the img at `path` and start computation of the solution"""

        self.pieces_ = None
        factor = 0.40
        while self.pieces_ is None:
            factor += 0.01
            self.extract = Extractor(path, viewer, green_screen, factor)
            self.pieces_ = self.extract.extract()

        # Apply EDGE_OFFSET to piece geometry for mechanical gap compensation
        for piece in self.pieces_:
            piece.apply_edge_offset(config.EDGE_OFFSET)
            piece.backup_initial_state() # Store initial state after offset

        self.border_pieces = [p for p in self.pieces_ if p.is_border]
        self.non_border_pieces = [p for p in self.pieces_ if not p.is_border]
        self.viewer = viewer
        self.green_ = green_screen
        self.connected_directions = []
        self.diff = {}
        self.edge_to_piece = {e: p for p in self.pieces_ for e in p.edges_}

        self.possible_dim = self.compute_possible_size(
            len(self.pieces_), len(self.border_pieces)
        )
        self.extremum = (-1, -1, 1, 1)

    def solve_puzzle(self):
        self.log(">>> START solving puzzle")

        # Identify all possible starting corners (pieces with at least 2 border edges)
        corners = [p for p in self.border_pieces if p.number_of_border() > 1]
        if not corners:
            self.log("No corner pieces found! Trying first border piece as a fallback.")
            corners = self.border_pieces[:1]
        if not corners:
            corners = self.pieces_[:1]

        best_attempt_count = 0

        # Robust solving: Try every corner candidate as the starting piece
        for start_piece in corners:
            # Try all 4 possible rotations for the starting piece
            for rotation in range(4):
                # Reset all pieces to their 'clean' state before each solve attempt
                for piece in self.pieces_:
                    piece.restore_initial_state()
                
                # Reset solver internal state
                self.connected_directions = []
                self.diff = {}
                self.corner_pos = []
                self.extremum = (0, 0, 1, 1)

                # Rotate logical edge directions
                start_piece.rotate_edges(rotation)
                
                # Expansion logic check: Our grid grows towards positive X and Y.
                # Therefore, the start piece at (0,0) must have borders at its South and West edges.
                if not (start_piece.edge_in_direction(Directions.S).connected and 
                        start_piece.edge_in_direction(Directions.W).connected):
                    continue
                
                self.log(f"Attempting solve starting with piece {self.pieces_.index(start_piece)} "
                         f"at rotation {rotation}")

                connected_pieces = [start_piece]
                border_pieces = [p for p in self.border_pieces if p != start_piece]
                non_border_pieces = self.non_border_pieces.copy()
                
                start_piece.coord = (0, 0)
                self.corner_pos = [((0, 0), start_piece)]

                # Solve the border frame first
                self.strategy = Strategy.BORDER
                connected_pieces = self.solve(connected_pieces, border_pieces)
                
                # Fill in the center pieces
                self.strategy = Strategy.FILL
                connected_pieces = self.solve(connected_pieces, non_border_pieces)

                # If we successfully placed all pieces, we are done
                if len(connected_pieces) == len(self.pieces_):
                    self.log(f">>> SUCCESS: All {len(connected_pieces)} pieces placed!")
                    self._finish_and_export()
                    return
                
                if len(connected_pieces) > best_attempt_count:
                    best_attempt_count = len(connected_pieces)
                
                self.log(f"Attempt failed. Placed {len(connected_pieces)}/{len(self.pieces_)} pieces.")

        self.log(f">>> FAILED to solve puzzle completely. Best attempt: {best_attempt_count} pieces.")
        self._finish_and_export()

    def _finish_and_export(self):
        """Final steps to translate, save, and trigger secondary solvers."""
        self.log(">>> FINALIZING result...")
        # Versuche Fehler durch Neu-Platzierung zu beheben
        self._attempt_local_fix(max_attempts=2)
        self.translate_puzzle()
        self.export_pieces(
            os.path.join(os.environ["ZOLVER_TEMP_DIR"], "stick.png"), 
            os.path.join(os.environ["ZOLVER_TEMP_DIR"], "colored.png"), 
            display=False
        )

        # Start the alternative solver concurrently (after main solver completes)
        try:
            from .alternative_solver import AlternativeSolver
            import queue
            alt_queue = queue.Queue()
            alt = AlternativeSolver(self, alt_queue)
            alt.start()
        except Exception:
            self.log("Alt solver failed to start")

        # Start the LEGO solver concurrently (after main solver completes)
        try:
            from .lego_solver import LegoSolver
            import queue
            lego_queue = queue.Queue()
            lego = LegoSolver(self, lego_queue)
            lego.start()
        except Exception:
            self.log("LEGO solver failed to start")

        # Two sets of pieces: Already connected ones and pieces remaining to connect to the others
        # The first piece has an orientation like that:
        #         N          edges:    0
        #      W     E              3     1
        #         S                    2
        #
        # Pieces are placed on a grid like that (X is the first piece at position (0, 0)):
        # +--+--+--+
        # |  |  |  |
        # +--+--+--+
        # |  | X|  |
        # +--+--+--+
        # |  |  |  |
        # +--+--+--+
        #
        # Then if we test the NORTH edge:
        # +--+--+--+
        # |  | X|  |
        # +--+--+--+
        # |  | X|  |
        # +--+--+--+
        # |  |  |  |
        # +--+--+--+
        # Etc until the puzzle is complete i.e. there is no pieces left on left_pieces.

    def get_bbox(self):
        bboxes = [p.get_bbox() for p in self.pieces_]
        return (
            min(bbox[0] for bbox in bboxes),
            min(bbox[1] for bbox in bboxes),
            max(bbox[2] for bbox in bboxes),
            max(bbox[3] for bbox in bboxes),
        )

    def rotate_bbox(self, angle, around):
        # Rotate corners only to optimize
        minX, minY, maxX, maxY = self.get_bbox()
        rotated = [
            rotate((x, y), angle, around) for x in [minX, maxX] for y in [minY, maxY]
        ]
        rotatedX = [p[0] for p in rotated]
        rotatedY = [p[1] for p in rotated]
        return (
            int(min(rotatedX)),
            int(min(rotatedY)),
            int(max(rotatedX)),
            int(max(rotatedY)),
        )

    def solve(self, connected_pieces, left_pieces):
        """
        Solve the puzzle by finding the optimal piece in left_pieces matching the edges
        available in connected_pieces

        :param connected_pieces: pieces already connected to the puzzle
        :param left_pieces: remaining pieces to place in the puzzle
        :param border: Boolean to determine if the strategy is border
        :return: List of connected pieces
        """

        if len(self.connected_directions) == 0:
            self.connected_directions = [
                ((0, 0), connected_pieces[0])
            ]  # ((x, y), p), x & y relative to the first piece, init with 1st piece
            self.diff = self.compute_diffs(
                left_pieces, self.diff, connected_pieces[0]
            )  # edge on the border of the block -> edge on a left piece -> diff between edges
        else:
            self.diff = self.add_to_diffs(left_pieces)

        while len(left_pieces) > 0:
            block_best_e, best_e, best_score = self.best_diff(
                self.diff, self.connected_directions, left_pieces)

            self.log(
                f"<--- New match ---> pieces left: {len(left_pieces)}, Score: {best_score:.2f}, "
                f"extremum: {self.extremum}, puzzle dimension: {display_dim(self.possible_dim)}"
            )

            # Check if match is too poor or not found. 
            # Scores >= 1000 usually indicate type mismatches (e.g. HOLE matching HOLE).
            threshold = getattr(config, 'MATCH_THRESHOLD', 1000.0)
            if block_best_e is None or best_e is None or best_score >= threshold:
                reason = "No match found" if block_best_e is None else f"Best match score {best_score:.2f} exceeds threshold {threshold}"
                self.log(f"{reason} — solver cannot continue on this path")
                break

            # Winkel vom Edge aufs Piece übertragen
            try:
                self.edge_to_piece[best_e].rotation_angle = getattr(best_e, "rotation_angle", 0)
            except Exception:
                pass

            block_best_p, best_p = (
                self.edge_to_piece[block_best_e],
                self.edge_to_piece[best_e],
            )

            if config.EDGE_OFFSET > 0:
                _bloc_pts = [np.asarray(ex.shape, dtype=np.float32)
                             for ex in block_best_p.edges_ if len(ex.shape) > 0]
                _centroid_bloc = tuple(np.concatenate(_bloc_pts, axis=0).mean(axis=0)) if _bloc_pts else None
                _cand_pts = [np.asarray(ex.shape, dtype=np.float32)
                             for ex in best_p.edges_ if len(ex.shape) > 0]
                _centroid_cand = tuple(np.concatenate(_cand_pts, axis=0).mean(axis=0)) if _cand_pts else None
            else:
                _centroid_bloc = _centroid_cand = None
            stick_pieces(block_best_e, best_p, best_e, final_stick=True,
                         centroid_bloc=_centroid_bloc, centroid_cand=_centroid_cand)

            self.update_direction(block_best_e, best_p, best_e)
            self.connect_piece(
                self.connected_directions, block_best_p, block_best_e.direction, best_p
            )

            connected_pieces.append(best_p)
            del left_pieces[left_pieces.index(best_p)]

            self.diff = self.compute_diffs(
                left_pieces, self.diff, best_p, edge_connected=block_best_e
            )

            self.export_pieces(
                os.path.join(os.environ["ZOLVER_TEMP_DIR"], "stick{0:03d}.png".format(len(self.connected_directions))),
                os.path.join(os.environ["ZOLVER_TEMP_DIR"], "colored{0:03d}.png".format(len(self.connected_directions))),
                name_colored="Step {0:03d}".format(len(self.connected_directions)),
            )

        return connected_pieces

    def compute_diffs(self, left_pieces, diff, new_connected, edge_connected=None):
        """
        Compute the diff between the left pieces edges and the new_connected piece edges
        by sticking them and compute the distance

        :param left_pieces: remaining pieces to place in the puzzle
        :param diff: pre computed diff between edges to speed up the process
        :param new_connected: Connected pieces to test for a match
        :return: updated diff matrix
        """

        # Remove former edge from the bloc border
        if edge_connected is not None:
            del diff[edge_connected]

        # build the list of edge to test
        edges_to_test = [
            (piece, edge)
            for piece in left_pieces
            for edge in piece.edges_
            if not edge.connected
        ]

        # Remove the edge of the new piece from the bloc border diffs
        for e in new_connected.edges_:
            for _, v in diff.items():
                if e in v:
                    del v[e]

            if e.connected:
                continue

            # Centroid of the bloc piece — computed once per bloc edge e.
            # Used to determine outward direction for the offset shape.
            if config.EDGE_OFFSET > 0:
                bloc_pts = [np.asarray(ex.shape, dtype=np.float32)
                            for ex in new_connected.edges_ if len(ex.shape) > 0]
                centroid_bloc = tuple(np.concatenate(bloc_pts, axis=0).mean(axis=0)) if bloc_pts else None
            else:
                centroid_bloc = None

            diff_e = {}
            for piece, edge in edges_to_test:
                if not e.is_compatible(edge):
                    continue
                for e2 in piece.edges_:
                    e2.backup_shape()

                # Centroid of the candidate piece before positioning — same
                # coordinate system as edge.shape, so the offset direction is correct.
                if config.EDGE_OFFSET > 0:
                    pre_cand_pts = [np.asarray(ex.shape, dtype=np.float32)
                                    for ex in piece.edges_ if len(ex.shape) > 0]
                    centroid_cand_pre = tuple(np.concatenate(pre_cand_pts, axis=0).mean(axis=0)) if pre_cand_pts else None
                else:
                    centroid_cand_pre = None

                stick_pieces(e, piece, edge, centroid_bloc=centroid_bloc, centroid_cand=centroid_cand_pre)

                # Centroid of the candidate piece after it has been positioned.
                if config.EDGE_OFFSET > 0:
                    cand_pts = [np.asarray(ex.shape, dtype=np.float32)
                                for ex in piece.edges_ if len(ex.shape) > 0]
                    centroid_cand = tuple(np.concatenate(cand_pts, axis=0).mean(axis=0)) if cand_pts else None
                else:
                    centroid_cand = None

                if self.green_:
                    diff_e[edge] = real_edge_compute(edge, e, centroid_cand, centroid_bloc)
                else:
                    diff_e[edge] = generated_edge_compute(edge, e, centroid_cand, centroid_bloc)
                for e2 in piece.edges_:
                    e2.restore_backup_shape()

            diff[e] = diff_e
        return diff

    def fallback(self, diff, connected_direction, left_piece, strat=Strategy.NAIVE):
        """If a strategy does not work fallback to another one"""

        self.log(
            "Fail to solve the puzzle with", self.strategy, "falling back to", strat
        )
        old_strat = self.strategy
        self.strategy = Strategy.NAIVE
        best_bloc_e, best_e, best_score = self.best_diff(diff, connected_direction, left_piece)
        self.strategy = old_strat
        return best_bloc_e, best_e, best_score

    def best_diff(self, diff, connected_direction, left_piece):
        """
        Find the best matching edge for a piece edge

        :param diff: pre computed diff between edges to speed up the process
        :param connected_direction: Direction of the edge to connect
        :param left_piece: Piece to connect
        :return: the best edge found in the bloc
        """

        best_bloc_e, best_e, _best_p, min_diff = None, None, None, float("inf")
        minX, minY, maxX, maxY = self.extremum

        if self.strategy == Strategy.FILL:
            best_coords = []

            # this is ugly
            for i in range(4, -1, -1):  # 4 to 0
                best_coord = []
                for x in range(minX, maxX + 1):
                    for y in range(minY, maxY + 1):
                        # Skip if coordinate is already occupied to prevent overlaps
                        if any(equals_tuple((x, y), c) for c, _ in connected_direction):
                            continue
                        neighbor = list(
                            filter(
                                lambda e: is_neighbor(
                                    (x, y), e[0], connected_direction
                                ),
                                connected_direction,
                            )
                        )
                        if len(neighbor) == i:
                            best_coord.append(((x, y), neighbor))
                best_coords.append(best_coord)

            for best_coord in best_coords:
                for c, neighbor in best_coord:
                    for p in left_piece:
                        for rotation in range(4):
                            diff_score = 0
                            p.rotate_edges(1)
                            last_test = None, None
                            for block_c, block_p in neighbor:
                                direction_exposed = Directions(sub_tuple(c, block_c))
                                edge_exposed = block_p.edge_in_direction(
                                    direction_exposed
                                )
                                edge = p.edge_in_direction(
                                    get_opposite_direction(direction_exposed)
                                )
                                if (
                                    edge_exposed.connected
                                    or edge.connected
                                    or not edge.is_compatible(edge_exposed)
                                ):
                                    diff_score = float("inf")
                                    break
                                else:
                                    diff_score += diff.get(edge_exposed, {}).get(edge, float("inf"))
                                    if diff_score == float("inf"):
                                        break
                                    last_test = edge_exposed, edge
                        
                        # Heuristic: Penalty for matching corners to non-corner positions
                        if p.type == TypePiece.ANGLE and i < 2:
                            diff_score += 500 

                        if diff_score < min_diff:
                            best_bloc_e, best_e, min_diff = (
                                last_test[0],
                                last_test[1],
                                diff_score,
                            )
                if best_e is not None:
                    break
                elif len(best_coord):
                    self.log("Fall back to a worst", self.strategy)
            if best_e is None:
                return self.fallback(
                    diff, connected_direction, left_piece
                )
            return best_bloc_e, best_e, min_diff

        elif self.strategy == Strategy.BORDER:
            best_coord = []
            for x in range(minX, maxX + 1):
                for y in range(minY, maxY + 1):
                    # Skip if coordinate is already occupied to prevent overlaps
                    if any(equals_tuple((x, y), c) for c, _ in connected_direction):
                        continue
                    neighbor = list(
                        filter(
                            lambda e: is_neighbor((x, y), e[0], connected_direction),
                            connected_direction,
                        )
                    )
                    if len(neighbor) == 1:
                        best_coord.append(((x, y), neighbor[0]))
                    elif len(neighbor) >= 2 and len(left_piece) == 1:
                        # Last remaining piece: position may be surrounded by 2 or 3
                        # already-placed neighbours (e.g. the centre edge piece in a 2×3
                        # grid).  Try every neighbour so that is_border_aligned can find
                        # a valid anchor regardless of list order.
                        for n in neighbor:
                            best_coord.append(((x, y), n))

            for c, neighbor in best_coord:
                for p in left_piece:
                    for rotation in range(4):
                        diff_score = 0
                        p.rotate_edges(1)
                        block_c, block_p = neighbor

                        direction_exposed = Directions(sub_tuple(c, block_c))
                        edge_exposed = block_p.edge_in_direction(direction_exposed)
                        edge = p.edge_in_direction(
                            get_opposite_direction(direction_exposed)
                        )

                        if p.type == TypePiece.ANGLE and (
                            not corner_puzzle_alignment(c, self.corner_pos)
                            or not self.corner_place_fit_size(c)
                        ):
                            diff_score = float("inf")
                        if p.type == TypePiece.BORDER and self.is_edge_at_corner_place(
                            c
                        ):
                            diff_score = float("inf")
                        if (
                            diff_score != 0
                            or edge_exposed.connected
                            or edge.connected
                            or not edge.is_compatible(edge_exposed)
                            or not p.is_border_aligned(block_p)
                        ):
                            diff_score = float("inf")
                        else:
                            diff_score = diff.get(edge_exposed, {}).get(edge, float("inf"))

                        if diff_score < min_diff:
                            best_bloc_e, best_e, min_diff = (
                                edge_exposed,
                                edge,
                                diff_score,
                            )
            if best_e is None:
                return self.fallback(
                    diff, connected_direction, left_piece, strat=Strategy.FILL
                )
            return best_bloc_e, best_e, min_diff

        elif self.strategy == Strategy.NAIVE:
            for block_e, block_e_diff in diff.items():
                for e, diff_score in block_e_diff.items():
                    if diff_score < min_diff:
                        best_bloc_e, best_e, min_diff = block_e, e, diff_score
            return best_bloc_e, best_e, min_diff
        return None, None, float("inf")

    def add_to_diffs(self, left_pieces):
        """Build the list of edge to test."""
        edges_to_test = [
            (piece, edge)
            for piece in left_pieces
            for edge in piece.edges_
            if not edge.connected
        ]

        for e, diff_e in self.diff.items():
            for piece, edge in edges_to_test:
                if not e.is_compatible(edge):
                    continue
                for e2 in piece.edges_:
                    e2.backup_shape()
                stick_pieces(e, piece, edge)
                
                # Berechne den Corner-Gap (Abstand der Eckpunkte nach dem Ausrichten)
                # Ein hoher Gap deutet auf eine falsche Platzierung hin, selbst wenn die Nase passt.
                gap_dist = np.linalg.norm(np.array(e.shape[-1]) - np.array(edge.shape[0]))

                if self.green_:
                    score = real_edge_compute(edge, e)
                else:
                    score = generated_edge_compute(edge, e)
                
                # Bestrafung für große Lücken an den Ecken
                # Wenn die Lücke > 15px ist, wird der Score massiv verschlechtert
                if gap_dist > 15.0:
                    score += gap_dist * 5.0
                
                diff_e[edge] = score
                for e2 in piece.edges_:
                    e2.restore_backup_shape()

        return self.diff

    def update_direction(self, e, best_p, best_e):
        """Update the direction of the edge after matching it"""

        opp = get_opposite_direction(e.direction)
        step = step_direction(opp, best_e.direction)
        for edge in best_p.edges_:
            edge.direction = rotate_direction(edge.direction, step)

    def connect_piece(self, connected_directions, curr_p, dir, best_p):
        """
        Then we need to search the other pieces already in the puzzle that are going to be also connected:
        +--+--+--+
        |  | X| O|
        +--+--+--+
        |  | X| X|
        +--+--+--+
        |  |  |  |
        +--+--+--+

        For example if I am going to put a piece at the marker 'O' only one edge will be connected to the piece
        therefore we need to search the adjacent pieces and connect them properly
        """

        old_coord = list(filter(lambda x: x[1] == curr_p, connected_directions))[0][0]
        new_coord = add_tuple(old_coord, dir.value)

        for coord, p in connected_directions:
            for d in directions:
                if equals_tuple(coord, add_tuple(new_coord, d.value)):
                    for edge in best_p.edges_:
                        if edge.direction == d:
                            edge.connected = True
                            break
                    for edge in p.edges_:
                        if edge.direction == get_opposite_direction(d):
                            edge.connected = True
                            break
        connected_directions.append((new_coord, best_p))

        minX, minY, maxX, maxY = self.extremum
        coeff = [1, 1, 1, 1]
        for i, d in enumerate(directions):
            if best_p.edge_in_direction(d).connected:
                coeff[i] = 0
        self.extremum = (
            min(minX, new_coord[0] - coeff[3]),
            min(minY, new_coord[1] - coeff[2]),
            max(maxX, new_coord[0] + coeff[1]),
            max(maxY, new_coord[1] + coeff[0]),
        )

        if best_p.type == TypePiece.ANGLE:
            self.corner_place_fit_size(new_coord, update_dim=True)
            self.corner_pos.append((new_coord, best_p))
        else:
            self.update_dimension()

        best_p.coord = (new_coord[1], new_coord[0])
        self.log("Placed:", best_p.type, "at", best_p.coord)

    def _evaluate_final_edge_scores(self):
        """
        Re-evaluates and logs the match scores for all connected edges
        after the puzzle has been solved. This helps in diagnosing
        sub-optimal placements.
        """
        self.log("\n>>> Evaluating Final Edge Match Scores...")
        evaluated_pairs = set()  # To avoid evaluating the same pair twice (e.g., A->B and B->A)
        piece_quality = {} # piece_id -> list of scores
        piece_corner_err = {} # piece_id -> list of corner distances
        
        # Dictionary to return for fix logic
        piece_errors = {} 

        # Create a mapping from coordinates to pieces for efficient lookup
        coord_to_piece_map = {coord: piece for coord, piece in self.connected_directions}

        for (coord_p1, p1) in self.connected_directions:
            # Get a unique identifier for p1, e.g., its index in the original pieces list
            p1_id = self.pieces_.index(p1) if p1 in self.pieces_ else f"UnknownPiece@{id(p1)}"
            if p1_id not in piece_quality:
                piece_quality[p1_id] = []
                piece_corner_err[p1_id] = []

            for e1 in p1.edges_:
                if e1.connected:
                    # Determine the expected coordinate of the neighbor
                    neighbor_coord = add_tuple(coord_p1, e1.direction.value)
                    
                    # Find the neighbor piece using the map
                    p2 = coord_to_piece_map.get(neighbor_coord)
                    
                    if p2 is None:
                        self.log(f"  Warning: Connected edge of Piece {p1_id} ({coord_p1}) {e1.direction.name} has no found neighbor at {neighbor_coord}")
                        continue

                    # Get a unique identifier for p2
                    p2_id = self.pieces_.index(p2) if p2 in self.pieces_ else f"UnknownPiece@{id(p2)}"
                    if p2_id not in piece_quality:
                        piece_quality[p2_id] = []
                        piece_corner_err[p2_id] = []

                    # Find the corresponding edge on the neighbor piece
                    e2 = p2.edge_in_direction(get_opposite_direction(e1.direction))
                    
                    # Ensure we don't evaluate the same pair twice (e.g., P1-N <-> P2-S and P2-S <-> P1-N)
                    pair_key = tuple(sorted(((p1_id, e1.direction.name), (p2_id, e2.direction.name))))
                    if pair_key in evaluated_pairs:
                        continue
                    evaluated_pairs.add(pair_key)

                    score = real_edge_compute(e1, e2) if self.green_ else generated_edge_compute(e1, e2)

                    # Berechne die Distanz zwischen den korrespondierenden Eckpunkten
                    # Da Kanten "face-to-face" liegen, trifft e1[Anfang] auf e2[Ende] und vice versa
                    dist_a = np.linalg.norm(np.array(e1.shape[0]) - np.array(e2.shape[-1]))
                    dist_b = np.linalg.norm(np.array(e1.shape[-1]) - np.array(e2.shape[0]))
                    avg_corner_dist = (dist_a + dist_b) / 2.0

                    piece_quality[p1_id].append(score)
                    piece_quality[p2_id].append(score)
                    piece_corner_err[p1_id].append(avg_corner_dist)
                    piece_corner_err[p2_id].append(avg_corner_dist)

                    self.log(f"  Piece {p1_id} ({coord_p1}) {e1.direction.name} <-> Piece {p2_id} ({neighbor_coord}) {e2.direction.name}: Score = {score:.2f}, Corner-Gap = {avg_corner_dist:.2f}px")

        # Output a summary per piece to identify the 'bad' ones
        self.log("\n>>> Quality Summary by Piece:")
        threshold = getattr(config, 'MATCH_THRESHOLD', 600.0)
        max_allowed_gap = getattr(config, 'MAX_CORNER_GAP', 30.0)
        suspicious = []
        for p_id, scores in piece_quality.items():
            if not scores: continue
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            avg_corner = sum(piece_corner_err[p_id]) / len(piece_corner_err[p_id]) if piece_corner_err[p_id] else 0.0
            
            # Ein Teil ist verdächtig, wenn der Score zu hoch ODER die Ecken zu weit auseinander sind
            is_bad = max_score >= threshold or avg_corner > max_allowed_gap
            status = "OK" if not is_bad else "BAD MATCH ❌"
            
            if status != "OK": suspicious.append(p_id)
            
            piece_errors[p_id] = {'max_score': max_score, 'avg_corner_err': avg_corner}
            self.log(f"  Piece {p_id:2}: Score-Max={max_score:6.2f}, Corner-Err-Avg={avg_corner:5.2f}px -> {status}")
        
        if suspicious:
            self.log(f"\nPotential errors detected in pieces: {suspicious}")

        self.log(">>> End Final Edge Match Scores\n")
        return suspicious, piece_errors

    def _unplace_piece(self, p_id):
        """Entnimmt ein Teil aus der Platzierungsliste und setzt seine Verbindungen zurück."""
        piece_to_remove = self.pieces_[p_id]
        
        removed_coord = None
        for coord, p in self.connected_directions:
            if p == piece_to_remove:
                removed_coord = coord
                break

        self.log(f"  Unplacing Piece {p_id} at {removed_coord}...")
        
        # Aus connected_directions entfernen
        self.connected_directions = [item for item in self.connected_directions if item[1] != piece_to_remove]
        
        # Kanten-Verbindungen kappen (beim Teil selbst und bei seinen Nachbarn)
        for edge in piece_to_remove.edges_:
            edge.connected = False
        
        # Kanten-Verbindungen bei den Nachbarn resetten, damit die Slots wieder verfügbar sind
        if removed_coord is not None:
            for coord, p in self.connected_directions:
                for e in p.edges_:
                    neighbor_pos = add_tuple(coord, e.direction.value)
                    if equals_tuple(neighbor_pos, removed_coord):
                        e.connected = False
        
        piece_to_remove.restore_initial_state()
        return piece_to_remove

    def _attempt_local_fix(self, max_attempts=1):
        """Identifiziert die zwei schlechtesten Teile und versucht sie neu zu platzieren."""
        for attempt in range(max_attempts):
            suspicious_ids, errors = self._evaluate_final_edge_scores()
            if not suspicious_ids:
                break
            
            # Sortiere nach dem höchsten Corner-Error
            sorted_suspicious = sorted(suspicious_ids, key=lambda x: errors[x]['avg_corner_err'], reverse=True)
            
            # Wenn Piece 0 (Startstück) betroffen ist, brechen wir den ganzen Solve-Versuch ab,
            # da das Fundament falsch ist. solve_puzzle wird dann die nächste Rotation/Ecke probieren.
            if 0 in sorted_suspicious and errors[0]['avg_corner_err'] > 20.0:
                self.log("[LOCAL FIX] Start piece (0) is suspicious. Aborting this solve attempt to try next start config.")
                return

            # Ansonsten versuchen wir die bis zu 3 schlechtesten Teile (außer Startstück) zu fixen
            to_fix_ids = [pid for pid in sorted_suspicious if pid != 0][:3]
            
            self.log(f"\n[LOCAL FIX] Attempt {attempt+1}: Targeting pieces {to_fix_ids} for re-placement.")
            
            # Backup des aktuellen Zustands für Revert
            backup_pieces_state = [p.get_current_state() for p in self.pieces_]
            backup_directions = list(self.connected_directions)
            backup_extremum = self.extremum
            backup_corner_pos = list(self.corner_pos)
            backup_diff = {k: v.copy() for k, v in self.diff.items()}
            old_strategy = self.strategy
            
            # Beide Teile entfernen
            removed_pieces = []
            for p_id in to_fix_ids:
                removed_pieces.append(self._unplace_piece(p_id))
            
            # Diffs für die verbleibenden Teile und die neu verfügbaren Slots initialisieren
            self.diff = {}
            for _, p in self.connected_directions:
                self.diff = self.compute_diffs(removed_pieces, self.diff, p)
            
            # Versuche Neu-Platzierung mit der Standard-Solve-Routine
            self.strategy = Strategy.FILL
            self.solve([item[1] for item in self.connected_directions], removed_pieces)
            
            # Re-evaluierung
            new_suspicious, new_errors = self._evaluate_final_edge_scores()
            
            # Prüfe, ob die Lösung jetzt besser ist (alle Teile platziert und weniger/kleinere Fehler)
            if len(self.connected_directions) == len(self.pieces_) and len(new_suspicious) < len(suspicious_ids):
                self.log(f"  [LOCAL FIX] Success: Swapped pieces, suspicious count reduced from {len(suspicious_ids)} to {len(new_suspicious)}.")
                self.strategy = old_strategy
            else:
                self.log(f"  [LOCAL FIX] No improvement. Reverting to previous state.")
                self.connected_directions = backup_directions
                for i, p in enumerate(self.pieces_):
                    p.set_state(backup_pieces_state[i])
                self.extremum = backup_extremum
                self.corner_pos = backup_corner_pos
                self.diff = backup_diff
                self.strategy = old_strategy
                break

    def translate_puzzle(self):
        """Translate all pieces to the top left corner to be sure the puzzle is in the image"""

        # Find minimum y (min_y) and minimum x (min_x) across all edges.
        # pixel[0] is y, pixel[1] is x.
        min_y = sys.maxsize
        min_x = sys.maxsize
        for p in self.pieces_:
            for e in p.edges_:
                for pixel in e.shape:
                    if pixel[0] < min_y:
                        min_y = pixel[0]
                    if pixel[1] < min_x:
                        min_x = pixel[1]

        # Use integer translation to avoid float keys in PuzzlePiece.pixels
        ty = int(math.floor(min_y))
        tx = int(math.floor(min_x))

        for p in self.pieces_:
            for e in p.edges_:
                for ip, _ in enumerate(e.shape):
                    e.shape[ip] += (-ty, -tx)

        for p in self.pieces_:
            p.translate(-tx, -ty)

    def export_pieces(
        self,
        path_contour,
        path_colored,
        name_contour=None,
        name_colored=None,
        display=True,
        display_border=False,
    ):
        """
        Export the contours and the colored image

        :param path_contour: Path used to export contours
        :param path_colored: Path used to export the colored image
        :return: the best edge found in the bloc
        """

        minX, minY, maxX, maxY = self.get_bbox()
        colored_img = np.zeros((maxX - minX, maxY - minY, 3), dtype=np.uint8)
        border_img = np.zeros((maxX - minX, maxY - minY, 3), dtype=np.uint8)

        for piece in self.pieces_:
            # Reframe piece pixels to (0, 0)
            tmp = [
                (x - minX, y - minY, c)
                for (x, y), c in piece.pixels.items()
                if 0 <= x - minX < colored_img.shape[0]
                and 0 <= y - minY < colored_img.shape[1]
            ]
            x, y, c = (
                list(map(lambda e: int(e[0]), tmp)),
                list(map(lambda e: int(e[1]), tmp)),
                list(map(lambda e: e[2], tmp)),
            )
            colored_img[x, y] = np.array(c, dtype=np.uint8)

            # ---- Rotation des Puzzleteils anwenden (visuell) ----
            if hasattr(piece, "rotation_angle") and piece.rotation_angle != 0:
                h, w = colored_img.shape[:2]
                # Erzeuge eine leere Maske für das aktuelle Teil
                mask = np.zeros_like(colored_img)
                mask[x, y] = c

                # Rotationsmatrix um den Mittelpunkt des Gesamtbilds
                M = cv2.getRotationMatrix2D((w // 2, h // 2), float(piece.rotation_angle), 1.0)
                mask = cv2.warpAffine(mask, M, (w, h),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(0, 0, 0))
                colored_img = np.maximum(colored_img, mask)
            # ------------------------------------------------------


            if config.DEBUG_FILE_OUTPUT == 1:
                # Contours
                for e in piece.edges_:
                    for y_float, x_float in e.shape:
                        y, x = int(y_float - minY), int(x_float - minX)
                        if (
                                0 <= y < border_img.shape[1] # Check bounds with integer y
                                and 0 <= x < border_img.shape[0] # Check bounds with integer x
                        ):
                            rgb = (0, 0, 0)
                            if e.type == TypeEdge.HOLE:
                                rgb = (102, 178, 255)
                            if e.type == TypeEdge.HEAD:
                                rgb = (255, 255, 102)
                            if e.type == TypeEdge.UNDEFINED:
                                rgb = (255, 0, 0)
                            if e.connected:
                                rgb = (0, 255, 0)

                            border_img[x, y, 0] = rgb[2]
                            border_img[x, y, 1] = rgb[1]
                            border_img[x, y, 2] = rgb[0]

                # Draw outer (offset) edge as tolerance band
                if config.EDGE_OFFSET > 0:
                    # Compute centroid directly from edge shape points so the
                    # coordinate system matches edge.shape — avoids any (row,col)
                    # vs (col,row) mismatch that can flip the outward normal.
                    all_edge_pts = np.concatenate(
                        [np.asarray(e.shape, dtype=np.float32)
                         for e in piece.edges_ if len(e.shape) > 0],
                        axis=0,
                    ) if any(len(e.shape) > 0 for e in piece.edges_) else None
                    centroid = tuple(all_edge_pts.mean(axis=0)) if all_edge_pts is not None else None
                    for e in piece.edges_:
                        if len(e.shape) < 2 or centroid is None:
                            continue
                        offset_pts = e.compute_offset_shape(config.EDGE_OFFSET, centroid)
                        pts_img = np.array(
                            [[(int(oy) - minY, int(ox) - minX)] for oy, ox in offset_pts],
                            dtype=np.int32,
                        )
                        cv2.polylines(
                            border_img, [pts_img], isClosed=False,
                            color=(0, 165, 255), thickness=1,  # orange (BGR)
                        )

                # Draw purple dots at piece corners (where edges meet)
                for e in piece.edges_:
                    if len(e.shape) == 0:
                        continue
                    ey, ex = e.shape[0]
                    ix, iy = int(ex) - minX, int(ey) - minY
                    if 0 <= ix < border_img.shape[0] and 0 <= iy < border_img.shape[1]:
                        cv2.circle(border_img, (iy, ix), 4, (128, 0, 128), -1)

                # Draw center of mass marker (cyan crosshair) for each piece
                if len(piece.pixels) > 0:
                    xs = [px for (px, py) in piece.pixels]
                    ys = [py for (px, py) in piece.pixels]
                    com_x = int(np.mean(xs)) - minX
                    com_y = int(np.mean(ys)) - minY
                    if 0 <= com_x < border_img.shape[0] and 0 <= com_y < border_img.shape[1]:
                        cv2.drawMarker(border_img, (com_y, com_x), (0, 255, 255),
                                       cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)

                # Draw color legend
                # Colors are in BGR (OpenCV convention) to match what's drawn in border_img
                legend = [
                    ((0, 255, 0),     "BORDER (flat edge)"),
                    ((255, 178, 102), "HOLE (indentation)"),
                    ((102, 255, 255), "HEAD (protrusion)"),
                    ((0, 0, 255),     "UNDEFINED"),
                    ((0, 165, 255),   f"OUTER EDGE (+{config.EDGE_OFFSET}px offset)"),
                ]
                box_size = 16
                padding = 6
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.45
                font_thickness = 1
                line_h = box_size + padding
                legend_h = len(legend) * line_h + padding
                legend_w = 210
                lx, ly = 10, 10
                cv2.rectangle(border_img, (lx - 2, ly - 2),
                              (lx + legend_w, ly + legend_h), (40, 40, 40), -1)
                for i, (color, label) in enumerate(legend):
                    y0 = ly + padding + i * line_h
                    cv2.rectangle(border_img, (lx + 4, y0),
                                  (lx + 4 + box_size, y0 + box_size), color, -1)
                    cv2.putText(border_img, label,
                                (lx + 4 + box_size + 6, y0 + box_size - 3),
                                font, font_scale, (220, 220, 220), font_thickness, cv2.LINE_AA)

                cv2.imwrite(path_contour, border_img)

                if config.DEBUG_SHOW_DIAGRAMS == 1:
                    show_image(border_img,"contour category image")

            cv2.imwrite(path_colored, colored_img)


    def compute_possible_size(self, nb_piece, nb_border) -> list[tuple]:
        """
        Compute all possible size of the puzzle based on the number
        of pieces and the number of border pieces
        """
        nb_edge_border = nb_border - 4
        nb_middle = nb_piece - nb_border
        possibilities = []
        for i in range(nb_edge_border // 2 + 1):
            w, h = i, (nb_edge_border // 2) - i
            if w * h == nb_middle:
                possibilities.append((w + 1, h + 1))
        self.log(
            "Possible sizes: (",
            nb_piece,
            "pieces with",
            nb_border,
            "borders among them):",
            display_dim(possibilities),
        )
        return possibilities

    def corner_place_fit_size(self, c, update_dim=False):
        """Update the possible dimensions of the puzzle when a corner is placed"""

        def almost_equals(idx, target, val):
            return val[idx] == target or val[idx] == -target

        # We have already picked a dimension
        if len(self.possible_dim) == 1:
            return (
                c[0] == 0
                or c[0] == self.possible_dim[0][0]
                or c[0] == -self.possible_dim[0][0]
            ) and (
                c[1] == 0
                or c[1] == self.possible_dim[0][1]
                or c[0] == -self.possible_dim[0][1]
            )

        if c[0] == 0:
            filtered = list(
                filter(lambda x: almost_equals(1, c[1], x), self.possible_dim)
            )
            if len(filtered):
                if update_dim and len(filtered) != len(self.possible_dim):
                    self.log(
                        "Update possible dimensions with corner place:",
                        display_dim(filtered),
                    )
                    self.possible_dim = filtered
                return True
            else:
                return False
        elif c[1] == 0:
            filtered = list(
                filter(lambda x: almost_equals(0, c[0], x), self.possible_dim)
            )
            if len(filtered):
                if update_dim and len(filtered) != len(self.possible_dim):
                    self.log(
                        "Update possible dimensions with corner place:",
                        display_dim(filtered),
                    )
                    self.possible_dim = filtered
                return True
            else:
                return False
        return False

    def is_edge_at_corner_place(self, c):
        """Determine of an edge is at a corner place"""

        if len(self.possible_dim) == 1:
            # We have already picked a dimension
            return (
                c[0] == 0
                or c[0] == self.possible_dim[0][0]
                or c[0] == -self.possible_dim[0][0]
            ) and (
                c[1] == 0
                or c[1] == self.possible_dim[0][1]
                or c[0] == -self.possible_dim[0][1]
            )
        return False

    def update_dimension(self):
        if len(self.possible_dim) == 1:
            return
        dims = []
        _, _, maxX, maxY = self.extremum
        for x, y in self.possible_dim:
            if maxX <= x and maxY <= y:
                dims.append((x, y))
        if len(dims) != len(self.possible_dim):
            self.log(
                "Update possible dimensions with extremum",
                self.extremum,
                ":",
                display_dim(dims),
            )
            self.possible_dim = dims
