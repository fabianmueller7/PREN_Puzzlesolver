import cv2
import numpy as np
import os
from .. import config
import math

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

    def log(self, *args):
        msg = " ".join(map(str, args))
        print(msg)
        if self.viewer:
            self.viewer.addLog(msg)

    @staticmethod
    def _piece_centroid(piece):
        pts = [np.asarray(e.shape, dtype=np.float32) for e in piece.edges_ if len(e.shape) > 0]
        return tuple(np.concatenate(pts, axis=0).mean(axis=0)) if pts else None

    def __init__(self, path, viewer=None, green_screen=False):
        self.pieces_ = None
        factor = 0.40
        while self.pieces_ is None:
            factor += 0.01
            self.extract = Extractor(path, viewer, green_screen, factor)
            self.pieces_ = self.extract.extract()

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

        if not self.pieces_:
            self.log("ERROR: No pieces detected — corner detection failed for all contours.")
            self.log("Try placing pieces so they don't touch each other, then re-scan.")
            return

        if config.DEBUG_PIECE_CENTERS == 1:
            _start_centers = {
                id(p): p.img_centroid if p.img_centroid is not None else self._piece_centroid(p)
                for p in self.pieces_
            }
            # Capture ALL edge points for each piece BEFORE solving
            # (shapes are still in original image coordinates).
            # Stored as a single (N, 2) array so the rotation estimator can use
            # all points for a robust circular-mean angle measurement.
            _start_ref_pts = {}
            for p in self.pieces_:
                sc = _start_centers[id(p)]
                all_pts = [np.asarray(e.shape, dtype=float) for e in p.edges_ if len(e.shape) > 0]
                if all_pts:
                    _start_ref_pts[id(p)] = (sc, np.concatenate(all_pts, axis=0))

            # Per-edge shape and direction snapshots for border-alignment rotation calculation.
            # Both must be captured together: the start-piece rotation loop (lines 120-133)
            # mutates e.direction AND e.shape, so we need their pre-rotation state.
            _start_edge_shapes = {
                id(p): {id(e): np.asarray(e.shape, dtype=float).copy() for e in p.edges_}
                for p in self.pieces_
            }
            _start_edge_dirs = {
                id(p): {id(e): e.direction for e in p.edges_}
                for p in self.pieces_
            }

        connected_pieces = []
        border_pieces = self.border_pieces.copy()
        non_border_pieces = self.non_border_pieces.copy()

        for piece in border_pieces:
            if piece.number_of_border() > 1:
                connected_pieces.append(piece)
                border_pieces.remove(piece)
                break

        self.log("Number of border pieces: ", len(border_pieces) + 1)

        self.export_pieces(
            os.path.join(os.environ["ZOLVER_TEMP_DIR"], "stick{0:03d}.png".format(1)),
            "Border types",
            display_border=True,
        )

        self.log(">>> START solve border")
        start_piece = connected_pieces[0]
        start_piece.coord = (0, 0)
        self.corner_pos = [((0, 0), start_piece)]

        for i in range(4):
            if (
                start_piece.edge_in_direction(Directions.S).connected
                and start_piece.edge_in_direction(Directions.W).connected
            ):
                break
            start_piece.rotate_edges(1)
            angle = -(math.pi / 2)
            center = start_piece.get_center()          # (center_row, center_col) — pixel space
            edge_center = (center[1], center[0])       # (center_col, center_row) — edge shape space
            for edge in start_piece.edges_:
                for idx, pt in enumerate(edge.shape):
                    edge.shape[idx] = rotate(pt, angle, edge_center)

        self.extremum = (0, 0, 1, 1)
        self.strategy = Strategy.BORDER
        connected_pieces = self.solve(connected_pieces, border_pieces)
        self.log(">>> START solve middle")
        self.strategy = Strategy.FILL
        self.solve(connected_pieces, non_border_pieces)

        self.log(">>> SAVING result...")
        self.translate_puzzle()
        self.export_pieces(
            os.path.join(os.environ["ZOLVER_TEMP_DIR"], "stick.png"),
            display=False,
        )

        if config.DEBUG_PIECE_CENTERS == 1:
            import json
            import math as _math

            # Grid dimensions from piece coords: p.coord = (gn, ge) where gn=north, ge=east.
            # Start piece (SW corner) is at (0, 0); puzzle grows N and E from there.
            coords = [p.coord for p in self.pieces_ if hasattr(p, "coord")]
            grid_H = (max(c[0] for c in coords) + 1) if coords else 1  # north axis
            grid_W = (max(c[1] for c in coords) + 1) if coords else 1  # east axis

            _REQUIRED_OUTWARD = {
                Directions.N: -_math.pi / 2,
                Directions.S:  _math.pi / 2,
                Directions.E:  0.0,
                Directions.W:  _math.pi,
            }

            def _border_R(piece, centroid_arr, edge_src_dict=None, edge_dir_dict=None):
                """Circular-mean outward-normal deviation for all BORDER edges.
                Uses edge_src_dict shapes (source image) when provided, else e.shape (solved layout).
                Uses edge_dir_dict direction labels when provided, else current e.direction.
                Returns angle in radians, or None if no usable border edges."""
                thetas = []
                for e in piece.edges_:
                    if e.type != TypeEdge.BORDER:
                        continue
                    if edge_src_dict is not None:
                        pts = edge_src_dict.get(id(e))
                    else:
                        pts = np.asarray(e.shape, dtype=float) if len(e.shape) >= 2 else None
                    if pts is None or len(pts) < 2:
                        continue
                    mid = np.asarray(pts, dtype=float).mean(axis=0)
                    dx = float(mid[0] - centroid_arr[0])
                    dy = float(mid[1] - centroid_arr[1])
                    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
                        continue
                    direction = edge_dir_dict.get(id(e), e.direction) if edge_dir_dict else e.direction
                    t = _math.atan2(dy, dx) - _REQUIRED_OUTWARD.get(direction, 0.0)
                    thetas.append((t + _math.pi) % (2 * _math.pi) - _math.pi)
                if not thetas:
                    return None
                ms = sum(_math.sin(t) for t in thetas) / len(thetas)
                mc = sum(_math.cos(t) for t in thetas) / len(thetas)
                return _math.atan2(ms, mc)

            def _estimate_rotation_by_shape_match(src_pts, src_centroid, cur_pts, cur_centroid):
                """
                Brute-force rotation search (coarse 2°, fine 0.25°): find the CCW angle
                (standard-math convention, positive=CCW) that best maps the centred source
                edge cloud onto the centred solved edge cloud via a distance-transform score.
                Returns degrees, or None if insufficient data.
                Adapted from PuzzleSimulator/tests/run_robot_solver.py.
                """
                import cv2 as _cv2

                if src_pts is None or len(src_pts) < 10 or cur_pts is None or len(cur_pts) < 10:
                    return None

                src_c = np.asarray(src_centroid, dtype=np.float32)
                cur_c = np.asarray(cur_centroid, dtype=np.float32)
                src_rel = np.asarray(src_pts, dtype=np.float32) - src_c
                cur_rel = np.asarray(cur_pts, dtype=np.float32) - cur_c

                # Downsample to keep compute time reasonable
                max_pts = 2000
                if len(src_rel) > max_pts:
                    src_rel = src_rel[::max(1, len(src_rel)//max_pts)]
                if len(cur_rel) > max_pts:
                    cur_rel = cur_rel[::max(1, len(cur_rel)//max_pts)]

                if len(src_rel) < 10 or len(cur_rel) < 10:
                    return None

                all_pts = np.vstack([src_rel, cur_rel])
                min_x, min_y = float(np.min(all_pts[:, 0])), float(np.min(all_pts[:, 1]))
                max_x, max_y = float(np.max(all_pts[:, 0])), float(np.max(all_pts[:, 1]))
                pad = 25
                w = int(max_x - min_x) + 2 * pad + 1
                h = int(max_y - min_y) + 2 * pad + 1
                if w <= 5 or h <= 5 or w > 3000 or h > 3000:
                    return None

                origin = np.array([pad - min_x, pad - min_y], dtype=np.float32)

                # Build distance-transform image from solved (target) edge points
                tgt_img = np.full((h, w), 255, dtype=np.uint8)
                tx = np.round(cur_rel[:, 0] + origin[0]).astype(np.int32)
                ty = np.round(cur_rel[:, 1] + origin[1]).astype(np.int32)
                vm = (tx >= 0) & (ty >= 0) & (tx < w) & (ty < h)
                tgt_img[ty[vm], tx[vm]] = 0
                # Dilate a little so pixel rounding is not penalised too harshly
                kernel = np.ones((3, 3), np.uint8)
                tgt_img = 255 - _cv2.dilate(255 - tgt_img, kernel, iterations=2)
                dist = _cv2.distanceTransform(tgt_img, _cv2.DIST_L2, 3)

                def _score(angle_deg):
                    a = _math.radians(angle_deg)
                    ca, sa = _math.cos(a), _math.sin(a)
                    rx = src_rel[:, 0] * ca - src_rel[:, 1] * sa + origin[0]
                    ry = src_rel[:, 0] * sa + src_rel[:, 1] * ca + origin[1]
                    ix = np.round(rx).astype(np.int32)
                    iy = np.round(ry).astype(np.int32)
                    valid = (ix >= 0) & (iy >= 0) & (ix < w) & (iy < h)
                    if np.count_nonzero(valid) < max(10, int(0.3 * len(src_rel))):
                        return float('inf')
                    return float(np.mean(dist[iy[valid], ix[valid]]))

                # Coarse search
                best, best_s = 0.0, float('inf')
                for a in np.arange(-180.0, 180.0, 2.0):
                    s = _score(a)
                    if s < best_s:
                        best_s, best = s, float(a)

                # Fine search ±3° around best
                for a in np.arange(best - 3.0, best + 3.01, 0.25):
                    n = ((a + 180) % 360) - 180
                    s = _score(n)
                    if s < best_s:
                        best_s, best = s, n

                return round(((best + 180) % 360) - 180, 2)

            # Compute per-piece rotations NOW, before the global alignment rotation
            # is applied to the solved layout.  This way raw_rot is purely the
            # solver-applied rotation (rotate_edges + Mover) with no source_R0 bias.
            pre_ec_list = [self._piece_centroid(p) for p in self.pieces_]
            rotation_degs = []
            for i, p in enumerate(self.pieces_):
                sc_i = _start_centers[id(p)]
                pre_ec = pre_ec_list[i]
                rdeg = 0.0
                if pre_ec is not None:
                    src_info = _start_ref_pts.get(id(p))
                    src_all = src_info[1] if src_info is not None else None
                    cur_list = [np.asarray(e.shape, dtype=float) for e in p.edges_ if len(e.shape) > 0]
                    cur_all = np.concatenate(cur_list, axis=0) if cur_list else None
                    raw_rot = _estimate_rotation_by_shape_match(
                        src_all, np.array(sc_i, dtype=float),
                        cur_all, np.array(pre_ec, dtype=float),
                    )
                    if raw_rot is not None:
                        # raw_rot: CCW in shape-match formula = CW in image.
                        # Negate to get robot CCW-positive convention.
                        solver_rot = -raw_rot
                        deg = ((solver_rot + 180) % 360) - 180
                        deg += config.PUZZLE_TARGET_ROTATION_DEG
                        if hasattr(p, "coord"):
                            ge_col = p.coord[1]
                            deg += config.COLUMN_ROTATION_CORRECTIONS.get(ge_col, 0.0)
                        deg = ((deg + 180) % 360) - 180
                        rdeg = round(deg, 1)
                rotation_degs.append(rdeg)

            # Centroids from the solved layout (for debug display coords).
            ec_list = [self._piece_centroid(p) for p in self.pieces_]

            records = []
            for i, p in enumerate(self.pieces_):
                sc = _start_centers[id(p)]
                ec = ec_list[i]
                robot_start = config.pixel_to_robot(sc[0], sc[1])

                # Map solved grid coordinate → target field robot mm.
                if hasattr(p, "coord"):
                    gn, ge = p.coord   # (north, east)
                    robot_end = list(config.grid_to_robot(ge, gn, grid_W, grid_H))
                else:
                    robot_end = list(robot_start)

                rotation_deg = rotation_degs[i]

                records.append({
                    "piece_index": i,
                    "grid_coord": list(p.coord) if hasattr(p, "coord") else None,
                    "start_center_px": [int(sc[0]), int(sc[1])],
                    "start_center_robot_mm": list(robot_start),
                    "end_center_px": [int(ec[0]), int(ec[1])] if ec else None,
                    "end_center_robot_mm": robot_end,
                    "rotation_deg": rotation_deg,
                })
            out_path = os.path.join(os.environ.get("ZOLVER_TEMP_DIR", "debug_output"), "piece_centers.json")
            with open(out_path, "w") as f:
                json.dump(records, f, indent=2)
            self.log("Piece centers saved to", out_path)

            self.export_pieces(
                os.path.join(os.environ["ZOLVER_TEMP_DIR"], "stick.png"),
                display=False,
            )

    def get_bbox(self):
        bboxes = [p.get_bbox() for p in self.pieces_]
        if not bboxes:
            return (0, 0, 0, 0)
        return (
            min(bbox[0] for bbox in bboxes),
            min(bbox[1] for bbox in bboxes),
            max(bbox[2] for bbox in bboxes),
            max(bbox[3] for bbox in bboxes),
        )

    def rotate_bbox(self, angle, around):
        minX, minY, maxX, maxY = self.get_bbox()
        rotated = [rotate((x, y), angle, around) for x in [minX, maxX] for y in [minY, maxY]]
        return (
            int(min(p[0] for p in rotated)),
            int(min(p[1] for p in rotated)),
            int(max(p[0] for p in rotated)),
            int(max(p[1] for p in rotated)),
        )

    def solve(self, connected_pieces, left_pieces):
        if len(self.connected_directions) == 0:
            self.connected_directions = [((0, 0), connected_pieces[0])]
            self.diff = self.compute_diffs(left_pieces, self.diff, connected_pieces[0])
        else:
            self.diff = self.add_to_diffs(left_pieces)

        while len(left_pieces) > 0:
            self.log(
                "<--- New match ---> pieces left: ", len(left_pieces),
                "extremum:", self.extremum,
                "puzzle dimension:", display_dim(self.possible_dim),
            )
            block_best_e, best_e = self.best_diff(self.diff, self.connected_directions, left_pieces)

            if block_best_e is None or best_e is None:
                self.log("No match found — solver cannot continue")
                break

            block_best_p = self.edge_to_piece[block_best_e]
            best_p       = self.edge_to_piece[best_e]

            _centroid_bloc = self._piece_centroid(block_best_p) if config.EDGE_OFFSET > 0 else None
            _centroid_cand = self._piece_centroid(best_p)       if config.EDGE_OFFSET > 0 else None
            stick_pieces(block_best_e, best_p, best_e,
                         centroid_bloc=_centroid_bloc, centroid_cand=_centroid_cand)

            self.update_direction(block_best_e, best_p, best_e)
            self.connect_piece(self.connected_directions, block_best_p, block_best_e.direction, best_p)

            connected_pieces.append(best_p)
            del left_pieces[left_pieces.index(best_p)]

            self.diff = self.compute_diffs(left_pieces, self.diff, best_p, edge_connected=block_best_e)

            self.export_pieces(
                os.path.join(os.environ["ZOLVER_TEMP_DIR"], "stick{0:03d}.png".format(len(self.connected_directions))),
            )

        return connected_pieces

    def compute_diffs(self, left_pieces, diff, new_connected, edge_connected=None):
        if edge_connected is not None:
            del diff[edge_connected]

        edges_to_test = [
            (piece, edge)
            for piece in left_pieces
            for edge in piece.edges_
            if not edge.connected
        ]

        centroid_bloc = self._piece_centroid(new_connected) if config.EDGE_OFFSET > 0 else None

        for e in new_connected.edges_:
            for _, v in diff.items():
                if e in v:
                    del v[e]

            if e.connected:
                continue

            diff_e = {}
            for piece, edge in edges_to_test:
                if not e.is_compatible(edge):
                    continue
                for e2 in piece.edges_:
                    e2.backup_shape()

                centroid_cand_pre = self._piece_centroid(piece) if config.EDGE_OFFSET > 0 else None
                stick_pieces(e, piece, edge, centroid_bloc=centroid_bloc, centroid_cand=centroid_cand_pre)
                centroid_cand = self._piece_centroid(piece) if config.EDGE_OFFSET > 0 else None

                if self.green_:
                    diff_e[edge] = real_edge_compute(edge, e, centroid_cand, centroid_bloc)
                else:
                    diff_e[edge] = generated_edge_compute(edge, e, centroid_cand, centroid_bloc)

                for e2 in piece.edges_:
                    e2.restore_backup_shape()

            diff[e] = diff_e
        return diff

    def fallback(self, diff, connected_direction, left_piece, strat=Strategy.NAIVE):
        self.log("Fail to solve the puzzle with", self.strategy, "falling back to", strat)
        old_strat = self.strategy
        self.strategy = strat
        best_bloc_e, best_e = self.best_diff(diff, connected_direction, left_piece)
        self.strategy = old_strat
        return best_bloc_e, best_e

    def best_diff(self, diff, connected_direction, left_piece):
        best_bloc_e, best_e, min_diff = None, None, float("inf")
        minX, minY, maxX, maxY = self.extremum
        occupied = {coord for coord, _ in connected_direction}

        if self.strategy == Strategy.FILL:
            best_coords = []
            for i in range(4, -1, -1):
                best_coord = []
                for x in range(minX, maxX + 1):
                    for y in range(minY, maxY + 1):
                        if (x, y) in occupied:
                            continue
                        neighbor = [e for e in connected_direction if is_neighbor((x, y), e[0], connected_direction)]
                        if len(neighbor) == i:
                            best_coord.append(((x, y), neighbor))
                best_coords.append(best_coord)

            for best_coord in best_coords:
                for c, neighbor in best_coord:
                    for p in left_piece:
                        for _ in range(4):
                            diff_score = 0
                            p.rotate_edges(1)
                            last_test = None, None
                            for block_c, block_p in neighbor:
                                direction_exposed = Directions(sub_tuple(c, block_c))
                                edge_exposed = block_p.edge_in_direction(direction_exposed)
                                edge = p.edge_in_direction(get_opposite_direction(direction_exposed))
                                if edge_exposed.connected or edge.connected or not edge.is_compatible(edge_exposed):
                                    diff_score = float("inf")
                                    break
                                diff_score += diff.get(edge_exposed, {}).get(edge, float("inf"))
                                if diff_score == float("inf"):
                                    break
                                last_test = edge_exposed, edge
                            if diff_score < min_diff:
                                best_bloc_e, best_e, min_diff = last_test[0], last_test[1], diff_score
                if best_e is not None:
                    break
                elif len(best_coord):
                    self.log("Fall back to a worst", self.strategy)
            if best_e is None:
                best_bloc_e, best_e = self.fallback(diff, connected_direction, left_piece)
            return best_bloc_e, best_e

        elif self.strategy == Strategy.BORDER:
            best_coord = []
            for x in range(minX, maxX + 1):
                for y in range(minY, maxY + 1):
                    if (x, y) in occupied:
                        continue
                    neighbor = [e for e in connected_direction if is_neighbor((x, y), e[0], connected_direction)]
                    if len(neighbor) == 1:
                        best_coord.append(((x, y), neighbor[0]))
                    elif len(neighbor) >= 2 and len(left_piece) == 1:
                        best_coord.append(((x, y), neighbor))

            for c, neighbor in best_coord:
                neighbors = neighbor if isinstance(neighbor, list) else [neighbor]
                for p in left_piece:
                    for _ in range(4):
                        p.rotate_edges(1)
                        if p.type == TypePiece.ANGLE and (
                            not corner_puzzle_alignment(c, self.corner_pos)
                            or not self.corner_place_fit_size(c)
                        ):
                            continue
                        if p.type == TypePiece.BORDER and self.is_edge_at_corner_place(c):
                            continue

                        diff_score = 0
                        last_pair = None, None
                        for block_c, block_p in neighbors:
                            direction_exposed = Directions(sub_tuple(c, block_c))
                            edge_exposed = block_p.edge_in_direction(direction_exposed)
                            edge = p.edge_in_direction(get_opposite_direction(direction_exposed))
                            if (edge_exposed.connected or edge.connected
                                    or not edge.is_compatible(edge_exposed)
                                    or not p.is_border_aligned(block_p)):
                                diff_score = float("inf")
                                break
                            diff_score += diff.get(edge_exposed, {}).get(edge, float("inf"))
                            if diff_score == float("inf"):
                                break
                            last_pair = edge_exposed, edge

                        if diff_score < min_diff and last_pair[0] is not None:
                            best_bloc_e, best_e, min_diff = last_pair[0], last_pair[1], diff_score

            if best_e is None:
                best_bloc_e, best_e = self.fallback(diff, connected_direction, left_piece, strat=Strategy.FILL)
            return best_bloc_e, best_e

        elif self.strategy == Strategy.NAIVE:
            for block_e, block_e_diff in diff.items():
                for e, score in block_e_diff.items():
                    if score < min_diff:
                        best_bloc_e, best_e, min_diff = block_e, e, score
            return best_bloc_e, best_e

        return None, None

    def add_to_diffs(self, left_pieces):
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
                if self.green_:
                    diff_e[edge] = real_edge_compute(edge, e)
                else:
                    diff_e[edge] = generated_edge_compute(edge, e)
                for e2 in piece.edges_:
                    e2.restore_backup_shape()
        return self.diff

    def update_direction(self, e, best_p, best_e):
        opp = get_opposite_direction(e.direction)
        step = step_direction(opp, best_e.direction)
        best_p.rotation_steps = (best_p.rotation_steps + step) % 4
        for edge in best_p.edges_:
            edge.direction = rotate_direction(edge.direction, step)

    def connect_piece(self, connected_directions, curr_p, dir, best_p):
        old_coord = next(coord for coord, p in connected_directions if p == curr_p)
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
        coeff = [0 if best_p.edge_in_direction(d).connected else 1 for d in directions]
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

    def translate_puzzle(self):
        all_shapes = [e.shape for p in self.pieces_ for e in p.edges_ if len(e.shape) > 0]
        if not all_shapes:
            return
        all_pts = np.concatenate(all_shapes)
        minX, minY = int(all_pts[:, 0].min()), int(all_pts[:, 1].min())
        for p in self.pieces_:
            for e in p.edges_:
                e.shape -= (minX, minY)

    def export_pieces(
        self,
        path_contour,
        name_contour=None,
        display=True,
        display_border=False,
    ):
        show_in_viewer = bool(self.viewer and display)
        save_to_file   = config.DEBUG_FILE_OUTPUT == 1
        if not show_in_viewer and not save_to_file:
            return

        # e.shape stores (col, row) — col maps to the Y axis, row to the X axis.
        minX, minY, maxX, maxY = self.get_bbox()
        border_img  = np.zeros((maxX - minX + 1, maxY - minY + 1, 3))

        for piece in self.pieces_:
            if config.DEBUG_FILE_OUTPUT == 1:
                for e in piece.edges_:
                    for ey, ex in e.shape:
                        ey, ex = ey - minY, ex - minX
                        if 0 <= ey < border_img.shape[1] and 0 <= ex < border_img.shape[0]:
                            rgb = (0, 0, 0)
                            if e.type == TypeEdge.HOLE:
                                rgb = (102, 178, 255)
                            elif e.type == TypeEdge.HEAD:
                                rgb = (255, 255, 102)
                            elif e.type == TypeEdge.UNDEFINED:
                                rgb = (255, 0, 0)
                            if e.connected:
                                rgb = (0, 255, 0)
                            border_img[ex, ey] = (rgb[2], rgb[1], rgb[0])

                if config.EDGE_OFFSET > 0:
                    valid_shapes = [e.shape for e in piece.edges_ if len(e.shape) > 0]
                    if valid_shapes:
                        centroid = tuple(np.concatenate(
                            [np.asarray(s, dtype=np.float32) for s in valid_shapes]
                        ).mean(axis=0))
                        for e in piece.edges_:
                            if len(e.shape) < 2:
                                continue
                            offset_pts = e.compute_offset_shape(config.EDGE_OFFSET, centroid)
                            pts_img = np.array(
                                [[(int(oy) - minY, int(ox) - minX)] for oy, ox in offset_pts],
                                dtype=np.int32,
                            )
                            cv2.polylines(border_img, [pts_img], isClosed=False, color=(0, 165, 255), thickness=1)

                for e in piece.edges_:
                    if len(e.shape) == 0:
                        continue
                    ey, ex = e.shape[0]
                    ix, iy = int(ex) - minX, int(ey) - minY
                    if 0 <= ix < border_img.shape[0] and 0 <= iy < border_img.shape[1]:
                        cv2.circle(border_img, (iy, ix), 4, (128, 0, 128), -1)

                centroid = self._piece_centroid(piece)
                if centroid is not None:
                    com_x = int(centroid[1]) - minX  # row - minX
                    com_y = int(centroid[0]) - minY  # col - minY
                    if 0 <= com_x < border_img.shape[0] and 0 <= com_y < border_img.shape[1]:
                        cv2.drawMarker(border_img, (com_y, com_x), (0, 255, 255),
                                       cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)

        if config.DEBUG_FILE_OUTPUT == 1:
            legend = [
                ((0, 255, 0),     "BORDER (flat edge)"),
                ((255, 178, 102), "HOLE (indentation)"),
                ((102, 255, 255), "HEAD (protrusion)"),
                ((0, 0, 255),     "UNDEFINED"),
                ((0, 165, 255),   f"OUTER EDGE (+{config.EDGE_OFFSET}px offset)"),
            ]
            box_size, padding = 16, 6
            line_h = box_size + padding
            lx, ly = 10, 10
            cv2.rectangle(border_img, (lx - 2, ly - 2),
                          (lx + 210, ly + len(legend) * line_h + padding), (40, 40, 40), -1)
            for i, (color, label) in enumerate(legend):
                y0 = ly + padding + i * line_h
                cv2.rectangle(border_img, (lx + 4, y0), (lx + 4 + box_size, y0 + box_size), color, -1)
                cv2.putText(border_img, label, (lx + 4 + box_size + 6, y0 + box_size - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
            config.save_debug_img(path_contour, border_img)
            if config.DEBUG_SHOW_DIAGRAMS == 1:
                show_image(border_img, "contour category image")

    def compute_possible_size(self, nb_piece, nb_border) -> list[tuple]:
        nb_edge_border = nb_border - 4
        nb_middle = nb_piece - nb_border
        possibilities = []
        for i in range(nb_edge_border // 2 + 1):
            w, h = i, (nb_edge_border // 2) - i
            if w * h == nb_middle:
                possibilities.append((w + 1, h + 1))
        self.log(
            "Possible sizes: (", nb_piece, "pieces with", nb_border,
            "borders among them):", display_dim(possibilities),
        )
        return possibilities

    def corner_place_fit_size(self, c, update_dim=False):
        def almost_equals(idx, target, val):
            return val[idx] == target or val[idx] == -target

        if len(self.possible_dim) == 1:
            return (
                c[0] == 0 or c[0] == self.possible_dim[0][0] or c[0] == -self.possible_dim[0][0]
            ) and (
                c[1] == 0 or c[1] == self.possible_dim[0][1] or c[0] == -self.possible_dim[0][1]
            )

        if c[0] == 0:
            filtered = [x for x in self.possible_dim if almost_equals(1, c[1], x)]
            if filtered:
                if update_dim and len(filtered) != len(self.possible_dim):
                    self.log("Update possible dimensions with corner place:", display_dim(filtered))
                    self.possible_dim = filtered
                return True
            return False
        elif c[1] == 0:
            filtered = [x for x in self.possible_dim if almost_equals(0, c[0], x)]
            if filtered:
                if update_dim and len(filtered) != len(self.possible_dim):
                    self.log("Update possible dimensions with corner place:", display_dim(filtered))
                    self.possible_dim = filtered
                return True
            return False
        return False

    def is_edge_at_corner_place(self, c):
        if len(self.possible_dim) == 1:
            return (
                c[0] == 0 or c[0] == self.possible_dim[0][0] or c[0] == -self.possible_dim[0][0]
            ) and (
                c[1] == 0 or c[1] == self.possible_dim[0][1] or c[0] == -self.possible_dim[0][1]
            )
        return False

    def update_dimension(self):
        if len(self.possible_dim) == 1:
            return
        _, _, maxX, maxY = self.extremum
        dims = [(x, y) for x, y in self.possible_dim if maxX <= x and maxY <= y]
        if len(dims) != len(self.possible_dim):
            self.log("Update possible dimensions with extremum", self.extremum, ":", display_dim(dims))
            self.possible_dim = dims
