import numpy as np
import cv2

from .Enums import TypeEdge, Directions


class Edge:
    """
    Wrapper for edges.
    Contains shape, colors, type and positions informations in the puzzle of an edge.
    """

    def __init__(
        self,
        shape,
        color=None,
        edge_type=TypeEdge.HOLE,
        connected=False,
        direction=Directions.N,
    ):
        self.shape = shape
        self.shape_backup = shape
        self.color = color
        self.type = edge_type
        self.connected = connected
        self.direction = direction

    def backup_shape(self):
        """Copy the shape for backup"""
        self.shape_backup = np.copy(self.shape)

    def restore_backup_shape(self):
        """Restore the shape previously backedup"""
        self.shape = self.shape_backup

    def compute_offset_shape(self, offset_px, piece_centroid):
        """
        Return a copy of self.shape shifted outward by offset_px pixels.

        :param offset_px: Number of pixels to shift each point outward along its local normal.
        :param piece_centroid: (c0, c1) tuple — centroid of the piece in the same coordinate
                               system as edge.shape, used to determine which side is outward.
        :return: numpy array of offset (y, x) points
        """
        pts = np.asarray(self.shape, dtype=np.float32)
        if len(pts) < 2 or offset_px == 0:
            return pts.copy()

        n = len(pts)
        normals = np.zeros((n, 2), dtype=np.float32)

        for i in range(n):
            prev = pts[i - 1] if i > 0 else pts[0]
            nxt  = pts[i + 1] if i < n - 1 else pts[-1]
            tangent = nxt - prev
            tang_len = np.linalg.norm(tangent)
            if tang_len < 1e-6:
                continue
            tangent /= tang_len
            # All normals computed in the same rotational direction (left of tangent).
            normals[i] = np.array([-tangent[1], tangent[0]], dtype=np.float32)

        # Decide sign once for the entire edge using the sum of dot products.
        # This avoids per-point sign flips that cause sudden jumps on near-vertical segments.
        centroid_arr = np.array(piece_centroid, dtype=np.float32)
        to_centroid = centroid_arr - pts          # (n, 2) vectors from each point to centroid
        total_dot = float(np.sum(normals * to_centroid))
        if total_dot > 0:
            normals = -normals                    # flip all normals to point away from centroid

        return pts + normals * offset_px

    def is_compatible(self, e2, relaxed=False):
        """BORDER edges cannot match anything. All other pairs are allowed;
        type priority is encoded in the distance score.
        Updated: BORDER edges are compatible only with other BORDER edges.
        Non-BORDER edges are compatible only with other Non-BORDER edges.
        """
        if self.type == TypeEdge.BORDER:
            return e2.type == TypeEdge.BORDER
        else: # self.type is HEAD, HOLE, or UNDEFINED
            return e2.type != TypeEdge.BORDER
