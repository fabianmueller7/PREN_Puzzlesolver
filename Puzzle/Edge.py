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

    def is_compatible(self, e2, relaxed=False):
        """BORDER edges cannot match anything. All other pairs are allowed;
        type priority is encoded in the distance score."""
        return (
            self.type  != TypeEdge.BORDER and
            e2.type    != TypeEdge.BORDER
        )
