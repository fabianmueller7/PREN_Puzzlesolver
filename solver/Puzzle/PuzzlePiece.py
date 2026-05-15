import numpy as np

from .Enums import TypeEdge, TypePiece, rotate_direction


class PuzzlePiece:
    """
    Wrapper used to store informations about pieces of the puzzle.
    Contains the position of the piece in the puzzle graph, a list of edges,
    the number of borders and the type of the piece.
    Piece geometry is represented entirely by edge contours (edge.shape arrays).
    """

    def __init__(self, edges):
        self.position = (0, 0)
        self.edges_ = edges
        self.nBorders_ = self.number_of_border()
        self.type = TypePiece(self.nBorders_)
        self.is_border = self.number_of_border() > 0
        self.rotation_steps = 0  # cumulative 90° CW steps applied during solving

    def get_bbox(self):
        # e.shape stores (col, row) pairs; returns (minX=min_row, minY=min_col, maxX=max_row, maxY=max_col)
        shapes = [e.shape for e in self.edges_ if len(e.shape) > 0]
        if not shapes:
            return (0, 0, 0, 0)
        all_pts = np.concatenate(shapes)
        rows = all_pts[:, 1]
        cols = all_pts[:, 0]
        return (int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max()))

    def get_center(self):
        minX, minY, maxX, maxY = self.get_bbox()
        return ((minX + maxX) // 2, (minY + maxY) // 2)

    def number_of_border(self):
        """Fast computations of the number of borders"""

        return len(list(filter(lambda x: x.type == TypeEdge.BORDER, self.edges_)))

    def rotate_edges(self, r):
        """Rotate the edges"""
        self.rotation_steps = (self.rotation_steps + r) % 4
        for e in self.edges_:
            e.direction = rotate_direction(e.direction, r)

    def edge_in_direction(self, dir):
        """Return the edge in the `dir` direction"""

        for e in self.edges_:
            if e.direction == dir:
                return e

    def is_border_aligned(self, p2):
        """Find if a border of the piece is aligned with a border of `p2`"""

        for e in self.edges_:
            if (
                e.type == TypeEdge.BORDER
                and p2.edge_in_direction(e.direction).type == TypeEdge.BORDER
            ):
                return True
        return False
