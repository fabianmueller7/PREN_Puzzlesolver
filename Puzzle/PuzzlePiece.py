import numpy as np

from .Enums import TypeEdge, TypePiece, rotate_direction
from .utils import rotate


class PuzzlePiece:
    """
    Wrapper used to store informations about pieces of the puzzle.
    Contains the position of the piece in the puzzle graph, a list of edges,
    the list of pixels composing the piece, the number of borders and the type
    of the piece.
    """
    initial_pixels = None
    initial_edge_directions = {}

    def __init__(self, edges, pixels):
        self.position = (0, 0)
        self.edges_ = edges
        self.pixels = pixels
        self.nBorders_ = self.number_of_border()
        self.type = TypePiece(self.nBorders_)
        self.is_border = self.number_of_border() > 0

    def get_bbox(self):
        x = list(map(lambda p: p[0], self.pixels))
        y = list(map(lambda p: p[1], self.pixels))
        return int(min(x)), int(min(y)), int(max(x)), int(max(y))

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

    def get_center(self):
        minX, minY, maxX, maxY = self.get_bbox()
        return ((minX + maxX) // 2, (minY + maxY) // 2)

    def translate(self, dx, dy):
        self.pixels = {(x + dx, y + dy): c for (x, y), c in self.pixels.items()}

    def rotate(self, angle, around):
        self.pixels = {
            rotate((x, y), angle, around, to_int=True): c
            for (x, y), c in self.pixels.items()
        }

    def get_image(self):
        minX, minY, maxX, maxY = self.get_bbox()
        img_p = np.full((maxX - minX + 1, maxY - minY + 1, 3), -1)
        for (x, y), c in self.pixels.items():
            img_p[x - minX, y - minY] = c
        return img_p

    def number_of_border(self):
        """Fast computations of the number of borders"""

        return len(list(filter(lambda x: x.type == TypeEdge.BORDER, self.edges_)))

    def rotate_edges(self, r):
        """Rotate the edges"""

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

    def apply_edge_offset(self, offset_px):
        """
        Permanently expand all edges of the piece outward by offset_px.
        This compensates for physical gaps or mechanical inaccuracies.
        """
        if offset_px <= 0:
            return
        center = self.get_center()  # (x, y)
        centroid_yx = (center[1], center[0])
        for edge in self.edges_:
            edge.shape = edge.compute_offset_shape(offset_px, centroid_yx)

    def backup_initial_state(self): # This method is called once per piece after extraction and offset
        """
        Stores the current state of the piece as its 'initial' state.
        This is used to reset the piece for new solve attempts.
        """
        self._initial_state = self.get_current_state()
        # Also backup edge shapes for temporary use in diff computation
        for e in self.edges_:
            e.backup_shape()

    def restore_initial_state(self):
        """Resets the piece to its stored 'initial' state."""
        if self._initial_state is not None:
            self.set_state(self._initial_state)

    def _backup_edge_shapes(self):
        """Backs up only the current edge shapes for temporary modifications."""
        for e in self.edges_:
            e.backup_shape() # This sets e.shape_backup

    def _restore_edge_shapes(self):
        """Restores only the edge shapes from their backup."""
        for e in self.edges_:
            e.restore_backup_shape() # This restores e.shape from e.shape_backup

    def get_current_state(self):
        """Returns a serializable representation of the piece's current state."""
        edge_states = []
        for e in self.edges_:
            edge_states.append({
                'shape': e.shape.copy() if isinstance(e.shape, np.ndarray) else e.shape,
                'color': e.color.copy() if isinstance(e.color, np.ndarray) else e.color,
                'type': e.type,
                'connected': e.connected,
                'direction': e.direction,
                'shape_backup': e.shape_backup.copy() if isinstance(e.shape_backup, np.ndarray) else e.shape_backup,
            })
        return {
            'position': self.position,
            'pixels': self.pixels.copy(),
            'nBorders_': self.nBorders_,
            'type': self.type,
            'is_border': self.is_border,
            'edges_': edge_states,
            'coord': self.coord if hasattr(self, 'coord') else None,
        }

    def set_state(self, state):
        """Restores the piece's state from a given state dictionary."""
        # Restore simple attributes
        self.position = state['position']
        self.pixels = state['pixels']
        self.nBorders_ = state['nBorders_']
        self.type = state['type']
        self.is_border = state['is_border']
        self.coord = state['coord']
        # Restore edge objects
        for i, e_state in enumerate(state['edges_']):
            self.edges_[i].shape = e_state['shape']
            self.edges_[i].color = e_state['color']
            self.edges_[i].type = e_state['type']
            self.edges_[i].connected = e_state['connected']
            self.edges_[i].direction = e_state['direction']
            self.edges_[i].shape_backup = e_state['shape_backup']
