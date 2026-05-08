import numpy as np
from .. import config

from .utils import rotate, angle_between


def stick_pieces(bloc_e, p, e, final_stick=False, centroid_bloc=None, centroid_cand=None):
    """
    Stick an edge of a piece to the bloc of already resolved pieces.

    :param bloc_e: bloc edge already solved
    :param p: piece to add to the bloc
    :param e: edge of p to stick
    :param final_stick: if True, also transform the pixel data of p
    :param centroid_bloc: centroid of the bloc piece (all edge points averaged).
                          When provided and EDGE_OFFSET > 0, alignment uses the
                          outward-offset edge endpoints instead of the raw contour.
    :param centroid_cand: centroid of the candidate piece (same coordinate system
                          as e.shape, i.e. before this stick call transforms it).
    :return: Nothing
    """
    # Determine reference points for alignment.
    # When offset mode is active and centroids are available, use the offset
    # (outward-shifted) edge endpoints so that the snap-to happens on the
    # padded boundary rather than the raw detected contour.
    if config.EDGE_OFFSET > 0 and centroid_bloc is not None and centroid_cand is not None:
        bloc_ref = bloc_e.compute_offset_shape(config.EDGE_OFFSET, centroid_bloc)
        e_ref    = e.compute_offset_shape(config.EDGE_OFFSET, centroid_cand)
    else:
        bloc_ref = np.asarray(bloc_e.shape, dtype=float)
        e_ref    = np.asarray(e.shape,      dtype=float)

    # Translation vector: move e_ref[-1] onto bloc_ref[0]
    translation = np.round(np.subtract(bloc_ref[0], e_ref[-1])).astype(int)

    # Compute angle between the two edges
    vec_bloc  = np.subtract(bloc_ref[0], bloc_ref[-1])
    vec_piece = np.subtract(e_ref[0],    e_ref[-1])
    angle = angle_between(
        (vec_bloc[0], vec_bloc[1], 0), (-vec_piece[0], -vec_piece[1], 0)
    )

    # Rotation centre is bloc_ref[0] (the point e_ref[-1] was just moved to)
    rot_center = bloc_ref[0]

    # First move the first corner of piece to the corner of bloc edge
    for edge in p.edges_:
        edge.shape += translation

    # Then rotate piece of `angle` degrees centered on the corner
    for edge in p.edges_:
        for i, point in enumerate(edge.shape):
            edge.shape[i] = rotate(point, -angle, rot_center)

    # Distribute any residual edge-length mismatch symmetrically at both corners.
    # After the above alignment, e_ref[-1] → bloc_ref[0] (by construction).
    # Apply the same rigid transform to e_ref[0] to find where the far endpoint
    # ended up, then compare with bloc_ref[-1].
    # Round to int: edge shapes must stay integer-typed so that placed edges used
    # as block edges in subsequent compute_diffs calls produce an integer translation
    # (numpy 2.x rejects float+=int / int+=float in-place upcasts).
    e_ref_start_translated = e_ref[0] + translation
    e_ref_start_rotated = np.array(
        rotate((float(e_ref_start_translated[0]), float(e_ref_start_translated[1])),
               -angle, rot_center)
    )
    gap = np.subtract(bloc_ref[-1], e_ref_start_rotated)
    correction_int = np.round(gap / 2.0).astype(int)
    for edge in p.edges_:
        edge.shape = edge.shape + correction_int

    if final_stick:
        # Rotation origin (row, col order as used by rotate())
        b_e0, b_e1 = float(rot_center[0]), float(rot_center[1])

        # Edges are: translate → rotate → apply correction_int.
        # Pixels must follow the same order: translate first (no correction yet),
        # then rotate, then shift target range by correction as a post-rotation
        # translation.  Folding correction into the pre-rotation translate would
        # produce a (I-R)*correction_int positional error for any non-zero angle.
        p.translate(int(translation[1]), int(translation[0]))

        # correction in pixel (col, row) space — edges store (row, col) so swap
        corr_col = int(correction_int[1])
        corr_row = int(correction_int[0])

        # Bounding boxes of origin/target space (target shifted by post-rotation correction)
        minX, minY, maxX, maxY = p.get_bbox()
        minX2, minY2, maxX2, maxY2 = p.rotate_bbox(angle, (b_e1, b_e0))
        minX2 += corr_col;  maxX2 += corr_col
        minY2 += corr_row;  maxY2 += corr_row

        # Recreate image from pixels
        img_p = p.get_image()

        # Retrieve new pixels by rotated target space into origin space.
        # Strip the post-rotation correction before rotating back.
        pixels = {}
        for px in range(minX2, maxX2 + 1):
            for py in range(minY2, maxY2 + 1):
                qx, qy = rotate((px - corr_col, py - corr_row), -angle, (b_e1, b_e0))
                qx, qy = int(qx), int(qy)
                if (
                    minX <= qx <= maxX
                    and minY <= qy <= maxY
                    and img_p[qx - minX, qy - minY][0] != -1
                ):
                    pixels[(px, py)] = img_p[qx - minX, qy - minY]
        p.pixels = pixels
