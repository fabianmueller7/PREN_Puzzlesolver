"""Dedicated solver for small puzzles (<=6 pieces, always a 3x2 or 2x3 grid).

Such a puzzle is exactly **4 corner pieces + 2 edge pieces**. Instead of forcing pieces
into a fixed slot template (which can commit a bad edge), this assembles by **confidence**:
score every candidate edge pair once, then merge pieces **best-first** (agglomerative).

The search is purely combinatorial on a **grid embedding** — each piece is tracked as a grid
cell + orientation; a seam fixes the relative cell/orientation of the two clusters. A merge is
accepted only if it stays grid-legal (no cell collision; 3x2/2x3 footprint; no BORDER edge
facing another piece). Best-first DFS with rollback recovers from dead-ends, so poor seams are
used last or never. Geometry is materialised once at the end (stick along grid adjacency) for
rendering / robot coords. The 4-corner+2-edge structure is the validation guard, not a fixed
placement order.
"""
import numpy as np

from .. import config
from .Distance import _geom_match_score, _offset_shape_for
from .Mover import stick_pieces
from .Enums import TypeEdge


# --------------------------------------------------------------------------- #
# Geometry helpers (shape coords are (col=x, row=y); row grows downward)
# --------------------------------------------------------------------------- #
def _centroid(piece):
    pts = [np.asarray(e.shape, dtype=float) for e in piece.edges_ if len(e.shape) > 0]
    return np.concatenate(pts, 0).mean(0)


def _snapshot(pieces):
    return {id(e): np.copy(e.shape) for p in pieces for e in p.edges_}


def _restore(pieces, snap):
    for p in pieces:
        for e in p.edges_:
            e.shape = np.copy(snap[id(e)])


def _seam_cost(cand_edge, bloc_edge, cand, bloc, green):
    """Shape-match cost for a seam. Uses _geom_match_score directly (overlap residual +
    curvature + type priority), deliberately SKIPPING the arc/chord-length gates in
    generated/real_edge_compute: pieces are variable-sized, so a true seam can mate a long
    edge against a shorter one. The overlap residual self-regularises gross mismatches."""
    s1 = _offset_shape_for(cand_edge, _centroid(cand))
    s2 = _offset_shape_for(bloc_edge, _centroid(bloc))
    return _geom_match_score(s1, s2, cand_edge, bloc_edge)


# --------------------------------------------------------------------------- #
# Grid directions. edges_ run in the fixed cyclic order below (verified: opposite =
# (k+2)%4); a piece's "offset" o maps edge index k -> grid direction _CYCLE[(k+o)%4].
# --------------------------------------------------------------------------- #
_CYCLE = ['S', 'E', 'N', 'W']
_CYC_IX = {d: i for i, d in enumerate(_CYCLE)}
_OPP = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}
_STEP = {'N': (-1, 0), 'S': (1, 0), 'E': (0, 1), 'W': (0, -1)}   # (drow, dcol)


def _dir_of(edge_idx, off):
    return _CYCLE[(edge_idx + off) % 4]


def _rot_cell(vec, k):
    """Rotate a (drow, dcol) grid vector by k +1-steps; one step: (dr,dc)->(-dc,dr)."""
    dr, dc = vec
    for _ in range(k % 4):
        dr, dc = -dc, dr
    return (dr, dc)


# --------------------------------------------------------------------------- #
# Seam scoring
# --------------------------------------------------------------------------- #
def _precompute_seams(pieces, green):
    """Score every compatible edge pair once (pose-independent: stick into a scratch frame,
    score, restore). Returns records {cost, pi, ki, pj, kj} (k* = edge index) best-first."""
    seams = []
    n = len(pieces)
    for i in range(n):
        for j in range(i + 1, n):
            pair = [pieces[i], pieces[j]]
            for ki, ei in enumerate(pieces[i].edges_):
                for kj, ej in enumerate(pieces[j].edges_):
                    if not ei.is_compatible(ej):
                        continue
                    snap = _snapshot(pair)
                    stick_pieces(ei, pieces[j], ej,
                                 centroid_bloc=_centroid(pieces[i]),
                                 centroid_cand=_centroid(pieces[j]))
                    cost = _seam_cost(ej, ei, pieces[j], pieces[i], green)
                    _restore(pair, snap)
                    if np.isfinite(cost):
                        seams.append({'cost': cost, 'pi': i, 'ki': ki, 'pj': j, 'kj': kj})
    seams.sort(key=lambda s: s['cost'])
    return seams


# --------------------------------------------------------------------------- #
# Combinatorial grid embedding (union-find clusters of grid cells + orientations)
# --------------------------------------------------------------------------- #
def _uf_find(parent, i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def _grid_merge(seam, pieces, parent, members, cell, off):
    """Embed the smaller cluster into the larger's grid frame via this seam. Accept only if
    grid-legal: no cell collision, 3x2/2x3 footprint, and no BORDER edge facing an occupied
    cell. Mutates (cell/off/parent/members) and returns True on accept; no-op + False else."""
    ri, rj = _uf_find(parent, seam['pi']), _uf_find(parent, seam['pj'])
    if ri == rj:
        return False
    if len(members[ri]) >= len(members[rj]):
        a_i, a_k, b_i, b_k, fixed_root, move_root = \
            seam['pi'], seam['ki'], seam['pj'], seam['kj'], ri, rj
    else:
        a_i, a_k, b_i, b_k, fixed_root, move_root = \
            seam['pj'], seam['kj'], seam['pi'], seam['ki'], rj, ri

    d_a = _dir_of(a_k, off[a_i])                       # direction from A toward B
    rB = (_CYC_IX[_OPP[d_a]] - b_k) % 4                # B's edge b_k must face back
    cellB = (cell[a_i][0] + _STEP[d_a][0], cell[a_i][1] + _STEP[d_a][1])
    delta = (rB - off[b_i]) % 4

    occupied = {cell[k]: k for k in members[fixed_root]}
    new_cells, new_offs = {}, {}
    for q in members[move_root]:
        vec = _rot_cell((cell[q][0] - cell[b_i][0], cell[q][1] - cell[b_i][1]), delta)
        cq = (cellB[0] + vec[0], cellB[1] + vec[1])
        if cq in occupied:
            return False
        occupied[cq] = q
        new_cells[q], new_offs[q] = cq, (off[q] + delta) % 4

    rows = [c[0] for c in occupied]
    cols = [c[1] for c in occupied]
    er, ec = max(rows) - min(rows) + 1, max(cols) - min(cols) + 1
    if er > 3 or ec > 3 or (er > 2 and ec > 2):
        return False

    comb_off = {k: off[k] for k in members[fixed_root]}
    comb_off.update(new_offs)
    for cc, pidx in occupied.items():
        for k, e in enumerate(pieces[pidx].edges_):
            if e.type == TypeEdge.BORDER:
                d = _dir_of(k, comb_off[pidx])
                if (cc[0] + _STEP[d][0], cc[1] + _STEP[d][1]) in occupied:
                    return False

    for q in members[move_root]:
        cell[q], off[q] = new_cells[q], new_offs[q]
    parent[move_root] = fixed_root
    members[fixed_root].extend(members[move_root])
    del members[move_root]
    return True


def _snap_grid(parent, members, cell, off):
    return (list(parent), {k: list(v) for k, v in members.items()}, dict(cell), dict(off))


def _restore_grid(snap, parent, members, cell, off):
    parent[:] = snap[0]
    members.clear(); members.update({k: list(v) for k, v in snap[1].items()})
    cell.clear(); cell.update(snap[2])
    off.clear(); off.update(snap[3])


def _structure_ok(pieces, cell, off):
    """3x2/2x3 footprint where every CONNECTOR edge faces an occupied neighbour and every
    BORDER edge faces empty space (corners at grid corners, edges on the hull)."""
    n = len(pieces)
    rows = [cell[i][0] for i in range(n)]
    cols = [cell[i][1] for i in range(n)]
    if {max(rows) - min(rows) + 1, max(cols) - min(cols) + 1} != {2, 3}:
        return False
    occ = {cell[i] for i in range(n)}
    for i, p in enumerate(pieces):
        for k, e in enumerate(p.edges_):
            d = _dir_of(k, off[i])
            has = (cell[i][0] + _STEP[d][0], cell[i][1] + _STEP[d][1]) in occ
            if e.type == TypeEdge.BORDER and has:
                return False
            if e.type in (TypeEdge.HEAD, TypeEdge.HOLE) and not has:
                return False
    return True


def _dfs(seams, si, pieces, parent, members, cell, off, acc_cost, budget, best):
    """Search structurally-valid complete assemblies, keeping the MIN-TOTAL-COST one in
    `best`. Per-seam shape scores can't separate true/false matches for these pieces, so
    we let the scores decide globally: a locally-cheap false pairing can't win if it forces
    a higher-total or structurally-invalid whole. Branch-and-bound on accumulated cost."""
    if len(members) == 1:
        if acc_cost < best['cost'] and _structure_ok(pieces, cell, off):
            best['cost'] = acc_cost
            best['cell'] = dict(cell)
            best['off'] = dict(off)
        return
    if acc_cost >= best['cost']:        # bound: can't beat the incumbent
        return
    budget[0] -= 1
    if budget[0] < 0:
        return
    for idx in range(si, len(seams)):
        s = seams[idx]
        if _uf_find(parent, s['pi']) == _uf_find(parent, s['pj']):
            continue
        snap = _snap_grid(parent, members, cell, off)
        if _grid_merge(s, pieces, parent, members, cell, off):
            _dfs(seams, idx + 1, pieces, parent, members, cell, off,
                 acc_cost + s['cost'], budget, best)
            _restore_grid(snap, parent, members, cell, off)
    return


# --------------------------------------------------------------------------- #
# Materialise geometry from the grid embedding (stick along grid adjacency)
# --------------------------------------------------------------------------- #
def _geometric_layout(pieces, cell, off):
    cellmap = {cell[i]: i for i in range(len(pieces))}
    anchor = min(range(len(pieces)), key=lambda i: (cell[i][0], cell[i][1]))
    placed, queue = {anchor}, [anchor]
    while queue:
        pi = queue.pop(0)
        for k in range(4):
            d = _dir_of(k, off[pi])
            nb = (cell[pi][0] + _STEP[d][0], cell[pi][1] + _STEP[d][1])
            nj = cellmap.get(nb)
            if nj is None or nj in placed:
                continue
            k2 = next(kk for kk in range(4) if _dir_of(kk, off[nj]) == _OPP[d])
            stick_pieces(pieces[pi].edges_[k], pieces[nj], pieces[nj].edges_[k2],
                         centroid_bloc=_centroid(pieces[pi]), centroid_cand=_centroid(pieces[nj]))
            placed.add(nj)
            queue.append(nj)


def solve_small(pieces, green=False, log=print):
    """Assemble a <=6-piece (4 corner + 2 edge) puzzle by best-first agglomerative merging
    with a grid-embedding validation guard + DFS backtracking. Mutates piece shapes into the
    solved layout and sets p.coord=(row,col). Returns True on success; else restores shapes."""
    corners = [p for p in pieces if p.nBorders_ == 2]
    edges = [p for p in pieces if p.nBorders_ == 1]
    centers = [p for p in pieces if p.nBorders_ == 0]
    if len(pieces) > 6 or len(corners) != 4 or len(edges) != 2 or centers:
        log(f"[small] not applicable: {len(corners)} corners / {len(edges)} edges / "
            f"{len(centers)} centers (need 4/2/0)")
        return False

    orig = _snapshot(pieces)
    n = len(pieces)
    parent = list(range(n))
    members = {i: [i] for i in range(n)}
    cell = {i: (0, 0) for i in range(n)}     # grid cell per piece (cluster-local frame)
    off = {i: 0 for i in range(n)}           # edge->direction rotation offset per piece

    seams = _precompute_seams(pieces, green)
    log("[small] {} candidate seams; best costs: {}".format(
        len(seams), ", ".join(f"{s['cost']:.0f}" for s in seams[:8])))

    # Find the minimum-total-cost structurally-valid 2x3/3x2 assembly (global use of the
    # match scores; per-seam discrimination is too weak for these pieces).
    best = {'cost': float('inf'), 'cell': None, 'off': None}
    _dfs(seams, 0, pieces, parent, members, cell, off, 0.0, [200000], best)

    if best['cell'] is None:
        _restore(pieces, orig)
        log("[small] no valid grid assembly found")
        return False
    cell.clear(); cell.update(best['cell'])
    off.clear(); off.update(best['off'])
    log(f"[small] best valid assembly total cost = {best['cost']:.0f}")

    rows = [cell[i][0] for i in range(n)]
    cols = [cell[i][1] for i in range(n)]
    r0, c0 = min(rows), min(cols)
    for i, p in enumerate(pieces):
        p.coord = (cell[i][0] - r0, cell[i][1] - c0)

    _geometric_layout(pieces, cell, off)
    log("[small] solved (agglomerative + grid embedding)")
    return True
