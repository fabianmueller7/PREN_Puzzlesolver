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


def _candidate_dims(n, nC, nE, nX):
    """Rectangle dimensions (w<=h, both >=2) of n pieces whose expected corner/edge/center
    counts match the actual classification. A w x h grid has 4 corners, 2(w-2)+2(h-2) edges,
    (w-2)(h-2) centers. Returns sorted (w,h) tuples (e.g. 4 -> [(2,2)], 6 -> [(2,3)])."""
    out = []
    for w in range(2, int(n ** 0.5) + 1):
        if n % w:
            continue
        h = n // w
        if (4, 2 * (w - 2) + 2 * (h - 2), (w - 2) * (h - 2)) == (nC, nE, nX):
            out.append((w, h))
    return out


def _fits_any(er, ec, cands):
    """Does a partial bbox of extents (er, ec) still fit inside some candidate rectangle?"""
    return any((er <= w and ec <= h) or (er <= h and ec <= w) for (w, h) in cands)


def _grid_merge(seam, pieces, parent, members, cell, off, cands):
    """Embed the smaller cluster into the larger's grid frame via this seam. Accept only if
    grid-legal: no cell collision, bbox still fits a candidate rectangle, and no BORDER edge
    facing an occupied cell. Mutates (cell/off/parent/members), returns True on accept."""
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
    if not _fits_any(er, ec, cands):
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


def _structure_ok(pieces, cell, off, cands):
    """A candidate-rectangle footprint where every CONNECTOR edge faces an occupied
    neighbour and every BORDER edge faces empty space (corners at grid corners, edges on
    the hull, centers interior)."""
    n = len(pieces)
    rows = [cell[i][0] for i in range(n)]
    cols = [cell[i][1] for i in range(n)]
    er, ec = max(rows) - min(rows) + 1, max(cols) - min(cols) + 1
    if (min(er, ec), max(er, ec)) not in cands:
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


def _dfs(seams, si, pieces, parent, members, cell, off, acc_cost, budget, best, cands):
    """Search structurally-valid complete assemblies, keeping the MIN-TOTAL-COST one in
    `best`. Per-seam shape scores can't separate true/false matches for these pieces, so
    we let the scores decide globally: a locally-cheap false pairing can't win if it forces
    a higher-total or structurally-invalid whole. Branch-and-bound on accumulated cost."""
    if len(members) == 1:
        if acc_cost < best['cost'] and _structure_ok(pieces, cell, off, cands):
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
        if _grid_merge(s, pieces, parent, members, cell, off, cands):
            _dfs(seams, idx + 1, pieces, parent, members, cell, off,
                 acc_cost + s['cost'], budget, best, cands)
            _restore_grid(snap, parent, members, cell, off)
    return


# --------------------------------------------------------------------------- #
# Materialise geometry from the grid embedding (axis-lock + least-squares place)
# --------------------------------------------------------------------------- #
_GLOBAL = {'N': np.array([0.0, -1.0]), 'S': np.array([0.0, 1.0]),
           'E': np.array([1.0, 0.0]), 'W': np.array([-1.0, 0.0])}   # outward (x, y), y down


def _apex_point(pts):
    """The connector apex (tab/hole centre) — point of max perpendicular deviation from
    the chord; the natural mating point of a seam, valid for any edge length (T-junctions).
    Accepts a points array (raw edge.shape, or an EDGE_OFFSET-shifted shape)."""
    pts = np.asarray(pts, dtype=float)
    a, b = pts[0], pts[-1]
    ch = b - a
    L = np.linalg.norm(ch)
    if L < 1e-6 or len(pts) < 3:
        return (a + b) / 2.0
    u = ch / L
    nrm = np.array([-u[1], u[0]])
    dev = (pts - a) @ nrm
    return pts[int(np.argmax(np.abs(dev)))]


def _axis_lock(piece, off_i):
    """Rotate a piece (about its centroid) so its edges align to the global axes, using the
    grid orientation `off_i` (edge k should face _GLOBAL[_dir_of(k, off_i)]). Removes the
    per-piece tilt that made single-seam sticking drift."""
    c = _centroid(piece)
    sin_sum = cos_sum = 0.0
    for k, e in enumerate(piece.edges_):
        cur = np.asarray(e.shape, dtype=float).mean(0) - c
        n = np.linalg.norm(cur)
        if n < 1e-6:
            continue
        cur /= n
        tgt = _GLOBAL[_dir_of(k, off_i)]
        ang = np.arctan2(tgt[1], tgt[0]) - np.arctan2(cur[1], cur[0])
        sin_sum += np.sin(ang)
        cos_sum += np.cos(ang)
    a = np.arctan2(sin_sum, cos_sum)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    for e in piece.edges_:
        e.shape = np.round((np.asarray(e.shape, dtype=float) - c) @ R.T + c).astype(int)


def _geometric_layout(pieces, cell, off):
    """Clean, drift-free placement from the exact grid embedding: axis-lock every piece,
    then least-squares translate (orientation fixed) so each seam's connector apexes
    coincide. Anchor piece 0. x and y decouple."""
    n = len(pieces)
    for i in range(n):
        _axis_lock(pieces[i], off[i])

    # Seam constraints: for grid-adjacent (i,j), apex_i + t_i == apex_j + t_j.
    cellmap = {cell[i]: i for i in range(n)}
    cons = []
    for i in range(n):
        for k in range(4):
            d = _dir_of(k, off[i])
            j = cellmap.get((cell[i][0] + _STEP[d][0], cell[i][1] + _STEP[d][1]))
            if j is None or j <= i:
                continue
            k2 = next(kk for kk in range(4) if _dir_of(kk, off[j]) == _OPP[d])
            # Mate on the EDGE_OFFSET-shifted (outward) apexes, not the raw ones, so the
            # layout abuses the manufacturing tolerance: aligning the offset edges leaves a
            # ~2*EDGE_OFFSET gap between the real edges instead of shoving them together.
            # With EDGE_OFFSET = 0 this is exactly the old raw-apex behaviour.
            cons.append((i, _apex_point(_offset_shape_for(pieces[i].edges_[k], _centroid(pieces[i]))),
                         j, _apex_point(_offset_shape_for(pieces[j].edges_[k2], _centroid(pieces[j])))))

    sol = {}
    for axis in (0, 1):
        A = np.zeros((len(cons) + 1, n))
        rhs = np.zeros(len(cons) + 1)
        for r, (i, ai, j, aj) in enumerate(cons):
            A[r, i] += 1.0
            A[r, j] -= 1.0
            rhs[r] = aj[axis] - ai[axis]
        A[-1, 0] = 1.0                                   # anchor piece 0
        sol[axis] = np.linalg.lstsq(A, rhs, rcond=None)[0]
    for i in range(n):
        t = np.array([sol[0][i], sol[1][i]])
        for e in pieces[i].edges_:
            e.shape = np.round(np.asarray(e.shape, dtype=float) + t).astype(int)


def solve_small(pieces, green=False, log=print):
    """Assemble a small (4-9 piece, w x h >=2 grid) puzzle by best-first agglomerative
    merging with a grid-embedding validation guard, choosing the min-total-cost valid whole.
    Mutates piece shapes into the solved layout and sets p.coord=(row,col); else restores."""
    n = len(pieces)
    nC = sum(1 for p in pieces if p.nBorders_ == 2)
    nE = sum(1 for p in pieces if p.nBorders_ == 1)
    nX = sum(1 for p in pieces if p.nBorders_ == 0)
    cands = _candidate_dims(n, nC, nE, nX) if 4 <= n <= 9 else []
    if not cands:
        nOver = sum(1 for p in pieces if p.nBorders_ > 2)
        log(f"[small] not applicable: {n} pieces, {nC} corners / {nE} edges / {nX} centers "
            f"— no rectangle matches this classification")
        log(f"[small] per-piece border counts: {[p.nBorders_ for p in pieces]}")
        if nOver:
            log(f"[small] {nOver} piece(s) have >2 flat edges — upstream edge classification "
                f"over-flagged connectors as BORDER (a real piece has at most 2 flat sides).")
        return False
    log(f"[small] candidate grid dim(s): {cands}")

    orig = _snapshot(pieces)
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
    _dfs(seams, 0, pieces, parent, members, cell, off, 0.0, [200000], best, cands)

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
