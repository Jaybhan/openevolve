"""ONE unified construction for z(m, n; s, t):

    construct_unified(m, n, s, t) = realize(profile(m,n,s,t), tower(m,s,t))

The unifying thesis (owner mandate): every winning structure in this project
is a maximal (t-1)-fold s-packing with algebraic symmetry — an ORBIT CLOSURE
of a canonical seed under a canonical group, dressed by a fixed operator
alphabet.  This module implements that as a single uniform rule; there is no
per-cell branching, only arithmetic applicability predicates (the same way
"q a prime power" gates PG(2,q) in the literature).

tower(m, s, t) — the master ordered block sequence, built once per (m,s,t):

  label ladder   for m' in {m, ..., m+3}: generate on m' labels and RESTRICT
                 blocks to [m] (point-deleted designs come out of richer
                 supersets: e.g. m = 9..11 from the 12-point Hadamard object).
  group ladder   on m' labels, every group whose arithmetic predicate holds:
                   EA   elementary abelian F_p^k  (m' = p^k):  flats +
                        affine hyperplane side-sets under an incremental
                        cap/arc predicate on the used normals;
                   PTD  pointed Frobenius {oo} u Z_{m'-1}  (m'-1 an odd
                        prime): the quadratic-residue block QR u {oo};
                   CYC  Z_{m'}  (always): the orbit-capacity-greedy base
                        set (the general fallback).
  operators      applied uniformly to every seed, in fixed order:
                   O1 orbit        all group translates;
                   O2 complement   the [m']-complement of each block;
                   O3 extension    B u {b} for the canonical base point b
                        not moved into B;
                   O4 fusion       unions of two orbit blocks through the
                        base point, minus the point;
                   O5 doubling     each block up to t-1 times (realize-time).
  priority       emitted blocks ordered by (size desc, ladder rank, lex);
                 illegal blocks are skipped at realize time, never trimmed.

profile(m, n, s, t) — the supply dial: the integer waterfill of the
counting bound sum_j C(c_j, s) <= (t-1) C(m, s) (two-sided), giving quota
targets per size class; realize sweeps the heavy-block quota over a small
closed window (the Lemma-A trade-off) and keeps the best verified result.

realize — walk the tower under an exact capacity frame taking blocks whose
every s-subset still has multiplicity headroom (each block up to t-1
times), respecting the quota; finish with the canonical completion
((s+1)-Steiner-layer fill, repeated s-blocks, weight-(s-1) pads) shared
with the reference router.  Verify; return the best.

Everything is deterministic.  The reference multi-family router
(zarankiewicz.construct) stays the benchmark; run price_of_unification()
for the honest per-cell comparison.
"""

import os
import sys
from itertools import combinations
from math import comb

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import zarankiewicz as Z  # capacity frame, verifier, completion, waterfill


# ---------------------------------------------------------------------------
# group/seed generators (uniform, arithmetic predicates only)
# ---------------------------------------------------------------------------


def _prime_power(v):
    for p in range(2, v + 1):
        if Z._is_prime(p):
            k, x = 0, 1
            while x < v:
                x *= p
                k += 1
            if x == v:
                return p, k
    return None


def _ea_blocks(v, s, t):
    """Elementary abelian F_p^k on labels 0..v-1 (digit encoding).  Emits,
    in canonical order:
      - hyperplane side-sets {x: <a,x> = c}, normals ordered CAP-FIRST (the
        canonical maximal cap of PG(k-1,2) is the affine complement
        {a: top digit set}); every (a,c) is emitted and the capacity frame
        at realize time keeps the compatible ones — this single rule
        instantiates BOTH the m<=15 hyperplane family and the m=16 cap
        family;
      - all affine lines (p >= 3): the AG(2,q) line systems;
      - all affine 2-flats (p = 2): the xor-zero quads (Steiner layer /
        SQS planes).
    One rule; instances differ only through (v, p, k)."""
    pp = _prime_power(v)
    if pp is None:
        return []
    p, k = pp
    if p ** k != v or k < 2:
        return []

    def digits(x):
        out = []
        for _ in range(k):
            out.append(x % p)
            x //= p
        return tuple(out)

    def dot(x, a):
        dx = digits(x)
        return sum(xx * aa for xx, aa in zip(dx, a)) % p

    blocks = []
    normals = sorted(range(1, v), key=lambda a: (digits(a)[-1] == 0, a))
    for a in normals:
        da = digits(a)
        # affine sides (c != 0, avoiding the group's fixed point 0) first
        for c in list(range(1, p)) + [0]:
            side = tuple(x for x in range(v) if dot(x, da) == c)
            if len(side) >= s:
                blocks.append(side)
    if p >= 3:
        seen = set()
        for dirv in range(1, v):
            dv = digits(dirv)
            for tr in range(v):
                tv = digits(tr)
                line = tuple(sorted(
                    sum(((a + i * b) % p) * (p ** j)
                        for j, (a, b) in enumerate(zip(tv, dv)))
                    for i in range(p)))
                seen.add(line)
        blocks.extend(sorted(seen))
    if p == 2 and k >= 2:
        seen = set()
        for x in range(v):
            for y in range(x + 1, v):
                for z in range(y + 1, v):
                    w = x ^ y ^ z
                    if w > z:
                        seen.add((x, y, z, w))
        blocks.extend(sorted(seen))
    return blocks


def _ptd_blocks(v, s, t):
    """Pointed Frobenius {oo} u Z_{v-1} (v-1 an odd prime): the orbit of
    QR u {oo} under translation — whose complement closure is the Hadamard
    3-design when v = 0 mod 4."""
    w = v - 1
    if w < 3 or not Z._is_prime(w) or w % 2 == 0:
        return []
    qr = set(Z._qr_set(w))
    blocks = []
    for i in range(w):
        blocks.append(tuple(sorted({(x + i) % w for x in qr} | {w})))
    return blocks


def _cyc_blocks(v, s, t):
    """Cyclic fallback over Z_v: quadratic-residue orbits when v is prime
    (the same QR object as the pointed generator, unpointed), then the
    orbit-capacity-greedy base sets."""
    blocks = []
    if Z._is_prime(v) and v >= 7:
        qr = Z._qr_set(v)
        for base in (qr, sorted(set(qr) | {0})):
            for i in range(v):
                blocks.append(tuple(sorted((x + i) % v for x in base)))
    bases = Z._orbit_capacity_greedy(v, s, t) if comb(v, s) <= 3000 else []
    for B in bases:
        for i in range(v):
            blocks.append(tuple(sorted((x + i) % v for x in B)))
    return blocks


def _prg_blocks(v, s, t):
    """Pair group Z_2^g (sign flips on g row-pairs, + singleton if v odd):
    'blown' quotient triples (unions of three pairs, quotient triples from
    the exact (t-1)-fold triangle packing of K_g) and the transversal orbit
    ordered by cosets of the derived 2-dim code (unused-pair rule) — the
    group-divisible layer, uniform in v."""
    if s != 3 or v < 6 or v > 16:
        return []
    g = v // 2
    single = (v - 1,) if v % 2 else ()
    pairs = [(2 * i, 2 * i + 1) for i in range(g)]
    qcap = {c: t - 1 for c in combinations(range(g), 2)}
    qtr = Z._exact_pack(qcap, list(combinations(range(g), 3)),
                        comb(g, 3), 2, node_budget=20000)
    blocks = [tuple(sorted(pairs[i] + pairs[j] + pairs[k]))
              for (i, j, k) in qtr]
    # derived coset order for the transversals
    use = {c: 0 for c in combinations(range(g), 2)}
    for T in qtr:
        for c in combinations(T, 2):
            use[c] += 1
    a, b = min(use, key=lambda c: (use[c], c))
    rest = sorted(set(range(g)) - {a, b})
    if len(rest) >= 2:
        d1 = (1 << a) | (1 << b)
        d2 = (1 << rest[-2]) | (1 << rest[-1])
        D = [0, d1, d2, d1 ^ d2]
    else:
        D = [0]

    def tv(x):
        return tuple(sorted([pairs[i][(x >> i) & 1] for i in range(g)]
                            + list(single)))
    vs = sorted(range(min(1 << g, 256)),
                key=lambda x: (min(x ^ d for d in D), x))
    blocks += [tv(x) for x in vs]
    return blocks


# ---------------------------------------------------------------------------
# the tower
# ---------------------------------------------------------------------------


_TOWER_CACHE = {}


def tower(m, s, t):
    """The master block multiset for row count m (see module doc), as
    (block, generator_index, ladder_distance, op, seq) records.  realize()
    linearizes it per LEADER: one symmetric generator class walks first
    (the thesis: each extremal matrix is one clean orbit family plus
    canonical completion), the rest follow."""
    key = (m, s, t)
    if key in _TOWER_CACHE:
        return _TOWER_CACHE[key]
    emitted = []
    seen = set()

    def emit(b, gi, dist, op):
        b = tuple(sorted(x for x in b if 0 <= x < m))
        if len(b) < s or len(b) > m:
            return
        if b in seen:
            return
        seen.add(b)
        emitted.append((b, gi, dist, op, len(emitted)))

    # label ladder: m..m+3, plus the next power of two >= m (the Boolean
    # frame) with two canonical embeddings: identity-prefix and
    # Hamming-weight order (the classical point orders)
    ladder = [(d, mp, None) for d, mp in enumerate(range(m, m + 4))]
    p2 = 1
    while p2 < m:
        p2 <<= 1
    if p2 <= 16 and p2 not in [mp for _, mp, _ in ladder]:
        ladder.append((p2 - m, p2, None))
    if p2 <= 16:
        wt = sorted(range(1, p2), key=lambda x: (bin(x).count("1"), x))
        wmap = {lab: i for i, lab in enumerate(wt)}
        ladder.append((p2 - m + 1, p2, wmap))

    for dist, mp, lmap in ladder:
        gens = (("EA", _ea_blocks(mp, s, t)),
                ("PTD", _ptd_blocks(mp, s, t) if lmap is None else []),
                ("PRG", _prg_blocks(mp, s, t)
                 if lmap is None and mp == m else []),
                ("CYC", _cyc_blocks(mp, s, t)
                 if dist == 0 and lmap is None else []))
        base_point = 0
        full = set(range(mp))
        for gi, (gname, blocks) in enumerate(gens):
            if not blocks:
                continue
            if lmap is not None:
                blocks = [tuple(sorted(lmap[x] for x in b if x in lmap))
                          for b in blocks]
                blocks = [b for b in blocks if len(b) >= s]
            level1 = list(dict.fromkeys(blocks))
            for b in level1:                              # O1 orbit
                emit(b, gi, dist, 0)
            for b in level1:                              # O2 complement
                emit(tuple(sorted(full - set(b))), gi, dist, 1)
            ext = []
            for b in level1:                              # O3 extension
                if base_point not in b:
                    eb = tuple(sorted(set(b) | {base_point}))
                    ext.append(eb)
                    emit(eb, gi, dist, 2)
            for eb in ext:                                # O2 o O3
                emit(tuple(sorted(full - set(eb))), gi, dist, 3)
            thru = [b for b in level1
                    if base_point in b and len(b) <= s + 1]
            for i in range(len(thru)):                    # O4 fusion (+O3)
                for j in range(i + 1, len(thru)):
                    fused = (set(thru[i]) | set(thru[j])) - {base_point}
                    if len(fused) <= s + 3:
                        emit(tuple(sorted(fused)), gi, dist, 4)
                        emit(tuple(sorted(fused | {base_point})),
                             gi, dist, 5)

    _TOWER_CACHE[key] = emitted
    return emitted


_LIN_CACHE = {}


def linearize(m, s, t, mode):
    """Canonical linearizations of the one tower multiset:
      mode 0..3  leader-major: that generator class entirely first
                 (size desc within), then the rest;
      mode "gen"    size-major, ties by (generator, ladder distance);
      mode "ladder" size-major, ties by (ladder distance, generator) —
                 lets same-size blocks of different generators interleave.
    """
    key = (m, s, t, mode)
    if key in _LIN_CACHE:
        return _LIN_CACHE[key]
    recs = tower(m, s, t)
    if mode == "gen":
        order = sorted(recs, key=lambda r: (-len(r[0]), r[1], r[2],
                                            r[3], r[4]))
    elif mode == "ladder":
        order = sorted(recs, key=lambda r: (-len(r[0]), r[2], r[1],
                                            r[3], r[4]))
    else:
        order = sorted(recs, key=lambda r: (r[1] != mode, -len(r[0]),
                                            r[2], r[3], r[4]))
    out = [r[0] for r in order]
    _LIN_CACHE[key] = out
    return out


# ---------------------------------------------------------------------------


def profile(m, n, s, t):
    """Two-sided waterfill size quotas (the supply dial's center)."""
    return Z.waterfill_profile(m, n, s, t)


def _realize_once(m, n, s, t, twr, heavy_quota, size_cap, double,
                  use_p2=True):
    cap = Z._new_cap(m, s, t)
    blocks = []
    heavy = 0
    passes = max(1, t - 1) if double else 1
    # phase 1: heavy structured blocks — one copy per pass over the tower
    # (a full orbit before any doubling)
    for _pass in range(passes):
        for b in twr:
            if len(blocks) >= n or heavy >= heavy_quota:
                break
            if len(b) <= s + 1 or len(b) > size_cap:
                continue
            if Z._block_fits(cap, b, s):
                Z._add_block(cap, b, s)
                blocks.append(b)
                heavy += 1
    # phase 2: structured (s+1)-blocks, capped by the Lemma-A trade
    # (an (s+1)-block spends s more capacity than a repeated s-block earns)
    B = (t - 1) * comb(m, s)
    k_lim = max(0, (B - n) // s) if use_p2 else 0
    k2 = 0
    for _pass in range(passes):
        for b in twr:
            if len(blocks) >= n or k2 >= k_lim:
                break
            if len(b) != s + 1:
                continue
            if Z._block_fits(cap, b, s):
                Z._add_block(cap, b, s)
                blocks.append(b)
                k2 += 1
    seed = list(blocks)
    best = None
    for use, order, grow in ((True, "exact", False), (True, "xor", True),
                             (True, "lex", False), (False, "lex", False)):
        if order == "exact" and comb(m, s + 1) > 350:
            continue
        done = Z.complete_blocks(m, n, s, t, seed, use_splus1=use,
                                 order=order, grow=grow)
        if done is None:
            continue
        e = sum(len(x) for x in done)
        if best is None or e > best[0]:
            best = (e, done)
    return best


def construct_unified(m, n, s, t):
    """The single-rule construction.  Returns (matrix, provenance)."""
    if s <= 0 or t <= 0:
        return None, "unified: degenerate"
    if m < s or n < t:
        return np.ones((m, n), dtype=int), "unified: trivial all-ones"
    if comb(m, s) > 60000:
        return None, "unified: capacity frame too large"
    prof = profile(m, n, s, t)
    center = sum(1 for c in prof if c > s + 1)
    quotas = sorted({max(0, center + d) for d in (-3, -2, -1, 0, 1, 2, 3)}
                    | {0, n})
    caps = sorted({c for c in (prof[0] - 1, prof[0], prof[0] + 1)
                   if s <= c <= m})                 # the supply dial's cap
    if not caps:
        caps = [m]
    best = (-1, None, 0, 0, 0)
    for leader in (0, 1, 2, 3, "gen", "ladder"):
        twr = linearize(m, s, t, leader)
        for sc in caps:
            for q in quotas:
                for dbl in (False, True):
                    for p2 in (True, False):
                        got = _realize_once(m, n, s, t, twr, q, sc, dbl,
                                            use_p2=p2)
                        if got and got[0] > best[0]:
                            best = (got[0], got[1], q, sc, leader)
    if best[1] is None:
        return None, "unified: nothing realizable"
    A = Z.blocks_to_matrix(m, n, best[1])
    if not Z.verify_kst_free(A, s, t):  # belt and braces; frame guarantees it
        return None, "unified: verification failed (bug)"
    leader_name = best[4] if isinstance(best[4], str) else \
        ("EA", "PTD", "PRG", "CYC")[best[4]]
    return A, (f"unified: leader={leader_name} quota={best[2]} "
               f"cap={best[3]} edges={best[0]}")


# ---------------------------------------------------------------------------
# the honest comparison
# ---------------------------------------------------------------------------


def price_of_unification(csv_path=None):
    """For every suite cell: unified vs router edges vs proven exact.
    Writes price_of_unification.csv; prints the headline."""
    import csv as _csv
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "snap", os.path.join(_HERE, "..", "harness", "evaluator_snapshot.py"))
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    cells = sorted(ev.KST_EXACT_VALUE.items())
    rows, u_exact, r_exact, losses = [], 0, 0, []
    for (m, n), z in cells:
        A_u, prov_u = construct_unified(m, n, 3, 3)
        e_u = int(A_u.sum()) if A_u is not None else 0
        ok_u = A_u is not None and ev.count_kst_violations(A_u, 3, 3) == 0
        A_r, prov_r, _ = Z.construct(m, n, 3, 3)
        e_r = int(A_r.sum())
        u_exact += (e_u == z and ok_u)
        r_exact += (e_r == z)
        if e_u < e_r:
            losses.append((m, n, e_r - e_u))
        rows.append({"m": m, "n": n, "exact": z, "unified": e_u,
                     "router": e_r, "unified_valid": int(ok_u),
                     "price": e_r - e_u,
                     "unified_prov": prov_u, "router_prov": prov_r[:60]})
    path = csv_path or os.path.join(_HERE, "price_of_unification.csv")
    with open(path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    total_price = sum(r["price"] for r in rows if r["price"] > 0)
    print(f"unified exact {u_exact}/161, router exact {r_exact}/161; "
          f"cells where unified < router: {len(losses)} "
          f"(total price {total_price} edges)")
    for m, n, d in losses:
        print(f"  {m}x{n}: -{d}")
    return rows


if __name__ == "__main__":
    for (m, n) in [(9, 22), (12, 22), (16, 16), (8, 17), (7, 15), (5, 9)]:
        A, prov = construct_unified(m, n, 3, 3)
        print(f"{m}x{n}: {int(A.sum())} free={Z.verify_kst_free(A, 3, 3)} "
              f"[{prov}]")
