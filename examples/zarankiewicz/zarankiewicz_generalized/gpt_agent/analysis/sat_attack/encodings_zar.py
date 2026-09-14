"""CNF encodings for Zarankiewicz z(m,n;3,3) SAT attacks.

Matrix model (targets 1, 2, 4):
  Boolean grid x[r][c], r < m rows, c < n columns.
  K_{3,3}-freeness == for every triple T of rows, at most 2 columns contain
  all three ones.  Per (T, c) an auxiliary y with (x_a & x_b & x_c) -> y and
  AtMost2 over the column y's of each triple (one-directional y is sound AND
  complete for the <=2 constraint: y is only *forced* true when the triple is
  fully present, and may be chosen false otherwise).
  Cardinality: total number of ones == target (equivalent to >= target for
  the decision question, since deleting ones preserves legality; the upper
  side is free extra propagation).
  Symmetry breaking (both simultaneously sound: first permute rows so degrees
  are non-increasing -- degrees are column-permutation invariant -- then sort
  columns lexicographically):
    * adjacent-column lex order  col_c >=_lex col_{c+1}  (prefix-equality aux)
    * row-degree monotonicity via exact two-directional sequential counters.

T_{3,3}(v) packing model (target 3):
  One pair of Booleans (a_B, b_B) per 4-subset B of [v]; multiplicity a+b,
  with b -> a breaking copy symmetry.  Per 3-subset t: exact unary counter
  (levels 1,2 + saturation clause forbidding a 3rd true) over the 2*(v-3)
  incident literals, yielding exact indicators g1_t (cov>=1), g2_t (cov>=2).
  Sum of coverage = 4*T identically, so T >= target  <=>  deficit
  sum_t (2 - cov t) <= 2*C(v,3) - 4*target, a small-bound cardinality over
  the 2*C(v,3) negated indicators -- avoiding a monster cardinality over all
  7752 block literals.
"""

from itertools import combinations

from pysat.card import CardEnc, EncType


# --------------------------------------------------------------------------
# exact (two-directional) sequential unary counter
# --------------------------------------------------------------------------

def exact_unary_counter(lits, top, clauses, max_level=None, saturate_at=None):
    """Sequential counter with full iff semantics.

    Returns (top, outs) where outs[k-1] is a literal exactly equivalent to
    "at least k of lits are true", for k = 1..K (K = max_level or len(lits)).
    If saturate_at = s is given, also forbids more than s true inputs
    (requires max_level >= s).
    """
    n = len(lits)
    K = n if max_level is None else min(max_level, n)
    prev = {}  # level -> register var of prefix i-1
    for i, l in enumerate(lits, start=1):
        cur = {}
        for k in range(1, min(i, K) + 1):
            top += 1
            s = top
            cur[k] = s
            s_pk = prev.get(k)          # "prefix has >= k"; None == false
            s_pk1 = prev.get(k - 1) if k > 1 else True  # ">= 0" == true
            # forward: s_pk -> s ; (s_pk1 & l) -> s
            if s_pk is not None:
                clauses.append([-s_pk, s])
            if s_pk1 is True:
                clauses.append([-l, s])
            elif s_pk1 is not None:
                clauses.append([-s_pk1, -l, s])
            # backward: s -> s_pk | (s_pk1 & l)
            base = [-s] + ([s_pk] if s_pk is not None else [])
            if s_pk1 is not True and s_pk1 is not None:
                clauses.append(base + [s_pk1])
            if s_pk1 is not None:       # s_pk1 None: that disjunct is false
                clauses.append(base + [l])
            else:
                clauses.append(base)
        if saturate_at is not None and prev.get(saturate_at) is not None:
            # already saturate_at trues in the prefix -> l must be false
            clauses.append([-prev[saturate_at], -l])
        prev = cur
    outs = [prev.get(k) for k in range(1, K + 1)]
    return top, outs


# --------------------------------------------------------------------------
# matrix encoder
# --------------------------------------------------------------------------

def encode_matrix(m, n, total, card_mode="equals", col_lex=True,
                  row_degmono=True, card_enc=EncType.totalizer,
                  col_weights=None, pair_cuts=False,
                  col_weight_window=None, max_weight_exact=False):
    """Encode: exists m x n 0/1 matrix, K_{3,3}-free, with `total` ones.

    col_weights: optional non-increasing list of n exact column weights
    summing to `total` (cube-and-conquer mode).  WLOG for the >=total
    decision question: delete ones to reach total, raise weight<2 columns
    to 2 (weight<=2 columns cover no triple), sort rows by degree, then
    columns by (weight desc, lex desc).  In this mode the global cardinality
    is dropped (implied) and lex ordering applies within equal-weight runs.

    Returns (clauses, nvars, X) with X[r][c] = grid variable id.
    """
    clauses = []
    X = [[r * n + c + 1 for c in range(n)] for r in range(m)]
    top = m * n
    if col_weights is not None:
        assert len(col_weights) == n and sum(col_weights) == total
        assert all(col_weights[i] >= col_weights[i + 1]
                   for i in range(n - 1))

    # --- triple constraints -------------------------------------------------
    for (a, b, c3) in combinations(range(m), 3):
        ys = []
        for c in range(n):
            top += 1
            y = top
            ys.append(y)
            clauses.append([-X[a][c], -X[b][c], -X[c3][c], y])
        cnf = CardEnc.atmost(lits=ys, bound=2, top_id=top,
                             encoding=EncType.seqcounter)
        clauses.extend(cnf.clauses)
        top = max(top, cnf.nv)

    # --- global cardinality / fixed column weights --------------------------
    if col_weights is not None:
        for c in range(n):
            w = col_weights[c]
            top, outs = exact_unary_counter(
                [X[r][c] for r in range(m)], top, clauses,
                max_level=min(w + 1, m))
            clauses.append([outs[w - 1]])          # weight >= w
            if w < m:
                clauses.append([-outs[w]])         # weight < w+1
    else:
        all_x = [X[r][c] for r in range(m) for c in range(n)]
        if card_mode == "equals":
            cnf = CardEnc.equals(lits=all_x, bound=total, top_id=top,
                                 encoding=card_enc)
        elif card_mode == "atleast":
            cnf = CardEnc.atleast(lits=all_x, bound=total, top_id=top,
                                  encoding=card_enc)
        else:
            raise ValueError(card_mode)
        clauses.extend(cnf.clauses)
        top = max(top, cnf.nv)

    # --- symmetry: adjacent-column lex (col_c >=_lex col_{c+1}) ------------
    if col_lex:
        for c in range(n - 1):
            if col_weights is not None and \
                    col_weights[c] != col_weights[c + 1]:
                continue  # lex only within equal-weight runs
            p_prev = None  # None == prefix trivially equal (row 0)
            for i in range(m):
                a1, b1 = X[i][c], X[i][c + 1]
                cond = [] if p_prev is None else [-p_prev]
                clauses.append(cond + [a1, -b1])       # prefix equal -> a>=b
                if i < m - 1:
                    top += 1
                    p = top
                    clauses.append(cond + [a1, b1, p])     # both 0 -> p
                    clauses.append(cond + [-a1, -b1, p])   # both 1 -> p
                    p_prev = p

    # --- column weight window / max-weight cube -----------------------------
    # col_weight_window=(lo,hi): every column weight in [lo,hi] (units on
    # exact unary counters; lo=2 is WLOG for the >=total question, hi from
    # the sum C(w,3) <= 2C(m,3) budget).  max_weight_exact: additionally
    # require some column to reach hi (cube "max weight == hi").
    if col_weight_window is not None and col_weights is None:
        lo, hi = col_weight_window
        tops_at_hi = []
        for c in range(n):
            top, outs = exact_unary_counter(
                [X[r][c] for r in range(m)], top, clauses,
                max_level=min(hi + 1, m))
            if lo >= 1:
                clauses.append([outs[lo - 1]])
            if hi < m:
                clauses.append([-outs[hi]])
            tops_at_hi.append(outs[hi - 1])
        if max_weight_exact:
            clauses.append(tops_at_hi)  # some column has weight >= hi

    # --- pair cutting planes (cube mode only) -------------------------------
    # For rows r,s: sum over columns containing both of (w_c - 2) equals
    # sum_t cov(r,s,t) <= 2(m-2).  One-directional AND indicators (forced
    # true only when both cells are 1) keep the cut sound and complete.
    if pair_cuts and col_weights is not None:
        for r, s in combinations(range(m), 2):
            lits = []
            for c in range(n):
                wc = col_weights[c]
                if wc <= 2:
                    continue
                top += 1
                p = top
                clauses.append([-X[r][c], -X[s][c], p])
                lits.extend([p] * (wc - 2))
            cnf = CardEnc.atmost(lits=lits, bound=2 * (m - 2), top_id=top,
                                 encoding=EncType.seqcounter)
            clauses.extend(cnf.clauses)
            top = max(top, cnf.nv)

    # --- symmetry: non-increasing row degrees -------------------------------
    if row_degmono:
        U = []
        for r in range(m):
            top, outs = exact_unary_counter(
                [X[r][c] for c in range(n)], top, clauses)
            U.append(outs)
        for r in range(m - 1):
            for k in range(n):
                clauses.append([-U[r + 1][k], U[r][k]])  # deg r+1>=k -> deg r>=k
    return clauses, top, X


def decode_matrix(model, m, n):
    """Grid values from a model (list of signed ints, index i -> var i+1)."""
    pos = set(l for l in model if l > 0)
    return [[1 if (r * n + c + 1) in pos else 0 for c in range(n)]
            for r in range(m)]


# --------------------------------------------------------------------------
# T_{3,3}(v) packing encoder
# --------------------------------------------------------------------------

def encode_t33(v, target, fix_first_block=True):
    """Encode: exists multiset of 4-subsets of [v], multiplicities <= 2,
    every 3-subset covered <= 2, with >= target blocks.

    Returns (clauses, nvars, blocks, avar, bvar).
    """
    blocks = list(combinations(range(v), 4))
    avar, bvar = {}, {}
    top = 0
    for B in blocks:
        top += 1
        avar[B] = top
    for B in blocks:
        top += 1
        bvar[B] = top
    clauses = [[avar[B], -bvar[B]] for B in blocks]      # b -> a

    trip_lits = {}
    for B in blocks:
        for T in combinations(B, 3):
            trip_lits.setdefault(T, []).extend((avar[B], bvar[B]))

    ntrip = len(trip_lits)
    deficit_lits = []
    for T in sorted(trip_lits):
        top, outs = exact_unary_counter(trip_lits[T], top, clauses,
                                        max_level=2, saturate_at=2)
        g1, g2 = outs
        deficit_lits.extend((-g1, -g2))

    D = 2 * ntrip - 4 * target        # allowed total deficit
    if D < 0:
        raise ValueError("target exceeds trivial coverage bound")
    cnf = CardEnc.atmost(lits=deficit_lits, bound=D, top_id=top,
                         encoding=EncType.seqcounter)
    clauses.extend(cnf.clauses)
    top = max(top, cnf.nv)

    if fix_first_block:
        clauses.append([avar[blocks[0]]])  # relabel: some block -> {0,1,2,3}
    return clauses, top, blocks, avar, bvar


def decode_t33(model, blocks, avar, bvar):
    pos = set(l for l in model if l > 0)
    out = []
    for B in blocks:
        mult = (1 if avar[B] in pos else 0) + (1 if bvar[B] in pos else 0)
        out.extend([list(B)] * mult)
    return out
