# EVOLVE-BLOCK-START
from typing import Optional


def find_isomorphism(G1: list, G2: list) -> Optional[list]:
    """
    Find a permutation mapping G1's vertices to G2's vertices such that
    G1[perm[i]][perm[j]] == G2[i][j] for all i, j.

    Args:
        G1: Adjacency matrix of first graph (n x n list of lists)
        G2: Adjacency matrix of second graph (n x n list of lists)

    Returns:
        A permutation list perm of length n where perm[i] is the vertex in G1
        that maps to vertex i in G2. Returns None if no isomorphism exists.
    """
    n = len(G1)
    # Trivial baseline: assume identity mapping works (G1 == G2)
    return list(range(n))


# EVOLVE-BLOCK-END


def solve(G1: list, G2: list) -> Optional[list]:
    """Public interface called by the evaluator."""
    return find_isomorphism(G1, G2)


if __name__ == "__main__":
    # Simple sanity check
    G1 = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    G2 = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    result = solve(G1, G2)
    print(f"Isomorphism found: {result}")
