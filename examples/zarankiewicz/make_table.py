#!/usr/bin/env python3
"""
Print a markdown table of all top-level zarankiewicz folders showing
n_sota (current best) and KST_UPPER_BOUND from each evaluator.py.
"""

import os
import re

SKIP = {"known_bounds", "successes", "to_be_improved", "extensively_tested"}
BASE = os.path.dirname(os.path.abspath(__file__))


def parse_mn(folder_name):
    """Extract (M, N) ints from a folder name like 'zarankiewicz_12,17*'."""
    m = re.search(r"(\d+),(\d+)", folder_name)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def read_upper_bound(evaluator_path):
    try:
        with open(evaluator_path) as f:
            for line in f:
                m = re.match(r"\s*KST_UPPER_BOUND\s*=\s*(\d+)", line)
                if m:
                    return int(m.group(1))
    except OSError:
        pass
    return None


def read_n_sota(folder_path):
    try:
        with open(os.path.join(folder_path, ".n_sota")) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


rows = []
for name in os.listdir(BASE):
    if name in SKIP:
        continue
    folder = os.path.join(BASE, name)
    if not os.path.isdir(folder):
        continue
    evaluator = os.path.join(folder, "evaluator.py")
    if not os.path.exists(evaluator):
        continue

    m, n = parse_mn(name)
    upper = read_upper_bound(evaluator)
    n_sota = read_n_sota(folder)
    rows.append((m, n, name, n_sota, upper))

rows.sort(key=lambda r: (r[0], r[1]))

header = f"{'Folder':<30} {'n_sota':>8} {'Upper Bound':>12}"
sep    = f"{'-'*30} {'-'*8} {'-'*12}"
print(header)
print(sep)
for m, n, name, n_sota, upper in rows:
    sota_str  = str(n_sota)  if n_sota  is not None else "—"
    upper_str = str(upper)   if upper   is not None else "?"
    flag = " =" if (n_sota is not None and upper is not None and n_sota == upper) else ""
    print(f"{name:<30} {sota_str:>8} {upper_str:>12}{flag}")
