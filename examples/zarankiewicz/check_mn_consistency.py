#!/usr/bin/env python3
"""
Audit every zarankiewicz_m,n directory and verify that the M/N values in
evaluator.py, initial_program.py, config_phase_1.yaml, config_phase_2.yaml,
and config_phase_3.yaml all match the m,n encoded in the folder name.

Usage:
    python check_mn_consistency.py
    python check_mn_consistency.py --fix   # auto-patch mismatched values
"""

import argparse
import os
import re
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
SKIP = {"known_bounds", "successes", "to_be_improved", "extensively_tested"}

# ── helpers ──────────────────────────────────────────────────────────────────

def parse_folder_mn(name: str):
    """Return (M, N) from a folder name like 'zarankiewicz_12,17*' or None."""
    m = re.search(r"zarankiewicz_(\d+),(\d+)", name)
    return (int(m.group(1)), int(m.group(2))) if m else None


def extract_py_mn(path: str):
    """
    Read M = <int> and N = <int> from a Python file.
    Returns (M, N) tuple of ints, or (None, None) if not found.
    """
    M = N = None
    try:
        with open(path) as f:
            for line in f:
                mm = re.match(r"^\s*M\s*=\s*(\d+)", line)
                mn = re.match(r"^\s*N\s*=\s*(\d+)", line)
                if mm:
                    M = int(mm.group(1))
                if mn:
                    N = int(mn.group(1))
    except OSError:
        pass
    return M, N


def extract_yaml_mn(path: str):
    """
    Scan the system_message block of a config YAML for patterns like:
        16×19, 16x19, 16 x 19, or z(16,19;...)
    Returns the first (M, N) pair found, or (None, None).
    """
    patterns = [
        r"(\d+)[×x]\s*(\d+)",       # 16×19 or 16x19
        r"z\s*\(\s*(\d+)\s*,\s*(\d+)\s*[;,]",  # z(16,19;...)
    ]
    try:
        with open(path) as f:
            text = f.read()
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                return int(m.group(1)), int(m.group(2))
    except OSError:
        pass
    return None, None


# ── fix helpers ───────────────────────────────────────────────────────────────

def fix_py_mn(path: str, correct_m: int, correct_n: int, current_m, current_n) -> bool:
    """Replace M = <wrong> and N = <wrong> in a Python file. Returns True if changed."""
    try:
        with open(path) as f:
            text = f.read()
    except OSError:
        return False

    new_text = re.sub(r"^(\s*M\s*=\s*)\d+", lambda mo: mo.group(1) + str(correct_m), text, flags=re.MULTILINE)
    new_text = re.sub(r"^(\s*N\s*=\s*)\d+", lambda mo: mo.group(1) + str(correct_n), new_text, flags=re.MULTILINE)

    if new_text != text:
        with open(path, "w") as f:
            f.write(new_text)
        return True
    return False


def fix_yaml_mn(path: str, correct_m: int, correct_n: int) -> bool:
    """Replace MxN / z(M,N;...) patterns in a YAML file. Returns True if changed."""
    try:
        with open(path) as f:
            text = f.read()
    except OSError:
        return False

    new_text = re.sub(
        r"(\d+)([×x]\s*)(\d+)",
        lambda mo: str(correct_m) + mo.group(2) + str(correct_n),
        text,
    )
    new_text = re.sub(
        r"(z\s*\(\s*)(\d+)(\s*,\s*)(\d+)(\s*[;,])",
        lambda mo: mo.group(1) + str(correct_m) + mo.group(3) + str(correct_n) + mo.group(5),
        new_text,
    )

    if new_text != text:
        with open(path, "w") as f:
            f.write(new_text)
        return True
    return False


# ── main audit ────────────────────────────────────────────────────────────────

def audit(fix: bool = False):
    all_ok = True
    folders = sorted(
        [
            name for name in os.listdir(BASE)
            if os.path.isdir(os.path.join(BASE, name))
            and name not in SKIP
            and parse_folder_mn(name) is not None
        ],
        key=lambda n: parse_folder_mn(n),
    )

    for folder_name in folders:
        folder_path = os.path.join(BASE, folder_name)
        expected = parse_folder_mn(folder_name)
        if expected is None:
            continue
        exp_m, exp_n = expected

        files = {
            "evaluator.py":       (extract_py_mn,   os.path.join(folder_path, "evaluator.py")),
            "initial_program.py": (extract_py_mn,   os.path.join(folder_path, "initial_program.py")),
            "config_phase_1.yaml":(extract_yaml_mn, os.path.join(folder_path, "config_phase_1.yaml")),
            "config_phase_2.yaml":(extract_yaml_mn, os.path.join(folder_path, "config_phase_2.yaml")),
            "config_phase_3.yaml":(extract_yaml_mn, os.path.join(folder_path, "config_phase_3.yaml")),
        }

        folder_ok = True
        issues = []

        for label, (extractor, fpath) in files.items():
            if not os.path.exists(fpath):
                issues.append(f"  MISSING  {label}")
                folder_ok = False
                continue

            got_m, got_n = extractor(fpath)

            if got_m is None or got_n is None:
                issues.append(f"  PARSE-FAIL  {label}  (could not extract M/N)")
                folder_ok = False
                continue

            if (got_m, got_n) != (exp_m, exp_n):
                issues.append(
                    f"  MISMATCH  {label}  "
                    f"found ({got_m},{got_n})  expected ({exp_m},{exp_n})"
                )
                folder_ok = False

                if fix:
                    if label.endswith(".py"):
                        changed = fix_py_mn(fpath, exp_m, exp_n, got_m, got_n)
                    else:
                        changed = fix_yaml_mn(fpath, exp_m, exp_n)
                    issues[-1] += "  →  FIXED" if changed else "  →  FIX FAILED"

        if folder_ok:
            print(f"OK  {folder_name}")
        else:
            all_ok = False
            print(f"FAIL  {folder_name}  (expected M={exp_m}, N={exp_n})")
            for issue in issues:
                print(issue)

    print()
    if all_ok:
        print("All directories consistent.")
    else:
        print("Inconsistencies found (see above).")
        if not fix:
            print("Re-run with --fix to auto-patch mismatched values.")
    return all_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit zarankiewicz_m,n M/N consistency.")
    parser.add_argument("--fix", action="store_true", help="Auto-patch mismatched values.")
    args = parser.parse_args()

    ok = audit(fix=args.fix)
    sys.exit(0 if ok else 1)
