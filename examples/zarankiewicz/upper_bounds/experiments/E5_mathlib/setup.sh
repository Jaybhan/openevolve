#!/usr/bin/env bash
# E5: how expensive is a Mathlib-backed Lean gate?  Creates a throwaway lake
# project depending on Mathlib (prebuilt oleans via `lake exe cache get`), then
# times checking a file that imports Mathlib.  Everything here is gitignored.
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d mathlib_probe ]; then
  lake new mathlib_probe math   # 'math' template = depends on Mathlib
fi
cd mathlib_probe
cat lean-toolchain
( time lake exe cache get ) 2>&1 | tail -5
cat > Probe.lean <<'LEAN'
import Mathlib
open Finset in
example (m : ℕ) : ∑ i ∈ range m, (1:ℕ) = m := by simp
theorem probe_choose (n : ℕ) : Nat.choose n 2 = n * (n - 1) / 2 := Nat.choose_two_right n
LEAN
echo "--- timing import Mathlib check (cold) ---"
( time lake env lean Probe.lean ) 2>&1 | tail -4
echo "--- timing (warm) ---"
( time lake env lean Probe.lean ) 2>&1 | tail -4
