import ZarPrune.Cond

/-!
Evolved prune `E_c285948723e7` — promoted 2026-09-21 23:37 by zar_ub/promote.py (design §4.4).
name: ckpt10_bf2acfca
origin: {"checkpoint": 10, "iteration": 0, "program_id": "bf2acfca-daca-4710-b79f-fa3c66ab1328", "program_path": "checkpoint_10/programs/bf2acfca-daca-4710-b79f-fa3c66ab1328.json", "run": "experiments/no_llm/run_stub"}
The candidate source between the namespace lines is VERBATIM the text the gate accepted
(sha1 of the normalised text = the module name).  Do not edit; re-promote instead.
-/
set_option autoImplicit false

namespace ZarPrune.Evolved.E_c285948723e7

/-- Pattern for a NEW prune: `kill` on the profile, `sound` via library lemmas. This one kills nothing. -/
def examplePrune (P : Params) : Prune P where
  name := "example (kills nothing)"
  kill := fun _ => false
  sound := by intro A h; simp at h

/-- The evolved library. Keep `counting P` first; append new prunes. -/
def candidate (P : Params) : Prune P :=
  Prune.ofList P [counting P, examplePrune P]

end ZarPrune.Evolved.E_c285948723e7
