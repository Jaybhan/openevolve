# Scripted Lean snippets (no-LLM bank)

Each `<name>.lean` is a complete `LEAN_SOURCE` body (it ends with `def candidate`);
an optional `<name>.py` beside it is the replacement for the Python `kill` region
(`def kill(...)` up to `# EVOLVE-BLOCK-END`).  `tools/stub_llm.py` and
`experiments/no_llm/replay_llm.py` turn them into SEARCH/REPLACE diffs against the
program shown in the prompt.  Expected gate outcome (design §4.2 ladder):

| snippet | expected | note |
|---|---|---|
| `deletion` | L5 | delete a column, bound the remainder by the ROW-side waterfill (new prune) |
| `schema_farkas` | L5 | summed column+row budgets (a Farkas combination of Arguments A / Aᵀ) |
| `residue_sketch` | L5 after auto-fill (measured: both holes closed by `apply colBudget/rowBudget <;> exact hv.1`); L2 if the fill list shrinks | same statement, both budget facts left as `have … := by sorry` holes |
| `broken_proof` | L1 | same statement, wrong final tactic |
| `unsound_row7` | battery 0 | Lean unchanged, Python kill fires on a witnessed case |
| `dgh4` | L5 | the hand-proved Davies-Gill-Horsley prune (`tests/candidates/lean_dgh4.py`, F-dgh4): `argDGH` inlined, `candidate := ofList [counting P, argDGH P]`; proven_gain > 0 only on DGH-attackable (wide) cells |
