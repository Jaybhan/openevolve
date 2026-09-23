-- SNIPPET:unsound_row7
/-- Lean side unchanged (the proved library); the Python mirror in the companion
`.py` adds an UNSOUND rule ("kill any case with a row of sum >= 7") that fires on a
witnessed case of the battery.  Expected: sound_battery = 0, score 0. -/
def candidate (P : Params) : Prune P := Prune.or (baseline P) (counting P)
