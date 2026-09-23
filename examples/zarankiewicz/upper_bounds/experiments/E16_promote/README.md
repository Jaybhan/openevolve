# E16 — promotion (§4.4) and the closure daemon (§7) — 2026-09-21, key H-promote

**Question.** Does a gate-accepted program become a trusted, replayed library module, and does one
daemon pass take an OpenEvolve checkpoint to a certified bound?

**Run.**
```bash
cd examples/zarankiewicz/upper_bounds; PY=../../../.venv/bin/python; export ZAR_UB_NO_LLM=1
$PY -m zar_ub closure-daemon --checkpoint-dir experiments/no_llm/run_stub --targets "9,9,50" \
    --poll 50 --budget 5e8 --now --pure              # -> daemon_run_stub_9_9_50.log
$PY -m unittest -v tests.test_promote                # -> test_promote.log
$PY -m zar_ub verify-certs cache/certs/m9_n9_s3_t3_w50 --fresh-lratcheck
$PY -m zar_ub audit-kills 9 9 3 3 50 --frac 0.05 --pure
```
Files here: `daemon_run_stub_9_9_50.log` (the daemon pass, verbatim), `test_promote.log` (the unit tests).
Results and numbers: `docs/build/H-promote.md`; the launch bullet: `experiments/LOG.md` §E16.
