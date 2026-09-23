"""The candidate subprocess must not be able to write into the project (gate cache, tables,
PIPELINE_BUG sentinel, ledger) nor read the gate-cache secret (docs/build/ATTACKS.md)."""
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
UB = os.path.dirname(HERE)
sys.path.insert(0, UB)
from zar_ub.sandbox import sandboxed_command, sandbox_available  # noqa: E402
from zar_ub.lean_gate import GATE_SECRET_PATH, _gate_secret  # noqa: E402

PROBE = r'''
import os, json, sys
out = {}
targets = {
  "gate": os.path.join(%(ub)r, "cache", "gate", "zz_sandbox_probe.json"),
  "sentinel": os.path.join(%(ub)r, "cache", "PIPELINE_BUG"),
  "table": os.path.join(%(ub)r, "cache", "zz_sandbox_probe_table.json"),
  "lean": os.path.join(%(ub)r, "lean", "ZarPrune", "ZzProbe.lean"),
  "tmp": os.path.join(%(tmp)r, "allowed.txt"),
}
for k, p in targets.items():
    try:
        with open(p, "w") as f: f.write("x")
        out[k] = "WROTE"; os.remove(p)
    except OSError as e:
        out[k] = "denied"
try:
    open(%(secret)r, "rb").read(); out["secret"] = "READ"
except OSError:
    out["secret"] = "denied"
import socket
try:
    socket.create_connection(("1.1.1.1", 53), timeout=2); out["net"] = "OPEN"
except OSError:
    out["net"] = "denied"
print(json.dumps(out))
'''


class TestSandbox(unittest.TestCase):
    def test_candidate_cannot_write_project_or_read_secret(self):
        if not sandbox_available():
            self.skipTest("no OS sandbox on this machine (gate cache is disabled instead)")
        _gate_secret()  # make sure the secret exists
        tmp = tempfile.mkdtemp(prefix="zar_ub_sbx_")
        probe = os.path.join(tmp, "probe.py")
        with open(probe, "w") as f:
            f.write(PROBE % {"ub": UB, "tmp": tmp, "secret": GATE_SECRET_PATH})
        cmd, sandboxed, backend = sandboxed_command([sys.executable, probe], allow_write=[tmp], deny_read=[GATE_SECRET_PATH])
        self.assertTrue(sandboxed, backend)
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60, cwd=tmp)
        self.assertTrue(proc.stdout.strip(), proc.stderr[-500:])
        out = json.loads(proc.stdout.strip().splitlines()[-1])
        for k in ("gate", "sentinel", "table", "lean"):
            self.assertEqual(out[k], "denied", f"{k}: {out}")
        self.assertEqual(out["secret"], "denied", out)
        self.assertEqual(out["net"], "denied", out)
        self.assertEqual(out["tmp"], "WROTE", out)

    def test_forged_cache_entry_is_rejected(self):
        from zar_ub import lean_gate as G
        key = "0" * 40
        d = tempfile.mkdtemp(prefix="zar_ub_gc_")
        with open(os.path.join(d, key + ".json"), "w") as f:
            json.dump({"version": G.WRAPPER_VERSION, "key": key, "results": [], "mac": "deadbeef"}, f)
        self.assertIsNone(G._cache_load(key, d))
        self.assertFalse(os.path.exists(os.path.join(d, key + ".json")), "forged entry must be deleted")


if __name__ == "__main__":
    unittest.main()
