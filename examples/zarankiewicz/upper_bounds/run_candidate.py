"""Sandboxed runner for a candidate program (design §1.3 stage 1, §2.1).

Runs in a fresh subprocess started by `evaluator.py` (cwd = a temporary COPY of the cache
tables, 90 s wall-clock timeout, network sockets disabled below).  It imports the candidate,
extracts the four genome objects and applies `kill()` to every case the evaluator asks for.

stdin : JSON {"program": path, "cases": [[m, n, s, t, w, rows, cols], ...], "timeout": 90}
stdout: JSON {"lean_source": str, "schema_data": dict, "notes": str, "mask": [bool, ...], "error": str|null}

Nothing here is trusted: the Python mask feeds the counterexample battery and the empirical
band only; credit comes from the Lean gate's own #eval of the candidate's kill.
"""
import importlib.util
import json
import os
import signal
import socket
import sys
import traceback

TIMEOUT_S = 90


def _no_network():
    """Best-effort network block: any socket creation raises inside this process."""
    def _blocked(*_a, **_k):
        raise OSError("network access is disabled inside run_candidate.py")
    socket.socket = _blocked  # type: ignore[assignment]
    socket.create_connection = _blocked  # type: ignore[assignment]
    socket.socketpair = _blocked  # type: ignore[assignment]
    for k in list(os.environ):
        if "KEY" in k.upper() or "TOKEN" in k.upper() or "SECRET" in k.upper():
            os.environ.pop(k, None)
    os.environ["http_proxy"] = os.environ["https_proxy"] = "http://127.0.0.1:9"
    os.environ["HTTP_PROXY"] = os.environ["HTTPS_PROXY"] = "http://127.0.0.1:9"


def _alarm(_signum, _frame):
    raise TimeoutError(f"candidate exceeded {TIMEOUT_S}s")


def main():
    req = json.load(sys.stdin)
    out = {"lean_source": "", "schema_data": {}, "notes": "", "mask": [], "error": None}
    timeout = int(req.get("timeout", TIMEOUT_S))
    _no_network()
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(max(1, timeout))
    try:
        spec = importlib.util.spec_from_file_location("candidate_program", req["program"])
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        out["lean_source"] = str(getattr(mod, "LEAN_SOURCE", ""))
        out["notes"] = str(getattr(mod, "NOTES", ""))[:8000]
        sd = getattr(mod, "SCHEMA_DATA", {})
        try:
            out["schema_data"] = json.loads(json.dumps(sd))  # must be JSON-serialisable
        except (TypeError, ValueError) as e:
            out["schema_data"] = {}
            out["error"] = f"SCHEMA_DATA is not JSON-serialisable: {e}"
        kill = getattr(mod, "kill")
        mask = []
        for (m, n, s, t, w, rows, cols) in req["cases"]:
            mask.append(bool(kill(m, n, s, t, w, tuple(rows), tuple(cols))))
        out["mask"] = mask
    except BaseException:  # noqa: BLE001  (includes TimeoutError from the alarm and SystemExit)
        out["error"] = (out["error"] + "\n" if out["error"] else "") + traceback.format_exc()[-3000:]
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
    sys.stdout.write(json.dumps(out))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
