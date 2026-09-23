"""OS-level sandbox for the untrusted candidate subprocess (stage 1).

The candidate's Python is LLM-written.  Without a sandbox it can write anywhere the
user can: forge the Lean-gate cache, tamper with the live case tables, plant or
remove the PIPELINE_BUG sentinel, poison the promotion ledger (docs/build/ATTACKS.md).
`sandboxed_command` wraps a command so that the child may only WRITE inside the
directories listed in `allow_write`, may not READ the gate-cache secret, and has
no network:

  * macOS: `sandbox-exec` with a Scheme profile (deny file-write* by default,
    allow the temp dir; deny file-read* of the secret; deny network*).
  * Linux: `bwrap` (bubblewrap) when installed: read-only root, the temp dir bound
    read-write, secret masked, --unshare-net.
  * otherwise: the command runs unsandboxed and `sandboxed` is False; callers must
    then disable every persistent side channel (the evaluator switches the gate
    cache off and warns once).
"""
from __future__ import annotations

import os
import shutil
import sys
from typing import List, Sequence, Tuple


def _q(p: str) -> str:
    return '"' + os.path.realpath(p).replace('"', '\\"') + '"'


def sandboxed_command(cmd: Sequence[str], allow_write: Sequence[str], deny_read: Sequence[str] = (),
                      deny_network: bool = True) -> Tuple[List[str], bool, str]:
    """Return (command, sandboxed, backend)."""
    cmd = list(cmd)
    if sys.platform == "darwin" and shutil.which("sandbox-exec"):
        lines = ["(version 1)", "(allow default)", "(deny file-write*)"]
        for p in allow_write:
            lines.append(f"(allow file-write* (subpath {_q(p)}))")
        # pipes/ttys are not files, but Python may touch these device nodes
        lines.append('(allow file-write* (literal "/dev/null"))')
        lines.append('(allow file-write* (regex #"^/dev/tty"))')
        for p in deny_read:
            lines.append(f"(deny file-read* (literal {_q(p)}))")
        if deny_network:
            lines.append("(deny network*)")
        return ["sandbox-exec", "-p", "\n".join(lines)] + cmd, True, "sandbox-exec"
    if sys.platform.startswith("linux") and shutil.which("bwrap"):
        w = ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--die-with-parent"]
        for p in allow_write:
            rp = os.path.realpath(p)
            w += ["--bind", rp, rp]
        for p in deny_read:
            rp = os.path.realpath(p)
            if os.path.exists(rp):
                w += ["--tmpfs", rp] if os.path.isdir(rp) else ["--ro-bind", "/dev/null", rp]
        if deny_network:
            w += ["--unshare-net"]
        return w + ["--"] + cmd, True, "bwrap"
    return cmd, False, "none"


def sandbox_available() -> bool:
    return sandboxed_command(["true"], [os.getcwd()])[1]
