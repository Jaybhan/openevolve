#!/usr/bin/env python3
"""Query OpenRouter for the key's usage (free endpoint) and append a line to
experiments/cost_ledger.md.  Usage: python experiments/cost.py "<label>"
"""
import json, os, sys, time, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__))
key = open(os.path.join(HERE, "..", ".openrouter_key")).read().strip()
req = urllib.request.Request("https://openrouter.ai/api/v1/auth/key", headers={"Authorization": f"Bearer {key}"})
d = json.load(urllib.request.urlopen(req))["data"]
label = sys.argv[1] if len(sys.argv) > 1 else "check"
line = f"| {time.strftime('%Y-%m-%d %H:%M')} | {label} | ${d['usage']:.4f} | ${16 - d['usage']:.4f} |\n"
path = os.path.join(HERE, "cost_ledger.md")
if not os.path.exists(path):
    open(path, "w").write("# OpenRouter spend ledger (hard cap $16)\n\n| time | label | total spent | remaining |\n|---|---|---|---|\n")
open(path, "a").write(line)
print(line.strip())
