#!/bin/bash
# Live dashboard for a running openevolve job.
#   ./watch.sh            -> refreshes every 10s
#   ./watch.sh 30         -> refreshes every 30s
cd "$(dirname "$0")"
INTERVAL="${1:-10}"
T=terminal_phase3.log

while true; do
  clear
  echo "=== ZARANKIEWICZ RUN  $(date '+%H:%M:%S') ==================================="
  if [ ! -f "$T" ]; then echo "waiting for $T ..."; sleep "$INTERVAL"; continue; fi

  ALIVE=$(ps aux | grep -cE '[o]penevolve-run')
  DONE=$(grep -c 'completed in' $T)
  SPIRAL=$(grep -c 'Reasoning tail' $T)
  LOST=$(grep -c 'attempts failed' $T)
  LAST=$(grep -E '^2026' $T | tail -1 | cut -c12-19)
  echo "process: $([ "$ALIVE" -gt 0 ] && echo RUNNING || echo STOPPED)   completed: $DONE/50   spirals: $SPIRAL   lost: $LOST   last log: $LAST"
  echo

  echo "--- PROMPT SIZE PER CALL (grows as the database fills) ---"
  grep -oE '\-> CALL .*prompt [0-9]+ chars \(~[0-9]+ tok\)' $T | tail -5 \
    | sed -E 's/.*prompt ([0-9]+) chars \(~([0-9]+) tok\)/   \1 chars  ~\2 tokens/' || echo "   (none yet)"
  echo

  echo "--- COMPLETED CALLS: time and how much of the budget was used ---"
  grep -oE '<- DONE in [0-9]+s \| completion_tokens=[0-9]+ of [0-9]+ \| finish=[a-z_]+' $T | tail -5 \
    | sed -E 's/<- DONE in ([0-9]+)s \| completion_tokens=([0-9]+) of ([0-9]+) \| finish=(.*)/   \1s   used \2 of \3 tokens   finish=\4/' || echo "   (none yet)"
  echo

  echo "--- LIVE MODEL OUTPUT (newest streaming dump) ---"
  LATEST=$(ls -t prompt_dumps/*.txt 2>/dev/null | head -1)
  if [ -n "$LATEST" ]; then
    echo "   $LATEST"
    tail -20 "$LATEST"
  else
    echo "   (no streaming dump yet)"
  fi
  echo

  echo "--- PROGRESS ---"
  if [ -f instance_log.jsonl ]; then
    python3 - <<'PY'
import json, collections
try:
    r=[json.loads(l) for l in open('instance_log.jsonl')]
except Exception:
    r=[]
if r:
    b=max(r,key=lambda x:x['combined_score'])
    ne=sum(1 for x in b['instances'] if x['is_exact'])
    print(f"   evals={len(r)}   BEST={b['combined_score']:.4f}   exact={ne}/161")
    byrow=collections.defaultdict(lambda:[0,0])
    for x in b['instances']:
        m=int(x['mn'].split('x')[0]); byrow[m][1]+=1
        if x['is_exact']: byrow[m][0]+=1
    for m in sorted(byrow):
        ok,tot=byrow[m]
        bar='#'*int(20*ok/tot)
        print(f"   m={m:>2}: {ok:>2}/{tot:<2} {ok/tot:>4.0%} {bar}")
    print()
    print("   recent scores: " + " ".join(f"{x['combined_score']:.2f}" for x in r[-10:]))
else:
    print("   (no evaluations yet)")
PY
  else
    echo "   (no evaluations yet)"
  fi
  echo
  echo "refreshing every ${INTERVAL}s -- Ctrl-C to stop"
  sleep "$INTERVAL"
done
