#!/usr/bin/env bash
# Builds external SAT tooling used for proof certificates (all gitignored):
#   cadical  - SAT solver with DRAT/LRAT proof output (binary)
#   drat-trim - DRAT checker + DRAT->LRAT conversion
#   lrat-check - LRAT checker (ships with drat-trim repo)
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x cadical/build/cadical ]; then
  [ -d cadical ] || git clone --depth 1 https://github.com/arminbiere/cadical.git
  (cd cadical && ./configure && make -j4)
fi
if [ ! -x drat-trim/drat-trim ]; then
  [ -d drat-trim ] || git clone --depth 1 https://github.com/marijnheule/drat-trim.git
  (cd drat-trim && make)
fi
ls -la cadical/build/cadical drat-trim/drat-trim drat-trim/lrat-check 2>/dev/null || true
