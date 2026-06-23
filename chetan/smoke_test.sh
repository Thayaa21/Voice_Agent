#!/usr/bin/env bash
# Smoke test for Chetan's agent.
# Assumes the agent is running on :9001 and a backend on :8000.
# Usage:  ./smoke_test.sh   (or: bash smoke_test.sh)
set -u
AGENT="${AGENT_URL:-http://localhost:9001}"

say() { printf '\n[caller] %s\n' "$2"
  curl -s -X POST "$AGENT/agent" -H "Content-Type: application/json" \
    -d "{\"text\": \"$2\", \"call_id\": \"$1\"}" \
    | python3 -c "import sys,json;d=json.load(sys.stdin);print('[agent ]',d['response'],'   (end_call='+str(d['end_call'])+')')"
}

echo "=================================================="
echo " README tests 1-5  (use a DISTINCTIVE name in real data,"
echo " e.g. 'Jordan Young, born September 6 1914')"
echo "=================================================="
say demo "Jordan Young, born September 6 1914"
say demo "why was my claim denied"
say demo "how much do I owe"
say demo "what conditions do I have"
say demo "do I need to stop my medication before my procedure"
say demo "bye"

echo
echo "=================================================="
echo " Disambiguation (common surname -> ask DOB -> resolve)"
echo "=================================================="
say dis "Aiden Garcia"
say dis "my date of birth is 1978-09-03"
say dis "why do I owe so much"
say dis "thanks, that's all"

echo
echo "=================================================="
echo " Two failed identity attempts -> end_call"
echo "=================================================="
say fail "asdf qwerty"
say fail "zxcv hjkl"
