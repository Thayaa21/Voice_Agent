#!/bin/bash
# Hospital Voice Agent — Start All Services

ROOT="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="/tmp/voice_agent_logs"
mkdir -p "$LOG_DIR"

echo "=== Starting Hospital Voice Agent ==="
echo ""

# Kill any leftover processes
pkill -f "uvicorn" 2>/dev/null
pkill -f "http.server 3000" 2>/dev/null
sleep 1

# 1. Graph RAG Backend (port 8000)
echo "[1/5] Graph RAG Backend     → port 8000"
cd "$ROOT/graph_rag_backend"
uvicorn graph_rag.api.app:app --port 8000 --host 0.0.0.0 > "$LOG_DIR/graph.log" 2>&1 &
echo $! > "$LOG_DIR/graph.pid"
sleep 12  # wait for 114MB snapshot to load

# 2. Insurance Agent (port 8002)
echo "[2/5] Dummy Insurance Agent → port 8002"
cd "$ROOT/insurance_agent"
uvicorn insurance_server:app --port 8002 --host 0.0.0.0 > "$LOG_DIR/insurance.log" 2>&1 &
echo $! > "$LOG_DIR/insurance.pid"
sleep 1

# 3. Hospital Agent (port 9001)
echo "[3/5] Hospital Agent        → port 9001"
cd "$ROOT/chetan"
uvicorn agent:app --port 9001 --host 0.0.0.0 > "$LOG_DIR/agent.log" 2>&1 &
echo $! > "$LOG_DIR/agent.pid"
sleep 1

# 4. Voice Infrastructure (port 8001)
echo "[4/5] Voice Infrastructure  → port 8001"
cd "$ROOT/rishi"
uvicorn server:app --port 8001 --host 0.0.0.0 > "$LOG_DIR/voice.log" 2>&1 &
echo $! > "$LOG_DIR/voice.pid"
sleep 1

# 5. Dashboard (port 3000)
echo "[5/5] Dashboard             → port 3000"
cd "$ROOT/dashboard"
python3 -m http.server 3000 > "$LOG_DIR/dashboard.log" 2>&1 &
echo $! > "$LOG_DIR/dashboard.pid"
sleep 1

# Health checks
echo ""
echo "=== Health Checks ==="
check() {
  local name=$1 url=$2
  result=$(curl -s "$url" 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print('OK')" 2>/dev/null || echo "WAITING")
  echo "  $name: $result"
}
check "Graph RAG (8000)" "http://localhost:8000/graph/stats"
check "Insurance  (8002)" "http://localhost:8002/health"
check "Agent      (9001)" "http://localhost:9001/health"
check "Voice      (8001)" "http://localhost:8001/health"
check "Dashboard  (3000)" "http://localhost:3000"

echo ""
echo "=== All services started ==="
echo ""
echo "  Dashboard:    http://localhost:3000"
echo "  Hospital line: +1 (531) 324-5471"
echo ""
echo "Logs:  $LOG_DIR/"
echo "Stop:  ./stop.sh"
echo ""
echo "Next: run 'ngrok http 8001' in a new terminal"
echo "      then paste the URL in Twilio console → Voice webhook"
