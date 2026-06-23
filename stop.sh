#!/bin/bash
# Hospital Voice Agent — Stop All Services

LOG_DIR="/tmp/voice_agent_logs"

echo "=== Stopping Hospital Voice Agent ==="
echo ""

# Kill by PID files
for service in graph insurance agent voice dashboard; do
  PID_FILE="$LOG_DIR/$service.pid"
  if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
      kill "$PID" 2>/dev/null
      echo "  Stopped $service (PID $PID)"
    else
      echo "  $service already stopped"
    fi
    rm -f "$PID_FILE"
  fi
done

# Kill any remaining uvicorn / http.server processes
pkill -f "uvicorn graph_rag" 2>/dev/null
pkill -f "uvicorn insurance_server" 2>/dev/null
pkill -f "uvicorn agent:app" 2>/dev/null
pkill -f "uvicorn server:app" 2>/dev/null
pkill -f "http.server 3000" 2>/dev/null

echo ""
echo "=== All services stopped ==="
