#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/lanjinxin/workspace/wearable_ai_challenge"
PYTHON="/home/lanjinxin/miniconda3/bin/python"
RUNTIME_DIR="$PROJECT_ROOT/logs/human_review"
PORT_FILE="$RUNTIME_DIR/server.port"
MODE_FILE="$RUNTIME_DIR/server.mode"
PASSWORD_FILE="$RUNTIME_DIR/access.password"
LOG_FILE="$RUNTIME_DIR/server.log"
TMUX_SESSION="ego-human-review"

mkdir -p "$RUNTIME_DIR"

is_running() {
  tmux has-session -t "$TMUX_SESSION" 2>/dev/null
}

saved_port() {
  tr -d '[:space:]' < "$PORT_FILE" 2>/dev/null || true
}

status() {
  local port mode
  port="$(saved_port)"
  mode="$(tr -d '[:space:]' < "$MODE_FILE" 2>/dev/null || true)"
  if is_running; then
    echo "running: tmux=$TMUX_SESSION port=$port mode=${mode:-local}"
    if [[ "$mode" == "public" ]]; then
      echo "browser URL: http://<the same server host used for SSH>:$port"
      echo "browser username: review"
      echo "browser password: $(cat "$PASSWORD_FILE")"
    else
      echo "server URL: http://127.0.0.1:$port"
    fi
    echo "log: $LOG_FILE"
    return 0
  fi
  echo "stopped"
  return 1
}

start() {
  local port="${1:-8770}"
  local mode="${2:-local}"
  local host="127.0.0.1"
  local auth_option=""
  local launch
  if ! [[ "$port" =~ ^[0-9]+$ ]] || (( port < 1024 || port > 65535 )); then
    echo "Invalid port: $port" >&2
    exit 2
  fi
  if is_running; then
    status
    exit 0
  fi

  if [[ "$mode" == "public" ]]; then
    host="0.0.0.0"
    if [[ ! -s "$PASSWORD_FILE" ]]; then
      umask 077
      openssl rand -hex 12 > "$PASSWORD_FILE"
    fi
    chmod 600 "$PASSWORD_FILE"
    auth_option="--auth-password-file '$PASSWORD_FILE'"
  elif [[ "$mode" != "local" ]]; then
    echo "Invalid mode: $mode" >&2
    exit 2
  fi

  echo "$port" > "$PORT_FILE"
  echo "$mode" > "$MODE_FILE"
  printf '\n[%s] starting human review server on %s:%s (%s)\n' "$(date -Is)" "$host" "$port" "$mode" >> "$LOG_FILE"
  launch="cd '$PROJECT_ROOT' && exec env PYTHONPATH='$PROJECT_ROOT/src' PYTHONUNBUFFERED=1 '$PYTHON' -m proactive_review.server --host '$host' --port '$port' --strict-port $auth_option >>'$LOG_FILE' 2>&1"
  tmux new-session -d -s "$TMUX_SESSION" "$launch"

  sleep 1
  if ! is_running; then
    echo "Server failed to start. Recent log:" >&2
    tail -n 30 "$LOG_FILE" >&2
    rm -f "$PORT_FILE" "$MODE_FILE"
    exit 1
  fi
  echo "started: tmux=$TMUX_SESSION"
  if [[ "$mode" == "public" ]]; then
    echo "browser URL: http://<the same server host used for SSH>:$port"
    echo "browser username: review"
    echo "browser password: $(cat "$PASSWORD_FILE")"
  else
    echo "server URL: http://127.0.0.1:$port"
  fi
  echo "log: $LOG_FILE"
  echo "The tmux session is detached; closing SSH will not stop it."
}

stop() {
  if ! is_running; then
    echo "Server is not running."
    rm -f "$PORT_FILE" "$MODE_FILE"
    return 0
  fi
  tmux kill-session -t "$TMUX_SESSION"
  rm -f "$PORT_FILE" "$MODE_FILE"
  echo "stopped: tmux=$TMUX_SESSION"
}

case "${1:-status}" in
  start)
    start "${2:-8770}" local
    ;;
  start-public)
    start "${2:-8770}" public
    ;;
  stop)
    stop
    ;;
  restart)
    stop
    start "${2:-8770}" local
    ;;
  restart-public)
    stop
    start "${2:-8770}" public
    ;;
  status)
    status
    ;;
  log)
    tail -n "${2:-50}" "$LOG_FILE"
    ;;
  attach)
    tmux attach-session -t "$TMUX_SESSION"
    ;;
  *)
    echo "Usage: $0 {start [port]|start-public [port]|stop|restart [port]|restart-public [port]|status|log [lines]|attach}" >&2
    exit 2
    ;;
esac
