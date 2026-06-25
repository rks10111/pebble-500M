#!/usr/bin/env bash
set -euo pipefail

host="${STATEMENT_LLM_HOST:-statement-llm}"
session="${1:-train}"

if [[ ! "${session}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "error: tmux session name may only contain letters, numbers, dots, underscores, and hyphens" >&2
  exit 2
fi

ssh -t "${host}" "
if tmux has-session -t '${session}' 2>/dev/null; then
  exec tmux attach-session -t '${session}'
else
  exec tmux new-session -s '${session}'
fi
"
