#!/usr/bin/env bash
set -euo pipefail

session="${1:-data-prep}"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! "${session}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "error: tmux session name may only contain letters, numbers, dots, underscores, and hyphens" >&2
  exit 2
fi

if tmux has-session -t "${session}" 2>/dev/null; then
  echo "error: tmux session already exists: ${session}" >&2
  echo "Attach with: tmux attach -t ${session}" >&2
  exit 1
fi

repo_dir_quoted="$(printf "%q" "${repo_dir}")"
command="cd ${repo_dir_quoted} && bash scripts/prepare_50b_data_and_sync.sh"

if [[ -n "${TMUX:-}" ]]; then
  current_session="$(tmux display-message -p '#S')"
  if tmux list-windows -t "${current_session}" -F '#W' | grep -Fxq "${session}"; then
    echo "error: tmux window already exists in ${current_session}: ${session}" >&2
    echo "Switch with: tmux select-window -t ${current_session}:${session}" >&2
    exit 1
  fi

  tmux new-window -n "${session}" "${command}"
  echo "started data prep in tmux window: ${current_session}:${session}"
  echo "Switch with: tmux select-window -t ${current_session}:${session}"
  exit 0
fi

exec tmux new-session -s "${session}" "${command}"
