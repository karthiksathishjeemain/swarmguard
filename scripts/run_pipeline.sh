#!/usr/bin/env bash
# Reproduce the VM-only action-grounded investigator at the frozen revision.
set -Eeuo pipefail
umask 077

root=/home/azureuser/ai-swarm
scratch=/mnt/ai-swarm-scratch
project="$root/project"
python="$root/.venv/bin/python"
config="$project/config/experiment.toml"

if [[ "$(uname -s)" != Linux ]] || ! mountpoint -q "$scratch"; then
  echo 'Azure Linux VM scratch mount required.' >&2
  exit 2
fi
if [[ ! -x "$python" ]]; then
  echo 'Run scripts/bootstrap_vm.sh on the VM first.' >&2
  exit 2
fi
if [[ ! -f "$config" ]]; then
  echo 'Missing config/experiment.toml.' >&2
  exit 2
fi
revision=$("$python" -c 'import sys,tomllib; print(tomllib.load(open(sys.argv[1],"rb"))["dataset"]["revision"])' "$config")
window_actions=$("$python" -c 'import sys,tomllib; print(tomllib.load(open(sys.argv[1],"rb"))["retrieval"]["preceding_actions"])' "$config")
controls=$("$python" -c 'import sys,tomllib; print(tomllib.load(open(sys.argv[1],"rb"))["retrieval"]["random_controls"])' "$config")
mkdir -p "$root/logs" "$root/reports" "$scratch/derived"
exec 9>"$root/pipeline.lock"
flock -n 9 || { echo 'Another pipeline run is active.' >&2; exit 2; }

run() {
  printf 'Running %s at %s\n' "$1" "$(date -u +%FT%TZ)"
  shift
  local script=$1
  shift
  "$python" "$project/scripts/$script" "$@"
}

if [[ ! -f "$root/source-docs/manifest.json" ]]; then
  run schema fetch_dataset_docs.py --revision "$revision"
fi

missing=()
for table in chat_messages computer_use_sessions computer_use_turns; do
  if [[ ! -f "$scratch/derived/$table.parquet" ]]; then
    missing+=("$table.jsonl.gz")
  fi
done
if ((${#missing[@]})); then
  run source fetch_text_files.py --revision "$revision" "${missing[@]}"
  for table in chat_messages computer_use_sessions computer_use_turns; do
    if [[ ! -f "$scratch/derived/$table.parquet" ]]; then
      run "project-$table" build_projected_tables.py --table "$table"
    fi
  done
fi

[[ -f "$scratch/derived/sent_message_links.parquet" ]] || run joins link_sent_messages.py
[[ -f "$scratch/derived/evidence_windows.parquet" ]] || run windows build_evidence_windows.py --window-actions "$window_actions" --controls "$controls"
[[ -f "$scratch/derived/ranked_review_leads.parquet" ]] || run ranking rank_review_leads.py
[[ -f "$scratch/derived/review_queue.parquet" ]] || run queue prepare_review_queue.py
[[ -f "$scratch/derived/review_queue_laya.parquet" ]] || run laya annotate_review_queue.py
run integrity investigate.py check
run metrics summarize_reviews.py
printf 'Pipeline complete at %s\n' "$(date -u +%FT%TZ)"
