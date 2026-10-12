#!/usr/bin/env bash
[ -n "${BASH_VERSION:-}" ] || exec bash "$0" "$@"
set -euo pipefail
export LC_ALL=C

cutoff="2026-08-18"
splitter="$HOME/bin/split_file.py"

git rev-parse --is-inside-work-tree >/dev/null
command -v python3 >/dev/null
if [[ ! -f "$splitter" ]]; then
    printf 'Splitter not found: %s\n' "$splitter" >&2
    exit 1
fi

file_list=$(mktemp)
trap 'rm -f -- "$file_list"' EXIT
git ls-files -z -- . | sort -z > "$file_list"

count=0
while IFS= read -r -d '' file; do
    [[ -f "$file" ]] || continue
    last_time=$(git log -1 --format='%ci' -- "$file")

    if [[ "${last_time:0:10}" > "$cutoff" ]]; then
        printf 'Splitting: %s\t%s\n' "$file" "$last_time"
        python3 "$splitter" "$file"
        count=$((count + 1))
    fi
done < "$file_list"

printf 'Done: %d files.\n' "$count"
