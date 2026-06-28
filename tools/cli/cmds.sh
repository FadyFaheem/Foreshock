#!/usr/bin/env bash

set -u
set -o pipefail

folder_name="tools/cli/commands"
scope=""

# Lowercase helper. macOS ships bash 3.2, which lacks bash 4's ${var,,}, so use tr.
to_lower() { printf '%s' "$1" | tr '[:upper:]' '[:lower:]'; }

usage() {
  cat <<'EOF'
Usage: cmds.sh [scope] [--scope VALUE]

Options:
  scope          Positional argument to limit results (same as --scope)
  -s, --scope    Limit results to files whose path or name includes the scope
  -h, --help    Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -s|--scope)
      [[ $# -lt 2 ]] && { echo "Missing value for $1" >&2; exit 1; }
      scope="$2"
      shift 2
      ;;
    --scope=*)
      scope="${1#*=}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      if [[ -z "$scope" ]]; then
        scope="$1"
        shift
      else
        echo "Unexpected argument: $1" >&2
        usage >&2
        exit 1
      fi
      ;;
  esac
done

if ! command -v fzf >/dev/null 2>&1; then
  echo "fzf not found in PATH." >&2
  exit 1
fi

if repo_root=$(git rev-parse --show-toplevel 2>/dev/null); then
  :
else
  repo_root=$(pwd)
fi

folder="${repo_root%/}/$folder_name"

if [[ ! -d "$folder" ]]; then
  echo "Commands folder not found: $folder_name" >&2
  exit 1
fi

declare -a files=()

if [[ -n "$scope" && -d "$folder/$scope" ]]; then
  while IFS= read -r -d '' path; do
    files+=("$path")
  done < <(find "$folder/$scope" -type f -print0 2>/dev/null)
elif [[ -n "$scope" ]]; then
  lower_scope=$(to_lower "$scope")
  while IFS= read -r -d '' path; do
    rel=${path#"$folder"/}
    rel=${rel#/}
    lower_rel=$(to_lower "$rel")
    base=$(basename "$path")
    lower_base=$(to_lower "$base")
    if [[ "$lower_rel" == *"$lower_scope"* || "$lower_base" == *"$lower_scope"* ]]; then
      files+=("$path")
    fi
  done < <(find "$folder" -type f -print0 2>/dev/null)
else
  while IFS= read -r -d '' path; do
    files+=("$path")
  done < <(find "$folder" -type f -print0 2>/dev/null)
fi

if [[ ${#files[@]} -eq 0 ]]; then
  printf "No files matched under '%s' with scope '%s'.\n" "$folder_name" "$scope" >&2
  exit 1
fi

declare -a groups=()
declare -a labels=()
declare -a cmds=()
declare -a display=()

index=0

for path in "${files[@]}"; do
  rel=${path#"$folder"/}
  rel=${rel#/}
  while IFS=$'\t' read -r group label cmd; do
    groups[index]="$group"
    labels[index]="$label"
    cmds[index]="$cmd"
    printf -v line '%d\t[%s] %s\t[%s]' "$index" "$group" "$label" "$cmd"
    display[index]="$line"
    ((index++))
  done < <(
    awk -v rel="$rel" '
      function ltrim(s) { sub(/^[ \t\r\n]+/, "", s); return s }
      function rtrim(s) { sub(/[ \t\r\n]+$/, "", s); return s }
      function trim(s) { return rtrim(ltrim(s)) }
      BEGIN { pending = "" }
      {
        raw = $0
        trimmed = trim(raw)
        if (raw ~ /^[[:space:]]*#/) {
          sub(/^[[:space:]]*#[[:space:]]*/, "", trimmed)
          pending = trimmed
          next
        }
        if (trimmed == "") { next }
        if (pending != "") {
          printf("%s\t%s\t%s\n", rel, pending, trimmed)
          pending = ""
        }
      }
    ' "$path"
  )
done

if [[ ${#cmds[@]} -eq 0 ]]; then
  echo "No commands found." >&2
  exit 1
fi

selection=$(printf '%s\n' "${display[@]}" | fzf \
  --height 80% \
  --reverse \
  --prompt 'run> ' \
  --with-nth=2 \
  --delimiter=$'\t' \
  --preview='printf "%s\n%s\n" "Command:" {3}' \
  --preview-window=right:40%:wrap)
[[ -z "$selection" ]] && exit 0

idx=${selection%%$'\t'*}
cmd=${cmds[idx]}

printf '>> %s\n' "$cmd"

# shellcheck disable=SC2086
eval "$cmd"
