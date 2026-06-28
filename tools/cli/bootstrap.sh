#!/usr/bin/env bash
#
# All-in-one Foreshock bring-up via Podman.
#
#   tools/cli/bootstrap.sh [dev|prod]      (default: dev)
#
# Steps (each is idempotent / safe to re-run):
#   1. Train the models on the host if models/ is empty (the pod mounts them RO).
#   2. Pre-pull container images with visible progress (first run is ~5 GB).
#   3. Start the Podman pod (--replace recreates containers, keeps PVC volumes).
#   4. Wait for Postgres, then for Ollama to finish pulling its models
#      (first run downloads ~1.3 GB), then for the API to answer /health.
#   5. Seed the RAG knowledge base inside the API container.
#
# Env: SKIP_TRAIN=1 to skip step 1.

set -uo pipefail

ENV="${1:-dev}"
case "$ENV" in
  dev)
    POD="foreshock-dev-pod"
    MANIFEST="infra/podman/foreshock-dev.yaml"
    API_URL="http://localhost:8000"
    FRONT_URL="http://localhost:3000"
    ;;
  prod)
    POD="foreshock-prod-pod"
    MANIFEST="infra/podman/foreshock-prod.yaml"
    API_URL="http://localhost:5000"
    FRONT_URL="http://localhost"
    ;;
  *)
    echo "usage: $(basename "$0") [dev|prod]" >&2
    exit 1
    ;;
esac

C_HEAD=$'\033[1;36m'; C_OK=$'\033[1;32m'; C_WARN=$'\033[1;33m'; C_ERR=$'\033[1;31m'; C_OFF=$'\033[0m'
log()  { printf '\n%s== %s%s\n' "$C_HEAD" "$*" "$C_OFF"; }
ok()   { printf '%s   ok: %s%s\n' "$C_OK" "$*" "$C_OFF"; }
warn() { printf '%s   warn: %s%s\n' "$C_WARN" "$*" "$C_OFF"; }
die()  { printf '%s   error: %s%s\n' "$C_ERR" "$*" "$C_OFF" >&2; exit 1; }

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT" || die "cannot cd to repo root"

# --- preflight ----------------------------------------------------------------
command -v podman >/dev/null 2>&1 || die "podman not found in PATH"
podman info >/dev/null 2>&1 || die "podman not ready (on macOS: 'podman machine start')"

PY="python3"
[ -x ".venv/bin/python" ] && PY=".venv/bin/python"

PG="${POD}-postgres-db"
OLL="${POD}-ollama"
API="${POD}-foreshock-api"

# --- 1) models ----------------------------------------------------------------
if [ "${SKIP_TRAIN:-0}" = "1" ]; then
  warn "SKIP_TRAIN=1 - not training models"
elif [ ! -f models/model.joblib ] || [ ! -f models/samples.npz ]; then
  log "Training the classifier (models/ missing) with $PY"
  "$PY" scripts/download_data.py || die "download_data.py failed"
  "$PY" scripts/train.py || die "train.py failed"
  ok "Classifier trained."
else
  ok "Classifier model present."
fi
if [ "${SKIP_TRAIN:-0}" != "1" ] && [ ! -f models/health.npz ]; then
  log "Training the v2 health model with $PY"
  "$PY" scripts/train_health.py || warn "train_health.py failed (v2 Health tab will 503)"
fi

# --- 2) images ----------------------------------------------------------------
# Pre-pull with visible progress so the first run never looks "frozen": these
# images total ~5 GB (the Ollama image alone is ~4 GB) and `podman play kube`
# pulls them silently.
log "Pulling images (first run is large - ~5 GB incl. the Ollama image)"
grep -E '^[[:space:]]*image:[[:space:]]' "$MANIFEST" | awk '{print $2}' | sort -u | while IFS= read -r img; do
  [ -n "$img" ] || continue
  printf '   - %s\n' "$img"
  podman pull "$img" >/dev/null || warn "pull failed for $img (play kube will retry)"
done
ok "Images ready."

# --- 3) pod -------------------------------------------------------------------
log "Starting the $ENV pod ($POD)"
podman play kube --replace "$MANIFEST" || die "podman play kube failed"

# --- 4a) Postgres -------------------------------------------------------------
log "Waiting for Postgres"
for _ in $(seq 1 60); do
  if podman exec "$PG" pg_isready -U postgres -d foreshock >/dev/null 2>&1; then
    ok "Postgres ready."; break
  fi
  printf '.'; sleep 2
done
podman exec "$PG" pg_isready -U postgres -d foreshock >/dev/null 2>&1 || warn "Postgres not ready (continuing)"

# --- 4b) Ollama models --------------------------------------------------------
LLM="$(podman exec "$OLL" printenv LLM_MODEL 2>/dev/null || echo llama3.2:1b)"
EMB="$(podman exec "$OLL" printenv EMBED_MODEL 2>/dev/null || echo nomic-embed-text)"
log "Waiting for Ollama models ($LLM + $EMB; first run downloads ~1 GB)"
for _ in $(seq 1 300); do
  LIST="$(podman exec "$OLL" ollama list 2>/dev/null || true)"
  if printf '%s' "$LIST" | grep -q "$LLM" && printf '%s' "$LIST" | grep -q "$EMB"; then
    ok "Models ready: $LLM, $EMB"; break
  fi
  printf '.'; sleep 2
done

# --- 4c) API ------------------------------------------------------------------
log "Waiting for the API (first start pip-installs deps)"
for _ in $(seq 1 150); do
  if curl -fsS "$API_URL/health" >/dev/null 2>&1; then ok "API healthy at $API_URL"; break; fi
  printf '.'; sleep 2
done
curl -fsS "$API_URL/health" >/dev/null 2>&1 || warn "API not answering yet at $API_URL/health"

# --- 5) seed the knowledge base ----------------------------------------------
log "Seeding the RAG knowledge base"
if podman exec "$API" python /app/scripts/seed_kb.py; then
  ok "Knowledge base seeded."
else
  warn "Seeding failed (retry later: podman exec $API python /app/scripts/seed_kb.py)"
fi

# --- done ---------------------------------------------------------------------
log "Up"
podman pod ps --filter "name=$POD" 2>/dev/null || true
printf '\n%sFrontend:%s %s\n%sAPI:%s      %s\n' "$C_OK" "$C_OFF" "$FRONT_URL" "$C_OK" "$C_OFF" "$API_URL"
printf 'Diagnostics tab needs the KB + Ollama; Live tab needs the v3 stream producer.\n'
