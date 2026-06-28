#!/usr/bin/env bash
#
# All-in-one Foreshock bring-up via Podman.
#
#   tools/cli/bootstrap.sh [dev|prod]      (default: dev)
#
# Steps (each is idempotent / safe to re-run):
#   1. Pre-pull container images with visible progress (first run is ~5 GB).
#   2. Start the Podman pod (--replace recreates containers, keeps PVC volumes).
#   3. Wait for the first-boot trainer init container (it trains the models inside
#      the pod on the first boot; a fast no-op once models exist).
#   4. Wait for Postgres, then for Ollama to finish pulling its models
#      (first run downloads ~1.3 GB), then for the API to answer /health.
#   5. Seed the RAG knowledge base inside the API container.

set -uo pipefail

ENV="${1:-dev}"
case "$ENV" in
  dev)
    POD="foreshock-dev-pod"
    MANIFEST="infra/podman/foreshock-dev.yaml"
    API_URL="http://localhost:8000"
    FRONT_URL="http://localhost:3000"
    DB_PORT=5432
    ;;
  prod)
    POD="foreshock-prod-pod"
    MANIFEST="infra/podman/foreshock-prod.yaml"
    API_URL="http://localhost:15000"
    FRONT_URL="http://localhost:18080"
    DB_PORT=15432
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

PG="${POD}-postgres-db"
OLL="${POD}-ollama"
API="${POD}-foreshock-api"
TRAINER="${POD}-trainer"

# --- 1) images ----------------------------------------------------------------
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

# --- 2) pod -------------------------------------------------------------------
log "Starting the $ENV pod ($POD)"
podman play kube --replace "$MANIFEST" || die "podman play kube failed"

# --- 3) first-boot training ---------------------------------------------------
# An init container (see the manifest) trains the models the first time the pod
# boots, before the app containers start; a fast no-op once models exist. Wait for
# it so the API never comes up model-less on a fresh host.
if podman container exists "$TRAINER" 2>/dev/null; then
  log "Waiting for first-boot model training (one-time: downloads data + trains)"
  while podman ps --filter "name=$TRAINER" --filter "status=running" --format '{{.Names}}' 2>/dev/null | grep -q "$TRAINER"; do
    printf '.'; sleep 3
  done
  ok "Training step finished."
fi

# --- 4a) Postgres -------------------------------------------------------------
# Probe the always-present 'postgres' db: pg_isready reports *server* readiness,
# not whether our app db exists (that is ensured below, before seeding).
log "Waiting for Postgres"
for _ in $(seq 1 60); do
  if podman exec "$PG" pg_isready -U postgres -d postgres -p "$DB_PORT" >/dev/null 2>&1; then
    ok "Postgres ready."; break
  fi
  printf '.'; sleep 2
done
if ! podman exec "$PG" pg_isready -U postgres -d postgres -p "$DB_PORT" >/dev/null 2>&1; then
  warn "Postgres not ready (continuing)"
  # hostNetwork binds Postgres directly on the *host*; a host PostgreSQL service
  # or a stray container on that port makes our postgres crash-loop with "Address
  # already in use", while clients silently reach the other server (e.g. a db
  # without 'foreshock'). Surface that here instead of a confusing silent failure.
  if podman logs --tail 25 "$PG" 2>&1 | grep -qi "address already in use"; then
    warn "Port $DB_PORT is already in use on the host (hostNetwork mode)."
    warn "  find it:  sudo ss -lptn 'sport = :$DB_PORT'"
    warn "  free it:  stop whatever owns it (e.g. sudo systemctl stop postgresql), then re-run"
  fi
fi

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
# The Postgres image only creates POSTGRES_DB on a *fresh* data volume; a reused
# PVC (or an interrupted first init) can leave the server running without the
# 'foreshock' database, so seeding fails with: database "foreshock" does not
# exist. Create it if missing so re-runs self-heal (the API's migrations then add
# pgvector + the schema).
if podman exec "$PG" pg_isready -U postgres -d postgres -p "$DB_PORT" >/dev/null 2>&1 \
   && ! podman exec "$PG" psql -U postgres -p "$DB_PORT" -tAc \
        "SELECT 1 FROM pg_database WHERE datname='foreshock'" 2>/dev/null | grep -q 1; then
  warn "Database 'foreshock' missing - creating it"
  podman exec "$PG" psql -U postgres -p "$DB_PORT" -c "CREATE DATABASE foreshock" >/dev/null 2>&1 \
    && ok "Database 'foreshock' created." || warn "Could not create 'foreshock' (seeding may fail)"
fi

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
