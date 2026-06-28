# Podman Configuration

Pod definitions for the Foreshock development and production environments. Each
pod runs the full stack: Postgres + pgvector (RAG store), a Kafka broker (v3 live
feed), an in-pod Ollama LLM (CPU-only), the Flask API, the React frontend, and a
Cloudflare Tunnel.

## Quick Start

```bash
# All-in-one: train models (if needed) -> start pod -> wait for Postgres/Ollama/API
# -> seed the knowledge base. Idempotent; pass `prod` for the prod pod.
bash tools/cli/bootstrap.sh dev

# ...or step by step:
# Start the dev pod
podman play kube infra/podman/foreshock-dev.yaml

# Stop
podman pod stop foreshock-dev-pod

# Delete (containers + pod)
podman pod rm -f foreshock-dev-pod

# Rebuild
podman pod rm -f foreshock-dev-pod && podman play kube infra/podman/foreshock-dev.yaml
```

Dev: frontend on <http://localhost:3000>, API on <http://localhost:8000>
(mapped to the container's port 5000; macOS reserves 5000 for AirPlay).

## Containers in each pod

| Container | Image | Dev port | Prod port | Purpose |
|-----------|-------|----------|-----------|---------|
| `postgres-db` | `pgvector/pgvector:pg16` | — (intra-pod) | — (intra-pod) | RAG store (Postgres + pgvector) |
| `kafka` | `apache/kafka:3.9.0` | 9092 | — (intra-pod) | v3 live sensor feed (KRaft) |
| `ollama` | `ollama/ollama` | — (intra-pod) | 11434 (host) | CPU-only local LLM + embeddings |
| `foreshock-api` | `python:3.11-slim` | 8000 host / 5000 container | 5000 (gunicorn) | REST API over the `src/` engine |
| `foreshock-frontend` | `node:20-alpine` | 3000 (vite dev) | 80 (built + `serve`) | React + MUI web UI |
| `cloudflared` | `cloudflare/cloudflared` | — | — | Public tunnel |

The dev pod uses normal port mappings; the prod pod uses `hostNetwork: true`
(common for single-host deployments behind a tunnel).

## Local LLM (Ollama, CPU-only)

The `ollama` container runs the LLM entirely in-pod — no host install and no GPU
required. On first start it pulls `llama3.2:1b` (generation) and
`nomic-embed-text` (embeddings) into a persistent volume (`*-ollama` PVC); later
starts reuse them. It is tuned to stay light on a small CPU box (4c/32 GB is
plenty): one request at a time (`OLLAMA_NUM_PARALLEL=1`) and models kept warm
(`OLLAMA_KEEP_ALIVE=24h`), for roughly **3 GB RAM** resident.

- **Footprint:** `llama3.2:1b` is ~0.8 GB on disk (Q4) and runs at a usable speed
  on AVX2 CPUs. For an even lighter/snappier box, set the `ollama` container's
  `LLM_MODEL` to `qwen2.5:0.5b`. Keep `EMBED_MODEL=nomic-embed-text` — the DB
  schema is `vector(768)`, which matches that model.
- **First call latency:** the first diagnosis after a fresh start waits for the
  model pull (~1 GB download). Watch it with `podman logs -f <pod>-ollama`.
- **Prod note:** with `hostNetwork: true` the container binds `11434` on the host,
  so stop any host `ollama serve` first.

## How the API container finds the engine

The repo root is mounted **read-only** at `/app`, so the container sees
`/app/src` (engine), `/app/infra/api` (API), and `/app/models` (trained model +
samples). The API runs from `/app/infra/api` with `PYTHONPATH=/app`, so
`import src` resolves. Train the model on the host first (`python scripts/train.py`)
so `models/model.joblib` and `models/samples.npz` exist before starting the pod.

## Required configuration before first run

1. **Host paths.** `foreshock-dev.yaml` is pre-filled with this repo's path on
   the dev machine. For `foreshock-prod.yaml`, update the `/home/ubuntu/Foreshock`
   placeholders to the project location on your prod host.
2. **Cloudflare Tunnel (optional).** See `infra/cloudflared/`. Create the tunnel,
   drop credentials into `creds/`, and set the `<TUNNEL_UUID>` placeholder in
   `config.yml` / `config.prod.yml`. Skip this if you only need localhost.

## Notes

- The frontend container runs `npm install` into the bind-mounted `frontend/`,
  which replaces host `node_modules` with Linux binaries. If you switch back to
  host dev afterwards, re-run `npm install` on the host.
- First API start is slow (it pip-installs numpy/scipy/scikit-learn). Subsequent
  restarts of the same container reuse the layer until the pod is recreated.
