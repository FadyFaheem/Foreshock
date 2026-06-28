# Setup Guide

Step-by-step bring-up for Foreshock. There is **no database and no auth** in v1.

## 1. Python engine + API (host)

```bash
python3 -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
pip install -r infra/api/requirements.txt
```

## 2. Get data and train the model

```bash
python scripts/download_data.py           # CWRU 12 kHz drive-end subset -> data/
python scripts/train.py                   # writes models/model.joblib + models/samples.npz
```

`models/` is committed, so if it is already populated you can skip training and
go straight to running the app.

## 3. Run (host dev)

```bash
# terminal 1 - API on http://localhost:8000
cd infra/api && python app.py

# terminal 2 - frontend on http://localhost:5173
cd frontend && npm install && npm run dev
```

Open <http://localhost:5173>. Vite proxies `/api` to the API.

> macOS uses port 5000 for the AirPlay Receiver, so host dev defaults the API to
> 8000. Override with `PORT=xxxx python app.py` and
> `VITE_API_PROXY=http://localhost:xxxx npm run dev` if needed.

## 3b. AI diagnostics (optional)

The Diagnostics tab (RAG + LLM diagnosis, agent, evals, observability) needs a
local LLM and a Postgres + pgvector database.

```bash
# Ollama (LLM) for HOST dev - via Podman, no native install (the pods ship their own).
podman run -d --name foreshock-ollama -p 11434:11434 \
  -v foreshock-ollama:/root/.ollama docker.io/ollama/ollama
podman exec foreshock-ollama ollama pull llama3.2:1b
podman exec foreshock-ollama ollama pull nomic-embed-text

# Postgres + pgvector for host dev (the Podman pod ships its own postgres-db).
podman run -d --name foreshock-pg -p 5432:5432 \
  -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=foreshock \
  -v "$PWD/infra/database:/docker-entrypoint-initdb.d:ro" \
  docker.io/pgvector/pgvector:pg16

# Seed the knowledge base and (optionally) score the LLM/RAG layer.
python scripts/seed_kb.py
python scripts/run_evals.py

# Run the API with DB + Ollama env (host dev).
cd infra/api && DB_HOST=localhost OLLAMA_HOST=http://localhost:11434 python app.py
```

In the Podman pod this is automatic: `postgres-db` **and** an `ollama` container
are included. The LLM runs in-pod (CPU-only) and pulls `llama3.2:1b` +
`nomic-embed-text` into a persistent volume on first start - no host Ollama
needed. The API gets `DB_HOST=localhost` + `OLLAMA_HOST=http://localhost:11434`.
(First diagnosis after a fresh start waits for the model pull; watch progress with
`podman logs -f foreshock-dev-pod-ollama`.) After `podman play kube`, seed the
pod's database once:

```bash
podman exec foreshock-dev-pod-foreshock-api python /app/scripts/seed_kb.py
```

## 3c. Predictive-maintenance lifecycle (v2 + v3)

```bash
# v2 - train the health autoencoder (writes models/health_ae.joblib + health.npz)
#   Real run-to-failure data (recommended): downloads + unpacks NASA IMS 2nd_test
#   into data/ims/ (~1 GB), then trains on genuinely progressive degradation.
python scripts/download_ims.py
python scripts/train_health.py     # auto-detects data/ims/ (CWRU-derived fallback if absent)

# v3 - needs a Kafka broker. The Podman pod includes one; for host dev:
podman run -d --name foreshock-kafka -p 9092:9092 \
  -e KAFKA_NODE_ID=1 -e KAFKA_PROCESS_ROLES=broker,controller \
  -e KAFKA_LISTENERS=PLAINTEXT://localhost:9092,CONTROLLER://localhost:9093 \
  -e KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://localhost:9092 \
  -e KAFKA_CONTROLLER_LISTENER_NAMES=CONTROLLER \
  -e KAFKA_LISTENER_SECURITY_PROTOCOL_MAP=CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT \
  -e KAFKA_CONTROLLER_QUORUM_VOTERS=1@localhost:9093 \
  -e KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR=1 \
  docker.io/apache/kafka:3.9.0
# then run the API (with KAFKA_BOOTSTRAP=localhost:9092) and stream:
python scripts/stream_producer.py 30 0.5
```

Open the **Health** and **Live** tabs in the UI. In the Podman pod both are
wired automatically (a `kafka` container + `KAFKA_BOOTSTRAP=localhost:9092`).

## 4. Run on Podman (optional)

Prerequisites:

```bash
# Install the podman CLI (Podman Desktop alone may not add it to PATH)
brew install podman          # macOS
podman machine init          # one-time
podman machine start
```

Then bring up the dev pod (train the model first so `models/` exists):

```bash
podman play kube infra/podman/foreshock-dev.yaml
# frontend: http://localhost:3000   API: http://localhost:8000 (container port 5000)
```

Or use the interactive runner:

```bash
brew install fzf                          # if not installed
alias cmds='bash tools/cli/cmds.sh'
cmds pods                                  # pick "Start foreshock-dev pod"
```

The dev pod mounts the repo read-only into the API container (so it can import
`src/` and load `models/`) and mounts `frontend/` read-write for the Vite
container. Note: the frontend container runs `npm install` into the bind-mounted
`frontend/`; if you switch back to host dev, re-run `npm install` on the host.

## 5. Public URL via Cloudflare Tunnel (optional)

```bash
cloudflared tunnel login
cloudflared tunnel create foreshock-dev
# copy the credentials JSON into infra/cloudflared/creds/<TUNNEL_UUID>.json
# set <TUNNEL_UUID> in infra/cloudflared/config.yml
cloudflared tunnel route dns foreshock-dev dev-foreshock.faheemlabs.com
```

The dev pod includes a `cloudflared` container that reads `config.yml` and
routes `/api/*` to the API (5000) and everything else to the frontend (3000).
For production use `foreshock-prod.yaml` + `config.prod.yml` (builds the SPA and
serves it on port 80).

## Troubleshooting

- **`podman: command not found`** — install the CLI (`brew install podman`); the
  Desktop app does not always add it to PATH.
- **Pod won't start** — ensure `podman machine start` has run and host paths in
  `infra/podman/foreshock-dev.yaml` match this repo's location.
- **API returns 503** — the model isn't trained; run `python scripts/train.py`.
- **Frontend can't reach the API (host dev)** — confirm the API is on 8000 and
  the Vite proxy target matches.
