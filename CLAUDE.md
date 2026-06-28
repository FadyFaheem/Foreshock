# CLAUDE.md

Architecture, conventions, and workflows for contributors and AI assistants
working in Foreshock. Read this before making changes.

## Project overview

Foreshock is a predictive-maintenance demo: it classifies rolling-bearing faults
(normal / inner race / outer race / ball) from vibration signals and serves the
results through a small web app.

- **Engine (`src/`)** — pure Python signal processing + scikit-learn. No web code.
- **API (`infra/api/`)** — Flask, application-factory + blueprint. Imports `src/`.
- **Frontend (`frontend/`)** — React + Vite + TypeScript + MUI. Calls the API.
- **AI layer (`infra/api/{db,llm,rag,agent,evals,ai}.py`)** — RAG diagnosis,
  agentic workflow, eval harness, and observability over a local Ollama LLM and
  a Postgres + pgvector store.
- **Infra (`infra/podman`, `infra/cloudflared`, `infra/database`)** — Podman
  pods + Cloudflare Tunnel + SQL migrations.
- **Tooling (`tools/cli`)** — fzf-based `cmds` runner.

This repo adopts an in-house full-stack template's **folder design**. There is
**no authentication**. A Postgres + pgvector database backs the AI layer; the
core signal/ML demo still runs without it (graceful degradation).

## The golden rule: analysis lives only in `src/`

The API and frontend must contain **no signal-processing or ML logic**. They
marshal requests/responses and render. If you need new analysis, add it to
`src/` and call it from the API.

## Stack summary

| Layer | Choice | Version |
|-------|--------|---------|
| Engine | NumPy / SciPy / scikit-learn / joblib | current |
| API | Flask + flask-cors; gunicorn (prod) | 3.x |
| Frontend | React / Vite / TypeScript (strict) / MUI / React Router | 19 / — / ~5.9 / 7 / 7 |
| Charts | Recharts | 3.x |
| Tests | pytest; Vitest + Testing Library | — |
| Containers | Podman (`play kube`) | — |
| Ingress | Cloudflare Tunnel | — |

## Backend conventions (Flask)

- **App factory.** `create_app()` in [infra/api/app.py](infra/api/app.py) builds
  the app, sets an order-preserving JSON provider, enables CORS, registers the
  blueprint, and loads the engine once. Module-level `app = create_app()` is what
  gunicorn imports (`app:app`).
- **Blueprint per domain.** The Foreshock routes live in
  [infra/api/predict.py](infra/api/predict.py) as `api_bp` with
  `url_prefix="/api"`. Add new domains as new blueprints and register them in
  `app.py`.
- **Thin routes.** Validate input, call into `src/` (and `engine.py`), format
  JSON. No business logic in handlers.
- **Engine loading.** [infra/api/engine.py](infra/api/engine.py) loads
  `models/model.joblib` + `models/samples.npz` once into a module singleton
  (analogous to the template's `db.py`). Endpoints return 503 until the model is
  trained.
- **Imports.** Modules in `infra/api` use flat imports among themselves
  (`from engine import ...`); `app.py` puts the repo root on `sys.path` so
  `import src` resolves whether run as `python app.py`, gunicorn, or pytest.

### Adding an API endpoint

1. Implement the analysis in `src/` (a function in `features.py`/`model.py`/etc.).
2. Add a thin route to `infra/api/predict.py` (or a new blueprint) that calls it.
3. Add a typed wrapper in `frontend/src/api/foreshock.ts`.
4. Render it in a component/page.
5. Add tests: `infra/api/tests/` (pytest) and `frontend/src/__tests__/` (Vitest).

## Frontend conventions (React + MUI)

- **Single API client.** All requests go through `apiFetch` / `apiJson` in
  [frontend/src/api/client.ts](frontend/src/api/client.ts) (FormData-aware). Typed
  domain calls live in [frontend/src/api/foreshock.ts](frontend/src/api/foreshock.ts).
- **Theme, not hardcoded colors.** Use MUI's `sx`/theme; brand + per-condition
  colors live in [frontend/src/theme/theme.ts](frontend/src/theme/theme.ts).
- **Shell.** [frontend/src/components/AppLayout.tsx](frontend/src/components/AppLayout.tsx)
  is the top-bar shell; the single page is
  [frontend/src/pages/AnalyzePage.tsx](frontend/src/pages/AnalyzePage.tsx).
- **Strict TypeScript.** `verbatimModuleSyntax` is on — use `import type` for
  type-only imports. No unused locals/params.

## AI layer conventions (`infra/api/`)

- **Local-only, key-free.** Generation + embeddings run on Ollama
  (`llama3.2:1b`, `nomic-embed-text`) via `llm.py`. Config is env-driven:
  `OLLAMA_HOST`, `LLM_MODEL`, `EMBED_MODEL`.
- **One store for relational + vector.** `db.py` wraps Postgres + pgvector
  (pooled `query`/`execute`, `run_migrations`). Embeddings are passed as numpy
  arrays so pgvector's adapter sends a `vector`, not `numeric[]`.
- **RAG.** `rag.py` embeds the query and retrieves by cosine distance. The
  corpus is seeded by `scripts/seed_kb.py`.
- **Agent.** `agent.py` orchestrates engine analysis + RAG + LLM + DB. Signal
  analysis still comes from `src/`; the agent only orchestrates and persists.
- **Observability.** Every LLM/embedding call logs latency + tokens (+ retrieval
  score) to `llm_calls` from `llm.py`; never let logging break a call.
- **Evals.** `evals.py` scores accuracy, retrieval precision/recall, and
  hallucination; persists to `eval_runs`.
- **Degrade gracefully.** No LLM -> templated diagnosis; no DB -> 503 on AI
  endpoints, but `/api/samples|signal|predict` keep working.

## v2 / v3 conventions

- **v2 health** (`src/health.py`, `infra/api/health_routes.py`). An autoencoder
  (sklearn MLP + PCA - no DL framework) trained on healthy windows only;
  reconstruction error is the health indicator, PCA(2) the embedding.
  `scripts/train_health.py` writes `models/health_ae.joblib` + `health.npz`; the
  API serves the precomputed timeline/embedding and can score a sample. Dataset-
  agnostic: uses NASA IMS under `data/ims/` if present, else a CWRU-derived
  run-to-failure timeline.
- **v3 streaming** (`infra/api/stream.py`). A background thread consumes the
  Kafka topic with MANUAL partition assignment (`assign` + `seek_to_end`) - not a
  consumer group - to avoid rebalance stalls, runs inference, and fans results to
  SSE subscribers. Producer: `scripts/stream_producer.py` or
  `POST /api/stream/simulate`. Note kafka-python 3.x dropped some kwargs
  (`api_version_auto_timeout_ms`, `consumer_timeout_ms`); keep client args minimal.
- Both degrade gracefully: no health model -> 503 on `/api/health/*`; no Kafka ->
  `/api/stream/status` reports `kafka: false` and `/simulate` returns 503.

## Engine conventions (`src/`)

- Constants and bearing geometry/fault-frequency math live in `config.py`.
- Loading and windowing are separate, testable functions in `data_loader.py`;
  `load_dataset` returns `(windows, labels, groups, rpms)` where `group` is the
  source recording.
- `features.py` exposes `extract_features` / `extract_features_batch` with an
  ordered `FEATURE_NAMES`, plus spectrum helpers the API reuses.
- `model.py` is a `StandardScaler -> RandomForest` pipeline with
  train/evaluate/save/load.

## Ports

| Context | API | Frontend |
|---------|-----|----------|
| Host dev | 8000 (`cd infra/api && python app.py`) | 5173 (`npm run dev`) |
| Podman dev pod | 8000 host / 5000 container | 3000 |
| Podman prod pod | 5000 (gunicorn) | 80 (built + `serve`) |

Host dev uses 8000 because macOS reserves 5000 for AirPlay. The Vite proxy target
is `VITE_API_PROXY` (default `http://localhost:8000`); the dev pod sets it to
`http://localhost:5000`.

## Testing

- **Python:** `pytest` from the repo root runs `tests/` (engine) and
  `infra/api/tests/` (API). The repo-root `conftest.py` puts the project root on
  `sys.path`. API tests skip themselves if the model isn't trained.
- **Frontend:** `cd frontend && npm test` (Vitest).
- Or `cmds test`.

## Do NOT

- Put analysis logic in `infra/api` or `frontend` — it belongs in `src/`.
- Hardcode colors in components — use the MUI theme.
- Add authentication (intentionally excluded). The database exists only for the
  AI layer (RAG, diagnoses, observability, evals) - keep the core demo working
  without it.
- Put LLM/DB calls in `src/` - the engine stays pure signal/ML; the AI layer
  lives in `infra/api/`.
- Commit Cloudflare credentials (`infra/cloudflared/creds/*.json` is gitignored).
- Edit the trained `models/` artifacts by hand — regenerate via `scripts/train.py`.
