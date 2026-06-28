# Podman Configuration

Pod definitions for the Foreshock development and production environments.
There is **no database** in this project (stateless demo), so the pods contain
just the API, the frontend, and a Cloudflare Tunnel.

## Quick Start

```bash
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
| `foreshock-api` | `python:3.11-slim` | 8000 host / 5000 container | 5000 (gunicorn) | REST API over the `src/` engine |
| `foreshock-frontend` | `node:20-alpine` | 3000 (vite dev) | 80 (built + `serve`) | React + MUI web UI |
| `cloudflared` | `cloudflare/cloudflared` | — | — | Public tunnel |

The dev pod uses normal port mappings; the prod pod uses `hostNetwork: true`
(common for single-host deployments behind a tunnel).

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
