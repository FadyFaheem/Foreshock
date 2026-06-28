#!/usr/bin/env sh
#
# First-boot model training, run by the pod's `trainer` init container (and usable
# by hand: `sh scripts/first_boot_train.sh`).
#
# Idempotent: trains only what is missing, so it is a fast no-op once the models
# exist. Best-effort: any step that fails degrades gracefully (the affected
# feature returns 503) instead of blocking the pod.
#
#   - Classifier (models/model.joblib + samples.npz): CWRU 12 kHz drive-end subset.
#   - Health model (models/health.npz): NASA IMS run-to-failure data when USE_IMS=1
#     (default), else a CWRU-derived timeline. IMS is unlabeled degradation data,
#     so only the health/anomaly model uses it; the 4-class classifier stays CWRU.
#
# Env:
#   USE_IMS=1   download + train the health model on NASA IMS (default; 0 to skip)

set -u

# Resolve the repo root whether run in the pod (/app) or from a host checkout.
cd "$(CDPATH= cd "$(dirname "$0")/.." && pwd)" || exit 0

need_clf=0
need_health=0
{ [ -f models/model.joblib ] && [ -f models/samples.npz ]; } || need_clf=1
[ -f models/health.npz ] || need_health=1

if [ "$need_clf" = 0 ] && [ "$need_health" = 0 ]; then
  echo "[trainer] models present - skipping training"
  exit 0
fi

echo "[trainer] first boot: installing Python deps"
pip install --quiet --no-cache-dir -r infra/api/requirements.txt || {
  echo "[trainer] pip install failed - skipping training"
  exit 0
}

# --- classifier (CWRU) --------------------------------------------------------
if [ "$need_clf" = 1 ]; then
  echo "[trainer] downloading CWRU subset + training the classifier"
  python scripts/download_data.py && python scripts/train.py \
    || echo "[trainer] classifier training failed (Diagnostics/Analyze will 503)"
fi

# --- health model (NASA IMS, optional) ----------------------------------------
if [ "$need_health" = 1 ]; then
  if [ "${USE_IMS:-1}" = "1" ] && [ -z "$(ls -A data/ims 2>/dev/null)" ]; then
    echo "[trainer] fetching NASA IMS run-to-failure data (one-time, ~1 GB)"
    # Extractors for the nested zip -> 7z -> rar archive (best-effort).
    pip install --quiet --no-cache-dir py7zr || true
    if command -v apt-get >/dev/null 2>&1; then
      apt-get update -qq >/dev/null 2>&1 \
        && apt-get install -y -qq --no-install-recommends unar >/dev/null 2>&1 || true
    fi
    python scripts/download_ims.py \
      || echo "[trainer] IMS download failed - health model will use the CWRU fallback"
  fi
  echo "[trainer] training the health model"
  python scripts/train_health.py \
    || echo "[trainer] health training failed (v2 Health tab will 503)"
fi

echo "[trainer] done"
