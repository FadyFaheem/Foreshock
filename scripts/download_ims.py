"""Fetch the NASA IMS bearing run-to-failure dataset (optional, for v2).

The IMS set is large (~6 GB) and the official NASA Prognostics repository is
frequently unavailable, so this script tries an optional mirror and otherwise
prints manual-download instructions. v2 works without it (it falls back to a
CWRU-derived run-to-failure timeline).

    python scripts/download_ims.py            # prints manual steps
    IMS_MIRROR_URL=<zip-url> python scripts/download_ims.py   # try a mirror
"""

from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402

IMS_DIR = config.DATA_DIR / "ims"
_MANUAL = """\
======================================================================
NASA IMS Bearing Data Set - manual download
The dataset is large (~6 GB) and the official host is often offline.

1. Get it from the NASA Prognostics Data Repository ("IMS Bearings") or a
   mirror (e.g. Kaggle: "NASA Bearing Dataset").
2. Unzip a test set so snapshots land under:
       data/ims/<test>/<timestamped snapshot files>
   e.g. data/ims/2nd_test/2004.02.12.10.32.39
3. Re-run:  python scripts/train_health.py
   (it auto-detects data/ims and uses the real run-to-failure timeline).
======================================================================"""


def main() -> int:
    IMS_DIR.mkdir(parents=True, exist_ok=True)
    mirror = os.getenv("IMS_MIRROR_URL")
    if not mirror:
        print(_MANUAL)
        return 0

    dest = IMS_DIR / "ims_download.zip"
    print(f"Downloading IMS from {mirror} ...")
    try:
        req = Request(mirror, headers={"User-Agent": "ForeshockIMS/1.0"})
        with urlopen(req, timeout=120) as resp:
            dest.write_bytes(resp.read())
        with zipfile.ZipFile(dest) as zf:
            zf.extractall(IMS_DIR)
        dest.unlink(missing_ok=True)
        print(f"Extracted IMS into {IMS_DIR}. Now run: python scripts/train_health.py")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"Mirror download failed: {exc}")
        print(_MANUAL)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
