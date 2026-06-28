"""Fetch the real NASA IMS bearing run-to-failure dataset (for v2).

Downloads the NASA PCoE "Bearings" package (mirrored on the PHM Society S3
bucket), unpacks the nested archives, and drops one run-to-failure test set under
``data/ims/<test>/`` so :mod:`scripts.train_health` trains the autoencoder on
genuinely progressive degradation instead of the CWRU-derived fallback.

The package nests three archive layers::

    4.+Bearings.zip  ->  4. Bearings/IMS.7z  ->  <test>.rar  ->  <snapshots>

so we stream the ~1 GB zip, pull out ``IMS.7z`` (zip), extract just the chosen
``<test>.rar`` (py7zr, or bsdtar), then unpack the snapshots (bsdtar/unar/unrar).

    python scripts/download_ims.py                  # 2nd_test (default, ~0.5 GB)
    python scripts/download_ims.py --test 1st_test  # a different run
    python scripts/download_ims.py --keep-archives  # keep the downloads
    IMS_MIRROR_URL=<zip-url> python scripts/download_ims.py

v2 still works without this (it falls back to a CWRU-derived timeline).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402

IMS_DIR = config.DATA_DIR / "ims"
# NASA PCoE repository, mirrored by the PHM Society on S3 (stable, fast).
DEFAULT_URL = "https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip"
INNER_7Z = "4. Bearings/IMS.7z"
VALID_TESTS = ("1st_test", "2nd_test", "3rd_test")

_MANUAL = """\
======================================================================
Automated IMS download failed. To install the data manually:

1. Download the "4. Bearings" package from the NASA Prognostics Data
   Repository ("Bearings" / IMS) or its PHM Society S3 mirror:
       https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip
2. Inside: 4. Bearings/IMS.7z -> <test>.rar (e.g. 2nd_test.rar).
3. Unpack a test set so snapshots land under:
       data/ims/<test>/<timestamped snapshot files>
   e.g. data/ims/2nd_test/2004.02.12.10.32.39
4. Re-run:  python scripts/train_health.py
   (it auto-detects data/ims and uses the real run-to-failure timeline).
======================================================================"""


def _download(url: str, dest: Path) -> None:
    print(f"Downloading IMS package from {url}")
    req = Request(url, headers={"User-Agent": "ForeshockIMS/1.0"})
    with urlopen(req, timeout=180) as resp, open(dest, "wb") as fh:  # noqa: S310
        total = int(resp.headers.get("Content-Length", 0))
        read = 0
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
            read += len(chunk)
            if total:
                print(
                    f"\r  {read / 1e6:7.1f} / {total / 1e6:.1f} MB "
                    f"({100 * read / total:4.1f}%)",
                    end="",
                    flush=True,
                )
        print()


def _extract_zip_member(zip_path: Path, member: str, dest: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf, zf.open(member) as src, open(dest, "wb") as out:
        shutil.copyfileobj(src, out, 1 << 20)


def _extract_7z_member(sevenz: Path, member: str, out_dir: Path) -> Path:
    """Extract a single member from a .7z (py7zr if present, else bsdtar)."""
    try:
        import py7zr  # noqa: PLC0415  (optional; bsdtar is the fallback)

        with py7zr.SevenZipFile(sevenz) as archive:
            archive.extract(path=out_dir, targets=[member])
    except ImportError:
        if not shutil.which("bsdtar"):
            raise RuntimeError(
                "Need py7zr (pip install py7zr) or bsdtar to read the .7z"
            ) from None
        subprocess.run(
            ["bsdtar", "-xf", str(sevenz), "-C", str(out_dir), member], check=True
        )
    return out_dir / member


def _extract_rar(rar_path: Path, out_dir: Path) -> None:
    """Unpack a .rar using whichever extractor is installed."""
    candidates = (
        ["bsdtar", "-xf", str(rar_path), "-C", str(out_dir)],
        ["unar", "-force-overwrite", "-quiet", "-output-directory", str(out_dir), str(rar_path)],
        ["unrar", "x", "-y", str(rar_path), f"{out_dir}{os.sep}"],
    )
    for cmd in candidates:
        if shutil.which(cmd[0]):
            subprocess.run(cmd, check=True)
            return
    raise RuntimeError("No RAR extractor found (install bsdtar, unar, or unrar).")


def _has_snapshots(test_dir: Path) -> bool:
    return test_dir.is_dir() and any(
        p.is_file() and not p.name.startswith(".") for p in test_dir.iterdir()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test", default="2nd_test", choices=VALID_TESTS, help="which IMS test set"
    )
    parser.add_argument(
        "--keep-archives", action="store_true", help="keep the downloaded archives"
    )
    args = parser.parse_args()

    test_dir = IMS_DIR / args.test
    if _has_snapshots(test_dir):
        n = sum(1 for _ in test_dir.iterdir())
        print(f"IMS already present: {test_dir} ({n} snapshots). Nothing to do.")
        print("Next: python scripts/train_health.py")
        return 0

    IMS_DIR.mkdir(parents=True, exist_ok=True)
    url = os.getenv("IMS_MIRROR_URL", DEFAULT_URL)
    zip_path = IMS_DIR / "bearings.zip"
    sevenz = IMS_DIR / "IMS.7z"
    rar = IMS_DIR / f"{args.test}.rar"

    try:
        if not zip_path.exists():
            _download(url, zip_path)
        print(f"Unpacking {INNER_7Z}")
        _extract_zip_member(zip_path, INNER_7Z, sevenz)
        print(f"Unpacking {args.test}.rar")
        _extract_7z_member(sevenz, f"{args.test}.rar", IMS_DIR)
        print(f"Extracting {args.test} snapshots")
        _extract_rar(rar, IMS_DIR)
    except Exception as exc:  # noqa: BLE001
        print(f"\nAutomated download failed: {exc}\n")
        print(_MANUAL)
        return 1

    if not args.keep_archives:
        for path in (zip_path, sevenz, rar):
            path.unlink(missing_ok=True)

    if not _has_snapshots(test_dir):
        print(f"Extraction finished but no snapshots found under {test_dir}.")
        print(_MANUAL)
        return 1

    n = sum(1 for _ in test_dir.iterdir())
    print(f"Done: {n} snapshots under {test_dir}.")
    print("Next: python scripts/train_health.py   (retrains v2 on real IMS data)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
