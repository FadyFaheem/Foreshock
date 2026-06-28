"""Download a curated subset of the CWRU Bearing Data Center dataset.

We fetch the 12 kHz Drive-End recordings for four conditions (Normal, Inner
race, Outer race @6 o'clock, Ball) at all four motor loads (0-3 HP). Having
multiple recordings per condition lets training split by *recording* and avoid
window leakage.

Run from the repo root:

    python scripts/download_data.py

Files are validated with ``scipy.io.loadmat`` and saved under
``data/<condition>/<NNN>.mat``. If automatic download fails (the CWRU site is
occasionally unavailable), the script prints manual-download instructions; the
same table is in the README.
"""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# Make ``src`` importable when run as a plain script (python scripts/...).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scipy.io  # noqa: E402

from src import config  # noqa: E402

# CWRU file numbers: 12 kHz Drive End, 0.007" faults, loads 0/1/2/3 HP.
FILE_NUMBERS: dict[str, list[int]] = {
    "normal": [97, 98, 99, 100],
    "inner_race": [105, 106, 107, 108],
    "outer_race": [130, 131, 132, 133],  # Outer race @6 o'clock (centred)
    "ball": [118, 119, 120, 121],
}

# Candidate URL templates, tried in order. The first is the current official
# host; the second is the legacy host kept as a fallback mirror.
URL_TEMPLATES: tuple[str, ...] = (
    "https://engineering.case.edu/sites/default/files/{n}.mat",
    "https://csegroups.case.edu/sites/default/files/{n}.mat",
)

_USER_AGENT = "Mozilla/5.0 (compatible; ForeshockDownloader/1.0)"
_TIMEOUT_S = 60


def _is_valid_mat(path: Path) -> bool:
    """A .mat file is usable if scipy can read it and it has a DE_time series."""
    try:
        mat = scipy.io.loadmat(str(path))
    except Exception:
        return False
    return any(
        isinstance(k, str) and k.endswith("DE_time") for k in mat
    )


def _download(url: str, dest: Path) -> bool:
    """Fetch ``url`` into ``dest``. Returns True on a valid .mat download."""
    try:
        req = Request(url, headers={"User-Agent": _USER_AGENT})
        with urlopen(req, timeout=_TIMEOUT_S) as resp:
            data = resp.read()
    except (HTTPError, URLError, TimeoutError) as exc:
        print(f"    - {url} failed: {exc}")
        return False

    dest.write_bytes(data)
    if _is_valid_mat(dest):
        return True
    print(f"    - {url} returned an invalid/HTML payload, discarding")
    dest.unlink(missing_ok=True)
    return False


def _fetch_file(number: int, dest: Path) -> bool:
    """Try each candidate URL for a file number until one validates."""
    for template in URL_TEMPLATES:
        if _download(template.format(n=number), dest):
            return True
    return False


def main() -> int:
    print("Downloading CWRU 12 kHz Drive-End subset into", config.DATA_DIR)
    missing: list[tuple[str, int, Path]] = []
    ok = 0

    for condition, numbers in FILE_NUMBERS.items():
        cond_dir = config.DATA_DIR / condition
        cond_dir.mkdir(parents=True, exist_ok=True)
        for number in numbers:
            dest = cond_dir / f"{number}.mat"
            if dest.exists() and _is_valid_mat(dest):
                print(f"  [skip] {condition}/{number}.mat already present")
                ok += 1
                continue
            print(f"  [get ] {condition}/{number}.mat")
            if _fetch_file(number, dest):
                ok += 1
            else:
                missing.append((condition, number, dest))

    total = sum(len(v) for v in FILE_NUMBERS.values())
    print(f"\nDownloaded/verified {ok}/{total} files.")

    if missing:
        _print_manual_instructions(missing)
        return 1
    print("All files present. You can now run: python scripts/train.py")
    return 0


def _print_manual_instructions(missing: list[tuple[str, int, Path]]) -> None:
    print("\n" + "=" * 70)
    print("Some files could not be downloaded automatically.")
    print("Download them manually from the CWRU Bearing Data Center:")
    print("  Normal baseline : https://engineering.case.edu/bearingdatacenter/normal-baseline-data")
    print("  12k Drive-End   : https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data")
    print("\nSave each numbered .mat file to the path shown below:")
    for condition, number, dest in missing:
        print(f"  {number}.mat  ->  {dest}")
    print("=" * 70)


if __name__ == "__main__":
    raise SystemExit(main())
