"""Repo-root pytest setup.

Adds the project root to ``sys.path`` so engine tests can ``import src`` and
the API tests (under infra/api/tests) can reach the engine. Each suite's own
conftest adds any extra paths it needs (e.g. infra/api for ``import app``).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
