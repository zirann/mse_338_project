import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: tests that exercise real models or are otherwise expensive; "
        "opt-in via `pytest -q -m slow`.",
    )
