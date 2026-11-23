import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
import run_scenarios  # noqa: E402


@pytest.fixture(scope="module")
def results(model):
    return run_scenarios.run(model=model)


def test_all_scenarios_pass(results):
    failed = {r["name"]: r["errors"] for r in results if not r["passed"]}
    assert not failed, failed
