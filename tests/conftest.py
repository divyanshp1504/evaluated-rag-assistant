import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.pipeline import RagPipeline  # noqa: E402


@pytest.fixture(scope="session")
def pipeline():
    return RagPipeline()
