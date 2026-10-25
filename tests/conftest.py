import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))

from fixtures import TEAM_REPO, make_fake_patches, make_fake_weights  # noqa: E402

if not (TEAM_REPO / "model.py").exists():
    raise RuntimeError(f"loveda-semantic-editing 저장소가 필요해요: {TEAM_REPO} (환경 변수 TEAM_REPO로 지정 가능)")
sys.path.insert(0, str(TEAM_REPO))


@pytest.fixture(scope="session")
def fake_patches(tmp_path_factory):
    return make_fake_patches(tmp_path_factory.mktemp("patches"))


@pytest.fixture(scope="session")
def fake_weights(tmp_path_factory):
    return make_fake_weights(tmp_path_factory.mktemp("weights"))


@pytest.fixture(scope="session")
def team_repo():
    return TEAM_REPO
