import pytest

from tests.synth import build


@pytest.fixture(scope="session")
def synth(tmp_path_factory):
    return build(tmp_path_factory.mktemp("synth"))
