import pytest

from tests.synth import build, build_10x


@pytest.fixture(scope="session")
def synth(tmp_path_factory):
    return build(tmp_path_factory.mktemp("synth"))


@pytest.fixture(scope="session")
def synth10x(tmp_path_factory):
    return build_10x(tmp_path_factory.mktemp("synth10x"))
