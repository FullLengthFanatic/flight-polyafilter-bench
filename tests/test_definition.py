import inspect
import json
import tomllib

from pafbench import bd67, config, sites
from pafbench.config import REPO
from pafbench.definition import definition


def test_json_on_disk_equals_code():
    assert json.loads((REPO / "docs" / "definition.json").read_text()) == definition()


def test_version_is_the_package_version():
    project = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
    assert definition()["version"] == project["version"] == "1.2.0"


def test_values():
    d = definition()
    assert d["label"] == {"window": 20, "templated_min": 8, "nontemplated_max": 4}
    assert d["site"]["short_read"]["clip_min"] == 20 and d["site"]["short_read"]["clip_purity_min"] == 0.9
    assert d["site"]["long_read"]["run_min"] == 10 and d["site"]["long_read"]["run_tolerance"] == 1
    assert d["site"]["long_read"]["min_mapq"] == 20 and d["rate"]["unique_mapq"] == 255


def test_function_defaults_are_the_constants():
    sig = inspect.signature
    assert sig(sites.sense_context).parameters["down"].default == config.SITE_WINDOW
    assert sig(sites.context_from_seq).parameters["down"].default == config.SITE_WINDOW
    assert sig(bd67.terminal_a_run).parameters["tolerance"].default == config.LONG_READ_RUN_TOLERANCE
    assert sig(bd67.accept_long_read).parameters["min_run"].default == config.LONG_READ_RUN_MIN
    assert sig(bd67.read_outcome).parameters["min_run"].default == config.LONG_READ_RUN_MIN
    assert sig(bd67.scan_long_reads).parameters["min_mapq"].default == config.LONG_READ_MIN_MAPQ
