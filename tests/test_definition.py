import inspect
import json
import subprocess
import sys
import tomllib

from pafbench import bd67, config, lrcontext, sites
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
    assert sig(lrcontext.scan_context).parameters["min_mapq"].default == config.LONG_READ_MIN_MAPQ


# A literal equal to its constant passes the test above. Here every constant is set to a value it does not have before
# the consumers are imported, and each consumer has to follow. spike_pa_row is probed through the tolerance it hands to
# terminal_a_run (pa_row stubbed, the call stopped once seen).
PROBE = """
import inspect, json
from pafbench import config
config.SITE_WINDOW, config.LONG_READ_RUN_MIN, config.LONG_READ_RUN_TOLERANCE, config.LONG_READ_MIN_MAPQ = 25, 11, 2, 30
from pafbench import bd67, lrcontext, sites, spikein

class Seen(Exception):
    pass

def spy(*args, **kwargs):
    call = inspect.signature(bd67.terminal_a_run).bind(*args, **kwargs)
    call.apply_defaults()
    raise Seen(call.arguments["tolerance"])

spikein.pa_row, spikein.terminal_a_run = lambda *_: [None] * len(spikein.PA_COLUMNS), spy
try:
    spikein.spike_pa_row(None, None, None)
except Seen as e:
    spike_tolerance = e.args[0]
d = lambda f, p: inspect.signature(f).parameters[p].default
print(json.dumps({"sites.sense_context": d(sites.sense_context, "down"),
                  "sites.context_from_seq": d(sites.context_from_seq, "down"),
                  "bd67.terminal_a_run": d(bd67.terminal_a_run, "tolerance"),
                  "bd67.accept_long_read": d(bd67.accept_long_read, "min_run"),
                  "bd67.read_outcome": d(bd67.read_outcome, "min_run"),
                  "bd67.scan_long_reads": d(bd67.scan_long_reads, "min_mapq"),
                  "lrcontext.scan_context": d(lrcontext.scan_context, "min_mapq"),
                  "spikein.spike_pa_row": spike_tolerance}))
"""


def test_consumers_follow_the_constants():
    out = subprocess.run([sys.executable, "-B", "-c", PROBE], cwd=REPO, check=True, stdout=subprocess.PIPE, text=True)
    assert json.loads(out.stdout) == {"sites.sense_context": 25, "sites.context_from_seq": 25, "bd67.terminal_a_run": 2,
                                      "bd67.accept_long_read": 11, "bd67.read_outcome": 11, "bd67.scan_long_reads": 30,
                                      "lrcontext.scan_context": 30, "spikein.spike_pa_row": 2}
