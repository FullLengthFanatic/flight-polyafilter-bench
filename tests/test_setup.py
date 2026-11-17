import shutil
import subprocess
import tomllib
from importlib.metadata import version

import pytest

from pafbench.config import FASTA, GTF, LIBRARIES, POLYAFILTER_COMMIT, REPO, TECAP_VERSION, VENDOR_POLYAFILTER
from pafbench.tecap import tecap_path
from pafbench.vendor import polyafilter_modules


def test_vendor_commit():
    head = subprocess.run(
        ["git", "-C", str(VENDOR_POLYAFILTER), "rev-parse", "--short=7", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert head == POLYAFILTER_COMMIT


def test_tecap_version():
    if shutil.which("tecap") is None:
        pytest.skip("tecap is not on PATH (stage 30 is optional, --no-tecap)")
    out = subprocess.run([tecap_path(), "--version"], capture_output=True, text=True, check=True).stdout.strip()
    assert out == f"tecap {TECAP_VERSION}"


def test_vendor_imports():
    BAMfilter, RNAseqAnalysis, SNRanalysis = polyafilter_modules()
    assert hasattr(BAMfilter, "scanBAMfilter") and hasattr(BAMfilter, "findSNRpieces")
    assert hasattr(RNAseqAnalysis, "getTransStartEnding") and hasattr(RNAseqAnalysis, "getTransEnding")
    assert hasattr(SNRanalysis, "getOverlaps") and hasattr(SNRanalysis, "removeOverlaps")


def test_inputs_exist():
    assert FASTA.exists() and GTF.exists()
    for name in ("1417", "1419", "1179", "1416"):
        assert LIBRARIES[name].exists(), name


def test_dependencies_are_declared_and_installed():
    project = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
    assert project["requires-python"] == ">=3.11"  # tomllib, read here, is in the standard library from 3.11
    deps = project["dependencies"]
    assert {"scanpy[scrublet]", "anndata", "scikit-learn", "leidenalg", "igraph", "h5py", "plotly", "umap-learn",
            "threadpoolctl"} <= set(deps)
    for d in deps:
        assert version(d.split("[")[0]), d
    assert version("scikit-image")  # what the scrublet extra of scanpy adds
