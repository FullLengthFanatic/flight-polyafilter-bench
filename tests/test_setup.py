import subprocess
import tomllib
from importlib.metadata import version

from pafbench.config import FASTA, GTF, LIBRARIES, POLYAFILTER_COMMIT, REPO, VENDOR_POLYAFILTER
from pafbench.vendor import polyafilter_modules


def test_vendor_commit():
    head = subprocess.run(
        ["git", "-C", str(VENDOR_POLYAFILTER), "rev-parse", "--short=7", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert head == POLYAFILTER_COMMIT


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
    deps = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["dependencies"]
    assert {"scanpy[scrublet]", "anndata", "scikit-learn", "leidenalg", "igraph", "h5py", "plotly", "umap-learn"} <= set(deps)
    for d in deps:
        assert version(d.split("[")[0]), d
    assert version("scikit-image")  # what the scrublet extra of scanpy adds
