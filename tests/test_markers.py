import gzip
from collections import Counter

from pafbench.config import CELLS, LIBRARIES, NUCLEI
from pafbench.markers import AMBIENT_ROD_TRANSCRIPTS, NOT_SCORED, PANEL, ROD_TF, SOURCES

BY_NAME = {c.name: c for c in PANEL}
# symbols of the panel that the reference of the libraries lacks; none today. A symbol goes here only after it was reported.
ABSENT_FROM_REFERENCE = set()


def test_panel_classes():
    assert [c.name for c in PANEL] == [
        "rod", "cone", "bipolar", "macroglia", "amacrine", "horizontal", "retinal ganglion cell", "microglia", "vascular",
        "RPE", "progenitor", "proliferating", "neurogenic or transitional", "photoreceptor precursor"]
    for c in PANEL:
        assert len(c.genes) >= 1 and len(c.sources) >= 1, c.name
        assert len(set(c.genes)) == len(c.genes), c.name
        assert set(c.sources) <= set(SOURCES), c.name
    assert {s for c in PANEL for s in c.sources} == set(SOURCES)
    assert sum(len(c.genes) for c in PANEL) == 99


def test_panel_rows_of_the_design_table():
    assert BY_NAME["rod"].genes == ("NRL", "NR2E3", "PDE6A", "CNGA1", "GNAT1", "SLC24A1", "GNGT1", "PDE6G", "PPEF2")
    assert BY_NAME["rod"].sources == ("Lukowski", "Cowan", "Menon", "Yan", "Liang")
    assert BY_NAME["vascular"].genes == ("CD34", "CDH5", "RGS5", "ADAMTS9", "EGFL7", "VWF", "PDGFRB", "ACTA2", "COL1A2", "PROCR")
    assert BY_NAME["vascular"].sources == ("Menon", "Cowan")
    assert BY_NAME["proliferating"].genes == ("MKI67",)
    assert BY_NAME["photoreceptor precursor"].genes == ("OTX2", "RCVRN", "FABP7", "DLL3")


def test_last_four_classes_are_for_1179_only():
    for c in PANEL[:10]:
        assert c.libraries == NUCLEI + CELLS, c.name
    for c in PANEL[10:]:
        assert c.libraries == CELLS == ("1179",), c.name


def test_weak_mark_is_on_progenitor_only():
    assert [c.name for c in PANEL if c.weak] == ["progenitor"]


def test_shared_genes_are_the_four_the_design_names():
    n = Counter(g for c in PANEL for g in c.genes)
    assert {g for g, k in n.items() if k > 1} == {"VSX2", "OTX2", "FABP7", "DLL3"}
    assert {c.name for c in PANEL if "VSX2" in c.genes} == {"bipolar", "progenitor"}
    assert {c.name for c in PANEL if "OTX2" in c.genes} == {"bipolar", "photoreceptor precursor"}
    for g in ("FABP7", "DLL3"):
        assert {c.name for c in PANEL if g in c.genes} == {"neurogenic or transitional", "photoreceptor precursor"}


def test_rod_subsets_and_genes_left_out():
    assert ROD_TF == ("NRL", "NR2E3") and set(ROD_TF) <= set(BY_NAME["rod"].genes)
    assert list(NOT_SCORED) == ["RHO"]
    assert not any("RHO" in c.genes for c in PANEL)
    assert AMBIENT_ROD_TRANSCRIPTS == ("RHO", "GNAT1", "PDE6G", "GNGT1", "CNGA1", "PDE6A", "SLC24A1")
    assert set(AMBIENT_ROD_TRANSCRIPTS) - {"RHO"} <= set(BY_NAME["rod"].genes)


def test_every_symbol_is_in_the_reference_of_the_libraries():
    symbols = {g for c in PANEL for g in c.genes} | set(AMBIENT_ROD_TRANSCRIPTS) | set(NOT_SCORED)
    for lib in NUCLEI + CELLS:
        with gzip.open(LIBRARIES[lib].parent / "filtered_feature_bc_matrix" / "features.tsv.gz", "rt") as fh:
            names = Counter(line.split("\t")[1] for line in fh)
        assert {g for g in symbols if names[g] == 0} == ABSENT_FROM_REFERENCE, lib
        assert all(names[g] == 1 for g in symbols - ABSENT_FROM_REFERENCE), lib
