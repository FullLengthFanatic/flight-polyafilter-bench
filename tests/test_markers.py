import gzip
import re
from collections import Counter

from pafbench.config import CELLS, LIBRARIES, NUCLEI, REPO
from pafbench.markers import AMBIENT_ROD_TRANSCRIPTS, NOT_SCORED, PANEL, ROD_TF, SOURCES, MarkerClass

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


def test_panel_is_the_table_of_the_design():
    # every row of the marker table of section 3.3: class, genes and sources in the order written there
    lines = (REPO / "docs/specs/2026-10-12-polyafilter-bench-milestone3-design.md").read_text().splitlines()
    start = lines.index("| Class | Genes scored | Sources |") + 2
    table = []
    for line in lines[start:]:
        if not line.startswith("|"):
            break
        name, genes, sources = (cell.strip() for cell in line.strip("|").split("|"))
        only_1179, weak = name.startswith("1179 only: "), name.endswith("(weak)")
        name = re.sub(r" \(.*\)$", "", name.removeprefix("1179 only: "))
        sources = re.sub(r" \(.*?\)| \d{4}", "", sources).split(", ")     # the quotes, "(HRCA)" and the years
        table.append(MarkerClass(name, tuple(genes.split(", ")), tuple(sources), CELLS if only_1179 else NUCLEI + CELLS, weak))
    assert len(table) == 14 and tuple(table) == PANEL


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
