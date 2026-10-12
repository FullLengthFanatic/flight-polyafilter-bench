"""Marker panel of stage 25 (design of 2026-10-12, section 3.3, decision D1): classes scored as their sources define them.

The wording read in each source is in design_m3_draft/research/methods_notes.md, section C.
"""
from collections import namedtuple

from .config import CELLS, NUCLEI

SOURCES = {
    "Lukowski": "Lukowski et al. 2019 EMBO J 38:e100811 (PMC6745503), adult human retina, scRNA-seq",
    "Cowan": "Cowan et al. 2020 Cell 182:1623 (PMC7505495), adult human retina and organoids, scRNA-seq",
    "Menon": "Menon et al. 2019 Nat Commun 10:4902 (PMC6814749), human macula and periphery, scRNA-seq",
    "Yan": "Yan et al. 2020 Sci Rep 10:9802 (PMC7299956), fovea and periphery",
    "Liang": "Liang et al. 2023 Cell Genomics 3:100298 (PMC10300490), human retina, snRNA-seq and snATAC-seq",
    "Li": "Li et al. 2026 Nat Genet (Human Retina Cell Atlas; preprint version PMC10680922)",
    "Sridhar": "Sridhar et al. 2020 Cell Rep 30:1644 (PMC7901645), fetal retina and organoids, scRNA-seq",
}

# weak: two of the three progenitor genes also mark mature classes (VSX2 bipolar, PAX6 amacrine), so the class table
# prints a warning with it
MarkerClass = namedtuple("MarkerClass", "name genes sources libraries weak", defaults=(NUCLEI + CELLS, False))

PANEL = (
    MarkerClass("rod", ("NRL", "NR2E3", "PDE6A", "CNGA1", "GNAT1", "SLC24A1", "GNGT1", "PDE6G", "PPEF2"),
                ("Lukowski", "Cowan", "Menon", "Yan", "Liang")),
    MarkerClass("cone", ("ARR3", "GNAT2", "GNGT2", "GUCA1C", "CNGA3", "PDE6H", "PDE6C", "OPN1LW", "OPN1MW", "OPN1SW"),
                ("Lukowski", "Cowan", "Menon", "Liang")),
    MarkerClass("bipolar", ("VSX2", "OTX2", "NETO1", "GRIK1", "TRPM1", "GRM6", "CAMK2B", "TMEM215", "VSX1", "IRX6", "PRKCA"),
                ("Lukowski", "Li", "Menon", "Cowan", "Yan")),
    # Muller glia and astrocytes as one class, as Menon and Cowan give them
    MarkerClass("macroglia", ("RLBP1", "GLUL", "APOE", "CLU", "SLC1A3", "CRABP1", "GFAP", "SPARCL1"),
                ("Menon", "Cowan", "Lukowski", "Liang", "Sridhar")),
    MarkerClass("amacrine", ("TFAP2A", "TFAP2B", "GAD1", "GAD2", "SLC6A9", "CALB1", "CHAT", "C1QL2", "SLC18A3"),
                ("Cowan", "Li", "Lukowski", "Menon", "Yan")),
    MarkerClass("horizontal", ("ONECUT1", "ONECUT2", "LHX1", "LNP1", "VAT1L"), ("Lukowski", "Menon", "Cowan", "Liang")),
    MarkerClass("retinal ganglion cell", ("RBPMS", "SLC17A6", "NEFL", "NEFM", "SNCG", "THY1", "POU4F2", "GAP43"),
                ("Yan", "Cowan", "Lukowski", "Menon", "Liang")),
    MarkerClass("microglia", ("C1QA", "TMEM119", "AIF1", "CD163", "CX3CR1", "CD74", "HLA-DRA", "HLA-DPA1", "HLA-DPB1"),
                ("Menon", "Cowan", "Lukowski", "Liang")),
    # endothelial and mural cells as one class: the sources read give one vascular set each
    MarkerClass("vascular", ("CD34", "CDH5", "RGS5", "ADAMTS9", "EGFL7", "VWF", "PDGFRB", "ACTA2", "COL1A2", "PROCR"),
                ("Menon", "Cowan")),
    MarkerClass("RPE", ("RPE65", "BEST1", "MITF", "MLANA", "TJP1"), ("Cowan",)),
    MarkerClass("progenitor", ("SOX2", "VSX2", "PAX6"), ("Sridhar",), CELLS, True),
    MarkerClass("proliferating", ("MKI67",), ("Cowan", "Sridhar"), CELLS),
    MarkerClass("neurogenic or transitional", ("ATOH7", "HES6", "PRDM13", "PTF1A", "FABP7", "DLL3", "NEUROD1"),
                ("Sridhar", "Cowan"), CELLS),
    MarkerClass("photoreceptor precursor", ("OTX2", "RCVRN", "FABP7", "DLL3"), ("Sridhar",), CELLS),
)

# second rod score of the class table: the two transcription factors only
ROD_TF = ("NRL", "NR2E3")
NOT_SCORED = {
    "RHO": 'ambient rod RNA reaches other cells: "The highest rod contamination among all the samples was observed in the '
           'snRNA-seq sample, where rhodopsin expression was surprisingly high and homogeneous in non-rod cell populations" '
           "(mouse; Fadl et al. 2020 Mol Vis, https://pmc.ncbi.nlm.nih.gov/articles/PMC7553720/)",
}
# ambient flag (decision D9): UMIs of these over the cell's UMIs; in a class other than rod it is an ambient indicator
AMBIENT_ROD_TRANSCRIPTS = ("RHO", "GNAT1", "PDE6G", "GNGT1", "CNGA1", "PDE6A", "SLC24A1")
