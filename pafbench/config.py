from pathlib import Path

from .flags import Setting
from .grid import published

REPO = Path(__file__).resolve().parents[1]
VENDOR_POLYAFILTER = REPO / "vendor" / "polyAfilter"
POLYAFILTER_COMMIT = "5ce0098"
REF_DIR = Path("/home/simone.picelli/Documents/Genome_ref/refdata-gex-GRCh38-2024-A")
FASTA = REF_DIR / "fasta" / "genome.fa"
GTF = REF_DIR / "genes" / "genes.gtf"
POLYASITE = Path("/data/scNoiseMeter/atlas.clusters.3.0.GRCh38.GENCODE_42.bed.gz")
RUNS = Path("/data/element_runs")
OUT = RUNS / "polyafilter_bench"
LIBRARIES = {
    "1417": RUNS / "1417_HsRetina_D28_3k" / "outs" / "possorted_genome_bam.bam",
    "1419": RUNS / "1419_HsRetina_D29_3k" / "outs" / "possorted_genome_bam.bam",
    "1179": RUNS / "1179_organoid_wk26_10xv4" / "outs" / "possorted_genome_bam.bam",
    "1416": RUNS / "1416_HsRetina_D28_20k" / "outs" / "possorted_genome_bam.bam",
}
BD67_BAM = RUNS / "BD67" / "BD67_2x_RT.fltnc.aligned.bam"
BD70_BAM = RUNS / "BD70" / "BD70.fltnc.aligned.bam"
LONG_READS = {"BD67": BD67_BAM, "BD70": BD70_BAM}
SIRV_DIR = OUT / "ref" / "sirv"
SIRV_FASTA = SIRV_DIR / "SIRV_ERCC_longSIRV_multi-fasta_20250402.fasta"
SIRV_GTF = SIRV_DIR / "SIRV_ERCC_longSIRV_multi-fasta-annotation_20250402.gtf"
ERCC_CONC_URL = "https://assets.thermofisher.com/TFS-Assets/LSG/manuals/cms_095046.txt"
LABEL_CUTOFFS = {"templated_min": 8, "nontemplated_max": 4}
A_BANDS = (("0-4", 0, 4), ("5-7", 5, 7), ("8-9", 8, 9), ("10-13", 10, 13), ("14-20", 14, 20))
MEMBER_CAP = 1000
SENS_CAPS = (500, 1000, 2000)
POPULATIONS = ("all", "cells", "noncells")
SEL_POPULATION = "cells"  # barcodes the headline selection and burden are computed on
SEL_UNITS = "with_member"  # molecule units of the headline selection
NUCLEI = ("1417", "1419")
CELLS = ("1179",)

# Milestone 3 (design of 2026-10-12, section 3.0). The clustering values and the thresholds are our proposals.
CHOSEN_SETTINGS = (Setting(250, 7, 1, False), Setting(200, 10, 2, True))  # eval/selection.json, modes e and i
CELLS_TUNED_SETTINGS = (Setting(250, 7, 2, False), Setting(450, 5, 1, True))  # eval/selection.json, cells_tuned 1179
# chosen, published, cells-tuned, then the remove-all control of each chosen setting (decision D3). The position of a
# setting is its bit in the removal mask of stage 23: append, never reorder.
IMPACT_SETTINGS = (*CHOSEN_SETTINGS, *published(), *CELLS_TUNED_SETTINGS,
                   *(Setting(s.covlen, None, 0, s.introns) for s in CHOSEN_SETTINGS))
XF_RECOUNT_TOLERANCE = 0
RULE_AGREEMENT_MAX = 1e-6
A4_PARALLEL = 4
UNIT_BIN = 1000
# leiden_flavor and leiden_n_iterations are the scanpy.tl.leiden call that tests/test_scanpy_stack.py checks; the igraph and
# the leidenalg flavour give different partitions
CLUSTER = {"target_sum": 10_000, "n_hvg": 2000, "n_pcs": 30, "k": 15, "resolution": 1.0, "resolutions_alt": (0.5, 2.0),
           "seeds": tuple(range(10)), "leiden_flavor": "igraph", "leiden_n_iterations": 2}
CLASS_MARGIN = 0.1
KNN_K = 15
KNN_FOLDS = 5
THIN_REPLICATES = 1
THIN_REPLICATES_CHOSEN = 3
MIN_CELLS_GENE = 3
RARE_PCTL = 99
TOP_GENES = 20
MARKER_LOSS = 0.5
DOWNSAMPLE_DRAWS = 100
JACCARD_DISSOLVED = 0.5
JACCARD_STABLE = 0.75
BIAS_POINTS = 5
EXPR_BINS = 4
MIN_STRATUM = 50
POOL_MIN = 30
REGION_REF_MIN = 200
SITE_TOL = 5
LONG_END_TOL = 10
NEAR_INTRON_TOL = 5
MIN_INTRON = 50
FIVE_PRIME_TOL = 50
NULL_SHIFT = 1000
DIST_END_BINS = (100, 300, 1000, 5000)
RERUN_DIFF = 0.001
PIGEON_DIR = Path("/data/long_read_annotation_tool/test_dataset/input_files")
REPORT_DIR = OUT / "report"
