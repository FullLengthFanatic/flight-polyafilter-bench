from pathlib import Path

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
NUCLEI = ("1417", "1419")
CELLS = ("1179",)
