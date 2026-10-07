from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VENDOR_POLYAFILTER = REPO / "vendor" / "polyAfilter"
POLYAFILTER_COMMIT = "5ce0098"
REF_DIR = Path("/home/simone.picelli/Documents/Genome_ref/refdata-gex-GRCh38-2024-A")
FASTA = REF_DIR / "fasta" / "genome.fa"
GTF = REF_DIR / "genes" / "genes.gtf"
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
