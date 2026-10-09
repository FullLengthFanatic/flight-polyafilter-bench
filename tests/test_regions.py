from pafbench.regions import REGION_NAMES, REGIONS, load_region_segments, region_at, regions_at

ROWS = [
    # coding transcript on +: exons 101-200, 301-400, 501-600 (1-based); CDS 151-350
    ("transcript", 101, 600, "+", "tc"), ("exon", 101, 200, "+", "tc"), ("exon", 301, 400, "+", "tc"),
    ("exon", 501, 600, "+", "tc"), ("CDS", 151, 200, "+", "tc"), ("CDS", 301, 350, "+", "tc"),
    # non-coding transcript on + whose exon covers part of tc's intron and of its 3' UTR exon
    ("transcript", 251, 560, "+", "tn"), ("exon", 251, 280, "+", "tn"), ("exon", 541, 560, "+", "tn"),
    # coding transcript on -: exons 1001-1100, 1201-1300; CDS 1051-1250
    ("transcript", 1001, 1300, "-", "tm"), ("exon", 1001, 1100, "-", "tm"), ("exon", 1201, 1300, "-", "tm"),
    ("CDS", 1051, 1100, "-", "tm"), ("CDS", 1201, 1250, "-", "tm"),
]


def _gtf(tmp_path):
    p = tmp_path / "r.gtf"
    with open(p, "w") as fh:
        fh.write("#comment\n")
        for ft, s, e, strand, tid in ROWS:
            fh.write(f"chrZ\tt\t{ft}\t{s}\t{e}\t.\t{strand}\t.\tgene_id \"g_{tid}\"; transcript_id \"{tid}\";\n")
    return p


def test_plus_strand_regions(tmp_path):
    seg = load_region_segments(_gtf(tmp_path))
    at = lambda pos: region_at(seg, "chrZ", True, pos)
    assert at(99) == "intergenic" and at(100) == "5UTR" and at(149) == "5UTR" and at(150) == "CDS"
    assert at(199) == "CDS" and at(200) == "intron" and at(260) == "ncexon" and at(300) == "CDS"
    assert at(349) == "CDS" and at(350) == "3UTR" and at(450) == "intron" and at(545) == "3UTR"
    assert at(599) == "3UTR" and at(600) == "intergenic"


def test_minus_strand_mirrors_utrs_and_strands_are_separate(tmp_path):
    seg = load_region_segments(_gtf(tmp_path))
    got = regions_at(seg, "chrZ", False, [1000, 1049, 1050, 1150, 1249, 1250, 1299, 1300])
    assert list(got) == ["3UTR", "3UTR", "CDS", "intron", "CDS", "5UTR", "5UTR", "intergenic"]
    assert region_at(seg, "chrZ", False, 150) == "antisense" and region_at(seg, "chrZ", False, 99) == "intergenic"
    assert region_at(seg, "chrZ", True, 1150) == "antisense" and region_at(seg, "chrZ", True, 1300) == "intergenic"
    assert region_at(seg, "chrQ", True, 150) == "intergenic"
    assert list(regions_at(seg, "chrZ", True, [])) == []


def test_region_names():
    assert REGIONS == ("intergenic", "intron", "ncexon", "5UTR", "CDS", "3UTR")
    assert REGION_NAMES == REGIONS + ("antisense",)
