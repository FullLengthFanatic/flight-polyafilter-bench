import importlib.util
from collections import defaultdict

import pandas as pd
import pysam
import pytest

from pafbench.config import REPO
from pafbench.exons import load_collapsed_exons, member_distance
from pafbench.genes import genes_at, load_gene_segments
from pafbench.molecules import MEMBER_COLUMNS, gene_match, member_distances, member_rows
from pafbench.sites import last_aligned_base

BY_K3 = {
    ("c0", "u0", "+"): [(0, "chrA", 3000, "g1"), (1, "chrB", 1500, "g3")],
    ("c1", "u1", "+"): [(2, "chrA", 2600, "g1")],
    ("c2", "u0", "-"): [(3, "chrA", 4300, "g2")],
    ("c0", "u1", "-"): [(4, "chrA", 5000, "")],
    ("c1", "u0", "+"): [(5, "chrB", 1500, "g3")],
}

# (ref, reference_start, cigar, flag) -> tags; every other record gets tags by its index (see _tagged)
OVERRIDE = {
    ("chrA", 950, "90M", 0): {"CB": "c0", "UB": "u0"},
    ("chrA", 1340, "60M400N30M", 0): {"CB": "c1", "UB": "u1"},
    ("chrA", 4396, "90M", 16): {"CB": "c2", "UB": "u0"},
    ("chrA", 5052, "90M", 16): {"CB": "c0", "UB": "u1"},
    ("chrA", 3400, "90M", 1024): {"CB": "c0", "UB": "u0"},
    ("chrA", 1357, "90M", 0): {"CB": "c0", "UB": "u0", "pa": 12},   # pa read: never a member
    ("chrA", 1394, "90M", 0): {"CB": "c0"},                          # no UB
    ("chrA", 1250, "90M", 256): {"CB": "c0", "UB": "u0"},            # secondary
}


@pytest.fixture(scope="module")
def tagged(synth, tmp_path_factory):
    """Copy of the synthetic BAM with CB/UB cycling over the molecule keys, some reads without CB or UB, some pa reads."""
    out = tmp_path_factory.mktemp("members") / "tagged.bam"
    with pysam.AlignmentFile(str(synth.bam)) as fin, pysam.AlignmentFile(str(out), "wb", template=fin) as fout:
        for i, a in enumerate(fin.fetch(until_eof=True)):
            tags = OVERRIDE.get((a.reference_name, a.reference_start, a.cigarstring, a.flag))
            if tags is None:
                tags = {}
                if i % 13 != 12:
                    tags["CB"] = f"c{i % 3}"
                if i % 7 != 6:
                    tags["UB"] = f"u{i % 2}"
                if i % 11 == 10:
                    tags["pa"] = 20
            a.set_tags(list(tags.items()))
            fout.write(a)
    pysam.index(str(out))
    return out


def _stage06_reference(bam_path, ref, by_k3, seg, exons):
    """scripts/06_molecules.py::_work as of e534d16 (milestone 1), with its globals passed in."""
    rows, coll = [], defaultdict(int)
    with pysam.AlignmentFile(str(bam_path)) as bam:
        for a in bam.fetch(ref):
            if a.is_secondary or a.is_supplementary or a.has_tag("pa") or not a.has_tag("CB") or not a.has_tag("UB"):
                continue
            hits = by_k3.get((a.get_tag("CB"), a.get_tag("UB"), "-" if a.is_reverse else "+"))
            if not hits:
                continue
            for mol, mref, site, sgenes in hits:
                if mref != ref:
                    coll[mol] += 1
                    continue
                d5, d3 = member_distances(a, site)
                e5, e3, mode = member_distance(a.get_blocks(), a.is_reverse, site, sgenes, exons, (d5, d3))
                mg = genes_at(seg, ref, not a.is_reverse, last_aligned_base(a))
                rows.append([mol, a.query_name, d5, d3, "N" in a.cigarstring, a.is_duplicate, mg, gene_match(sgenes, mg), e5, e3, mode])
    return rows, dict(coll)


@pytest.fixture(scope="module")
def annot(synth):
    return load_gene_segments(synth.gtf), load_collapsed_exons(synth.gtf)


def _new(bam_path, ref, seg, exons):
    coll = {}
    with pysam.AlignmentFile(str(bam_path)) as bam:
        rows = member_rows(bam.fetch(ref), ref, BY_K3, seg, exons, coll)
    return rows, coll


def test_member_rows_equal_stage06_logic(tagged, annot):
    seg, exons = annot
    total = 0
    for ref in ("chrA", "chrB"):
        rows, coll = _new(tagged, ref, seg, exons)
        ref_rows, ref_coll = _stage06_reference(tagged, ref, BY_K3, seg, exons)
        assert rows == ref_rows
        assert coll == ref_coll
        total += len(rows)
    assert total > 20
    with pysam.AlignmentFile(str(tagged)) as bam:
        assert member_rows(bam.fetch("chrA"), "chrA", BY_K3, seg, exons) == _stage06_reference(tagged, "chrA", BY_K3, seg, exons)[0]


def test_member_rows_by_hand(tagged, annot):
    seg, exons = annot
    rows, coll = _new(tagged, "chrA", seg, exons)
    with pysam.AlignmentFile(str(tagged)) as bam:
        name = {(a.reference_start, a.cigarstring, a.flag): a.query_name for a in bam.fetch("chrA")}
    df = pd.DataFrame(rows, columns=MEMBER_COLUMNS).set_index("qname")
    # 950 90M fwd, mol 0 (site 3000, g1): d5 3000-950, d3 3000-1040; gene at 1039 is g1; the last cDNA base 2999 is
    # intronic in g1 (exons 1000-1400, 1800-2700, 3000-4000), so the genomic pair is kept
    assert df.loc[name[(950, "90M", 0)]].tolist() == [0, 2050, 1960, False, False, "g1", True, 2050, 1960, "genome"]
    # spliced 1340 60M400N30M, mol 2 (site 2600): blocks 1340-1400 and 1800-1830 and 2599 are exonic;
    # exon distances 60 + 800 = 860 and 2600 - 1830 = 770; genomic 1260 and 770
    assert df.loc[name[(1340, "60M400N30M", 0)]].tolist() == [2, 1260, 770, True, False, "g1", True, 860, 770, "exon"]
    # reverse 4396-4486, mol 3 (site 4300, g2 exon 4200-4700): d5 4486-4300, d3 4396-4300, both exonic
    assert df.loc[name[(4396, "90M", 16)]].tolist() == [3, 186, 96, False, False, "g2", True, 186, 96, "exon"]
    # reverse 5052-5142, mol 4 (site 5000, intergenic label ""): read gene g2 does not match, genomic distances
    assert df.loc[name[(5052, "90M", 16)]].tolist() == [4, 142, 52, False, False, "g2", False, 142, 52, "genome"]
    # duplicate 3400-3490, mol 0: past the site (negative distances), dup kept
    assert df.loc[name[(3400, "90M", 1024)]].tolist() == [0, -400, -490, False, True, "g1", True, -400, -490, "genome"]
    # the pa read, the read without UB and the secondary give no row
    for key in ((1357, "90M", 0), (1394, "90M", 0), (1250, "90M", 256)):
        assert name[key] not in df.index
    # mol 1 sits on chrB: c0/u0 forward reads on chrA are collisions, not members
    assert 1 not in set(df.mol) and coll[1] == int((df.mol == 0).sum())


def test_stage06_work_uses_member_rows(tagged, annot):
    spec = importlib.util.spec_from_file_location("stage06", REPO / "scripts" / "06_molecules.py")
    stage06 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage06)
    seg, exons = annot
    stage06._BY_K3, stage06._SEG, stage06._EXONS = BY_K3, seg, exons
    for ref in ("chrA", "chrB"):
        df, coll = stage06._work((tagged, ref))
        ref_rows, ref_coll = _stage06_reference(tagged, ref, BY_K3, seg, exons)
        pd.testing.assert_frame_equal(df, pd.DataFrame(ref_rows, columns=MEMBER_COLUMNS))
        assert coll == ref_coll
