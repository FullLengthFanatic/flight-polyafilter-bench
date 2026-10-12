"""The synthetic 10x BAM and outs folder (tests/synth.build_10x) checked against the design it is built from.

Counts are redone here from the BAM with plain dicts, apart from the pandas code of the fixture.
"""
import gzip
from collections import Counter, defaultdict

import h5py
import numpy as np
import pandas as pd
import pysam
import pytest

from pafbench.flags import record_key, setting_tag
from tests.synth import CELLS_10X, SETTINGS_10X, UNCALLED_10X, build_10x, run_original

TAGS = [setting_tag(s) for s in SETTINGS_10X]
GENES = ["g1", "g2", "g3", "g4"]
CELL = {v: k for k, v in CELLS_10X.items()}
# designed read groups per case: {gene: (n_reads, n_winner_reads, n_pa, re_class, end3, end3_nonpa, reads removed at each
# of TAGS)}. end3 is the 3'-most aligned base of the group in transcript sense, 0-based: the last aligned base of the last
# read of g1 and g3 (start + 89, start + 59 for a pa read with its 60 aligned bases), the lowest start of g2 (minus strand);
# end3_nonpa the same over the reads without pa, -1 without any.
GROUPS = {
    "one_inside": {"g1": (1, 1, 0, 1, 1239, 1239, 1, 1, 1, 1)},
    "two_one_inside": {"g1": (2, 2, 0, 1, 2239, 2239, 1, 1, 2, 2)},
    "spliced": {"g1": (3, 3, 0, 1, 3109, 3109, 1, 1, 3, 3)},
    "protected": {"g1": (1, 1, 0, 1, 2594, 2594, 0, 0, 0, 0)},
    "intronic": {"g1": (1, 1, 0, 2, 1539, 1539, 0, 1, 0, 1)},
    "exon_intron": {"g1": (2, 2, 0, 3, 1539, 1539, 0, 1, 1, 2)},
    "pa_only": {"g1": (1, 1, 1, 1, 3659, -1, 0, 0, 1, 1)},
    "pa_and_member": {"g1": (2, 2, 1, 1, 3939, 3789, 0, 0, 1, 1)},
    "three_vs_one": {"g1": (3, 3, 0, 3, 3439, 3439, 2, 3, 2, 3), "g3": (1, 0, 0, 1, 1489, 1489, 0, 0, 0, 0)},
    "leads_by_one": {"g1": (2, 2, 0, 1, 3109, 3109, 1, 1, 2, 2), "g2": (1, 0, 0, 1, 4290, 4290, 0, 0, 0, 0)},
    "tie": {"g1": (2, 0, 0, 1, 3969, 3969, 0, 0, 0, 0), "g2": (2, 0, 0, 1, 4210, 4210, 0, 0, 0, 0)},
    "tie_broken": {"g1": (2, 0, 0, 3, 2594, 2594, 0, 1, 0, 1), "g2": (2, 0, 0, 1, 4210, 4210, 0, 0, 0, 0)},
    "g3_two": {"g3": (2, 2, 0, 1, 989, 989, 0, 0, 2, 2)},
    "three_genes": {"g1": (3, 3, 0, 3, 2594, 2594, 1, 2, 1, 2), "g2": (2, 0, 0, 3, 4250, 4250, 0, 1, 0, 1),
                    "g3": (2, 0, 0, 1, 1489, 1489, 0, 0, 0, 0)},
    "three_vs_two": {"g1": (3, 3, 0, 1, 3969, 3969, 2, 2, 2, 2), "g2": (2, 0, 0, 1, 4210, 4210, 0, 0, 0, 0)},
    "g2_inside": {"g2": (1, 1, 0, 1, 4550, 4550, 1, 1, 1, 1)},
    "alone_xf19": {"g3": (1, 0, 0, 1, 1489, 1489, 0, 0, 0, 0)},
    "alone_xf19_partner": {"g1": (4, 4, 0, 1, 3979, 3979, 0, 0, 0, 0)},
    "g2_partial": {"g2": (2, 2, 0, 1, 5350, 5350, 1, 1, 2, 2)},
    "winner_keeps": {"g1": (3, 3, 0, 1, 3899, 3899, 1, 1, 1, 1), "g3": (1, 0, 0, 1, 1489, 1489, 0, 0, 0, 0)},
    "chain": {"g1": (1, 1, 0, 1, 3889, 3889, 0, 0, 0, 0), "g2": (1, 0, 0, 1, 4250, 4250, 0, 0, 0, 0),
              "g3": (1, 0, 0, 1, 1489, 1489, 0, 0, 0, 0)},
    "chain_end": {"g1": (4, 4, 0, 1, 3994, 3994, 0, 0, 0, 0)},
    # the pa read is the 3'-most of its group (lowest start on the minus strand), the member ends 80 nt upstream of it
    "minus_pa": {"g2": (2, 2, 1, 1, 4210, 4290, 0, 0, 0, 0)},
}
# the matrix Cell Ranger would write, from the designed xf bits
MATRIX = {("c1", "g1"): 8, ("c2", "g1"): 2, ("c2", "g3"): 1, ("c3", "g1"): 2, ("c3", "g2"): 1, ("c4", "g1"): 2, ("c4", "g2"): 1,
          ("c5", "g1"): 2, ("c5", "g2"): 1}
# record counters of design 3.1 per called cell, RE and presence of a GX tag: (records, records with a UB, records removed
# at each of TAGS), written by hand from the reads of the fixture
CELL_RECORDS = {
    ("c1", "E", True): (14, 13, 6, 6, 12, 12), ("c1", "N", True): (2, 2, 0, 2, 0, 2), ("c1", "E", False): (1, 1, 0, 0, 0, 0),
    # two antisense and two sense records, one of the sense ones without a UB; -i removes the sense ones
    ("c1", "N", False): (3, 2, 0, 2, 0, 2), ("c1", "I", False): (1, 1, 0, 0, 0, 0),
    ("c2", "E", True): (15, 15, 3, 3, 6, 6), ("c2", "N", True): (2, 2, 0, 2, 0, 2), ("c2", "I", False): (1, 1, 0, 0, 0, 0),
    ("c3", "E", True): (11, 11, 4, 4, 4, 4), ("c3", "N", True): (2, 2, 0, 2, 0, 2),
    ("c4", "E", True): (11, 11, 2, 2, 3, 3),
    ("c5", "E", True): (9, 9, 0, 0, 0, 0),
}


def _records(bam):
    out = []
    with pysam.AlignmentFile(str(bam)) as fh:
        for a in fh.fetch(until_eof=True):
            out.append({"qname": a.query_name, "flag": a.flag, "ref": a.reference_name, "pos": a.reference_start,
                        "end": a.reference_end, "cigar": a.cigarstring, "qlen": a.query_length, "mapq": a.mapping_quality,
                        "key": record_key(a),
                        "types": {t: ty for t, _, ty in a.get_tags(with_value_type=True)}} | dict(a.get_tags()))
    return out


@pytest.fixture(scope="module")
def recs(synth10x):
    return _records(synth10x.bam)


def _in_group(r):
    """The read group rule of design 3.1: called cell, UB, single-gene GX of the feature list, xf bit 1."""
    return r.get("CB") in CELL and "UB" in r and r.get("GX") in GENES and r["xf"] & 1 and not r["flag"] & 0x900


def _rule(groups):
    """Documented rule on {(cell, umi): {gene: reads}}: the gene with strictly the most reads, a tie counts nothing."""
    out = {}
    for unit, by_gene in groups.items():
        alive = {g: n for g, n in by_gene.items() if n > 0}
        top = max(alive.values(), default=0)
        winners = [g for g, n in alive.items() if n == top]
        out[unit] = winners[0] if len(winners) == 1 else None
    return out


def _outs(s):
    d = s.outs / "filtered_feature_bc_matrix"
    with gzip.open(d / "barcodes.tsv.gz", "rt") as fh:
        barcodes = fh.read().split()
    with gzip.open(d / "features.tsv.gz", "rt") as fh:
        features = [line.rstrip("\n").split("\t") for line in fh]
    with gzip.open(d / "matrix.mtx.gz", "rt") as fh:
        lines = fh.read().splitlines()
    return barcodes, features, lines


def test_same_genome_and_annotation_as_build(synth, synth10x):
    assert synth10x.fasta.read_text() == synth.fasta.read_text()
    assert synth10x.gtf.read_text() == synth.gtf.read_text()


def test_bam_is_coordinate_sorted_and_indexed_with_cell_ranger_comment_lines(synth10x, recs):
    with pysam.AlignmentFile(str(synth10x.bam)) as fh:
        hdr = fh.header.to_dict()
        tid = {r: fh.get_tid(r) for r in fh.references}
        fetched = sum(fh.count(r) for r in fh.references)
    assert synth10x.bam == synth10x.outs / "possorted_genome_bam.bam" and synth10x.bam.with_suffix(".bam.bai").exists()
    assert hdr["HD"]["SO"] == "coordinate"
    assert hdr["CO"] == ["10x_bam_to_fastq:R1(CR:CY,UR:UY)", "10x_bam_to_fastq:R2(SEQ:QUAL)",
                         'library_info:{"library_id":0,"library_type":"Gene Expression","gem_group":1,"target_set_name":null}']
    order = [(tid[r["ref"]], r["pos"]) for r in recs]
    assert order == sorted(order)
    assert fetched == len(recs) == len(synth10x.reads) and {r["ref"] for r in recs} == {"chrA", "chrB"}
    assert len({r["qname"] for r in recs}) == len(recs) and not any(r["flag"] & 0x900 for r in recs)


def _opt(v):
    return None if pd.isna(v) else v


def test_records_carry_the_designed_tags(synth10x, recs):
    want = synth10x.reads.set_index("qname")
    for r in recs:
        w = want.loc[r["qname"]]
        assert (r.get("CB"), r.get("UB"), r.get("GX")) == (_opt(w.cb), _opt(w.ub), _opt(w.gx)), r["qname"]
        assert (r["UR"], r["RE"], r["xf"], r.get("pa", 0)) == (w.ur, w.re, w.xf, w.pa), r["qname"]
        assert (r["ref"], r["pos"], r["end"], bool(r["flag"] & 16)) == (w.ref, w.pos, w.end, w.reverse), r["qname"]
        assert r.get("pa", 1) > 0        # our observation on 1417, chr22:20 to 24 Mb: all 210 pa tags are above 0
        assert ("GN" in r) == ("GX" in r) and r.get("GN", "") == r.get("GX", "").upper()
        assert len(r["UR"]) == len(r["UY"]) == 12 and len(r["CR"]) == len(r["CY"]) == 16
        assert r.get("CB", r["CR"] + "-1") == r["CR"] + "-1"
        valid = min(ord(q) - 33 for q in r["UY"]) >= 10              # UMI_MIN_QV of the Cell Ranger source, design 2.3
        assert ("UB" in r) == valid
        for t, ty in r["types"].items():
            assert ty == {"RE": "A", "xf": "i", "pa": "i"}.get(t, "Z"), (r["qname"], t)
    assert set(synth10x.reads.xf) == {0, 17, 19, 25}


def test_every_designed_case_is_present(synth10x, recs):
    case = synth10x.reads.set_index("qname").case
    by_case = defaultdict(list)
    for r in recs:
        by_case[case[r["qname"]]].append(r)
    (multi,) = by_case["multi_gene"]
    assert multi["GX"] == "g1;g2" and multi["GN"] == "G1;G2" and multi["xf"] == 0 and multi["mapq"] == 255 and "UB" in multi
    (mm,) = by_case["multimapped"]
    assert mm["GX"] == "g1" and mm["xf"] == 0 and mm["mapq"] < 255 and mm["CB"] in CELL and "UB" in mm
    (no_ub,) = by_case["no_ub"]
    assert "UB" not in no_ub and no_ub["xf"] == 17 and no_ub["GX"] == "g1" and not no_ub["flag"] & 1024
    assert (no_ub["CB"], no_ub["UR"]) in {(r["CB"], r["UR"]) for r in by_case["two_one_inside"]}
    assert {r["CB"] for r in by_case["uncalled"]} == {UNCALLED_10X} and UNCALLED_10X not in CELL
    assert {r["xf"] for r in by_case["uncalled"]} == {17, 25} and {r["GX"] for r in by_case["uncalled"]} == {"g1", "g4"}
    (no_cb,) = by_case["no_cb"]
    assert "CB" not in no_cb and no_cb["xf"] == 17 and no_cb["UB"] == no_cb["UR"] and not no_cb["flag"] & 1024
    # AN names transcripts with offset and CIGAR on an exonic record and the gene on an intronic one
    assert {r["RE"]: r["AN"] for r in by_case["antisense"]} == {"E": "t1,-200,90M;t2,-200,90M", "N": "g1,-"}
    for r in by_case["antisense"]:
        assert "GX" not in r and r["xf"] == 0 and r["flag"] & 16
    # pa reads are whole reads with the trimmed 3' end soft-clipped: exactly pa bases, or more
    assert sorted((case[r["qname"]], r["cigar"], r["pa"], r["qlen"]) for r in recs if "pa" in r) == [
        ("minus_pa", "30S60M", 28, 90), ("pa_and_member", "60M30S", 30, 90), ("pa_only", "60M30S", 30, 90)]
    assert all(r["flag"] & 16 for r in by_case["minus_pa"]) and not any(r["flag"] & 16 for r in by_case["pa_only"])
    assert len(by_case["intergenic"]) == 2
    for r in by_case["intergenic"]:
        assert r["RE"] == "I" and "GX" not in r and "AN" not in r and r["xf"] == 0
    # sense strand, intronic, outside every read group: the kind of record that carries the intronic removal of 1179
    with_ub, without_ub = sorted(by_case["intronic_no_gx"], key=lambda r: "UB" not in r)
    for r in (with_ub, without_ub):
        assert r["RE"] == "N" and "GX" not in r and "AN" not in r and r["xf"] == 0 and r["CB"] in CELL and not r["flag"] & 16
    assert "UB" in with_ub and "UB" not in without_ub
    # a counted read that is antisense to another gene keeps its GX next to the AN
    (intronic,) = by_case["intronic"]
    assert (intronic["RE"], intronic["GX"], intronic["AN"], intronic["xf"]) == ("N", "g1", "g2,-", 25)
    spliced = by_case["spliced"]
    assert sum("N" in r["cigar"] for r in spliced) == 1 and max(r["pos"] for r in spliced) - min(r["pos"] for r in spliced) >= 400
    assert {r["ref"] for r in by_case["three_vs_one"]} == {r["ref"] for r in by_case["winner_keeps"]} == {"chrA", "chrB"}
    for name in ("leads_by_one", "tie", "tie_broken", "three_vs_two"):
        assert {r["ref"] for r in by_case[name]} == {"chrA"}, name
    # the two patterns of the Cell Ranger source: raw UMIs that differ from the corrected one
    corrected = {name: sorted((r["GX"], r["UR"] != r["UB"], r["xf"]) for r in by_case[name])
                 for name in ("alone_xf19_partner", "alone_xf19", "chain", "chain_end")}
    assert corrected == {
        "alone_xf19_partner": [("g1", False, 17), ("g1", False, 17), ("g1", True, 17), ("g1", True, 25)],
        "alone_xf19": [("g3", False, 19)],
        "chain": [("g1", True, 25), ("g2", False, 19), ("g3", False, 19)],
        "chain_end": [("g1", False, 17), ("g1", False, 17), ("g1", True, 17), ("g1", True, 25)],
    }
    assert sum(r["UR"] != r.get("UB", r["UR"]) for r in recs) == 5
    assert set(by_case) == set(GROUPS) | {"multi_gene", "multimapped", "no_ub", "uncalled", "no_cb", "antisense", "intergenic",
                                              "intronic_no_gx"}


def test_record_counters_per_cell_region_and_gx_are_the_designed_ones(synth10x, recs):
    removed = synth10x.reads.set_index("qname")[TAGS]
    naive = {}
    for r in recs:
        if r.get("CB") in CELL:
            n = naive.setdefault((CELL[r["CB"]], r["RE"], "GX" in r), [0] * (2 + len(TAGS)))
            n[0] += 1
            n[1] += "UB" in r
            n[2:] = [k + int(x) for k, x in zip(n[2:], removed.loc[r["qname"]])]
    assert {k: tuple(n) for k, n in naive.items()} == CELL_RECORDS
    # what a stage that drops removals of records without GX, or takes AN for "no GX", would get wrong
    no_gx = [n for (_, _, gx), n in CELL_RECORDS.items() if not gx]
    assert sum(n[0] for n in no_gx) > sum(n[1] for n in no_gx) and any(any(n[2:]) for n in no_gx)
    assert sum("AN" in r and "GX" in r and r["xf"] & 1 for r in recs) == 1


def test_groups_table_is_the_designed_one_and_equals_a_naive_count_of_the_bam(synth10x, recs):
    removed = synth10x.reads.set_index("qname")[TAGS]
    naive = {}
    for r in recs:
        if _in_group(r):
            g = naive.setdefault((r["CB"], r["UB"], r["GX"]),
                                 {"n": 0, "win": 0, "pa": 0, "re": set(), "last": [], "last_nonpa": [], "rm": [0] * len(TAGS)})
            g["n"] += 1
            g["win"] += not r["xf"] & 2
            g["pa"] += r.get("pa", 0) > 0
            g["re"].add(r["RE"])
            g["most"] = min if r["flag"] & 16 else max       # the 3'-most base is the lowest one on the minus strand
            g["last"].append(r["pos"] if r["flag"] & 16 else r["end"] - 1)
            g["last_nonpa"] += [] if "pa" in r else g["last"][-1:]
            g["rm"] = [k + int(x) for k, x in zip(g["rm"], removed.loc[r["qname"]])]
    naive = {k: (g["n"], g["win"], g["pa"], ("E" in g["re"]) + 2 * ("N" in g["re"]), g["most"](g["last"]),
                 g["most"](g["last_nonpa"], default=-1), *g["rm"]) for k, g in naive.items()}
    t = synth10x.groups
    columns = ["case", "cb", "ub", "gene", "n_reads", "n_winner_reads", "n_pa", "re_class", "end3", "end3_nonpa"]
    assert list(t.columns) == columns + [f"rm_{x}" for x in TAGS]
    table = {(r.cb, r.ub, r.gene): tuple(int(x) for x in r[4:]) for r in t.itertuples(index=False)}
    assert table == naive and len(t) == len(naive) == 33
    designed = {}
    for case, g in t.groupby("case"):
        assert g.cb.nunique() == 1 and g.ub.nunique() == 1, case                       # a case is one cell and one UMI
        designed[case] = {r.gene: tuple(int(x) for x in r[4:]) for r in g.itertuples(index=False)}
    assert designed == GROUPS
    assert t.groupby(["cb", "ub"]).case.nunique().max() == 1


@pytest.mark.parametrize("st", SETTINGS_10X, ids=setting_tag)
def test_removed_reads_are_those_the_original_tool_removes(synth10x, recs, st):
    want = run_original(synth10x, st.covlen, st.min_len, st.mism, st.introns)
    tag = setting_tag(st)
    key = {r["qname"]: r["key"] for r in recs}
    assert {key[q] for q in synth10x.reads.qname[synth10x.reads[tag]]} == want
    assert 0 < len(want) < len(recs)


def test_controls_remove_what_their_settings_remove(synth10x):
    r = synth10x.reads
    assert [(s.min_len is None, s.introns) for s in SETTINGS_10X] == [(False, False), (False, True), (True, False), (True, True)]
    assert len({s.covlen for s in SETTINGS_10X}) == 1
    assert (r[TAGS[0]] <= r[TAGS[2]]).all() and (r[TAGS[1]] <= r[TAGS[3]]).all()
    assert (r[TAGS[0]] != r[TAGS[1]]).any() and (r[TAGS[0]] != r[TAGS[2]]).any()


def test_matrix_equals_an_independent_count_from_the_xf_bits(synth10x, recs):
    barcodes, features, lines = _outs(synth10x)
    assert barcodes == sorted(CELL) and [f[0] for f in features] == GENES
    assert all(f == [g, g.upper(), "Gene Expression"] for f, g in zip(features, GENES))
    assert lines[0] == "%%MatrixMarket matrix coordinate integer general" and lines[1].startswith("%metadata_json:")
    entries = [tuple(int(x) for x in line.split()) for line in lines[3:]]
    assert lines[2].split() == [str(len(GENES)), str(len(barcodes)), str(len(entries))]
    assert [(e[1], e[0]) for e in entries] == sorted((e[1], e[0]) for e in entries)       # by barcode, then feature
    matrix = {(barcodes[c - 1], features[f - 1][0]): n for f, c, n in entries}
    counted = {(r["CB"], r["UB"], r["GX"]) for r in recs if r.get("CB") in CELL and "UB" in r and r["xf"] & 1 and not r["xf"] & 2}
    assert matrix == Counter((cb, gx) for cb, _, gx in counted)
    assert matrix == {(CELLS_10X[c], g): n for (c, g), n in MATRIX.items()}


def test_molecule_info_rows_equal_the_counted_groups(synth10x, recs):
    with h5py.File(synth10x.outs / "molecule_info.h5", "r") as f:
        assert set(f) == {"barcodes", "barcode_idx", "feature_idx", "umi", "count", "gem_group", "barcode_info", "features"}
        cols = {k: f[k][:] for k in ("barcode_idx", "feature_idx", "umi", "count", "gem_group")}
        bar, fid, pf = f["barcodes"][:].astype(str), f["features/id"][:].astype(str), f["barcode_info/pass_filter"][:]
    # dtypes of the Cell Ranger 9.0.1 file (our observation, h5py on 1417's molecule_info.h5)
    assert {k: str(v.dtype) for k, v in cols.items()} == {"barcode_idx": "uint64", "feature_idx": "uint32", "umi": "uint32",
                                                         "count": "uint32", "gem_group": "uint16"}
    assert pf.dtype == np.uint64 and pf.shape == (len(CELL), 3) and not pf[:, 1:].any()
    assert list(bar) == sorted(c[:-2] for c in list(CELL) + [UNCALLED_10X]) and list(fid) == GENES
    assert [bar[i] + "-1" for i in pf[:, 0]] == sorted(CELL)
    # the uncalled barcode sorts second, so a barcode index is not a row of barcodes.tsv.gz
    assert pf[:, 0].tolist() == [0, 2, 3, 4, 5]
    order = list(zip(cols["barcode_idx"].tolist(), cols["feature_idx"].tolist(), cols["umi"].tolist()))
    assert order == sorted(order) and len(set(order)) == len(order)
    support = Counter((r["CB"], r["GX"], r["UB"]) for r in recs if "CB" in r and "UB" in r and r["xf"] in (17, 25))
    code = lambda umi: sum("ACGT".index(b) << 2 * i for i, b in enumerate(reversed(umi)))
    rows = {(f"{bar[b]}-{g}", fid[x], int(u)): int(n)
            for b, x, u, n, g in zip(*(cols[k] for k in ("barcode_idx", "feature_idx", "umi", "count", "gem_group")))}
    assert rows == {(cb, gx, code(ub)): n for (cb, gx, ub), n in support.items()}
    called = Counter((cb, gx) for cb, gx, _ in rows if cb in CELL)
    assert called == {(CELLS_10X[c], g): n for (c, g), n in MATRIX.items()}
    assert {gx for cb, gx, _ in rows if cb == UNCALLED_10X} == {"g1", "g4"}
    # check A2 of the design: the groups with a winner read are the rows of the called cells, and n_reads is count
    winners = synth10x.groups[synth10x.groups.n_winner_reads > 0]
    assert {(r.cb, r.gene, code(r.ub)): r.n_reads for r in winners.itertuples()} == {k: n for k, n in rows.items()
                                                                                     if k[0] in CELL}


def test_duplicate_flag_sits_exactly_on_xf17_reads_with_ub(recs):
    for r in recs:
        if "CB" in r:
            assert bool(r["flag"] & 1024) == (r["xf"] == 17 and "UB" in r), r["qname"]
    per_group = defaultdict(list)
    for r in recs:
        if "CB" in r and "UB" in r and r["xf"] & 1:
            per_group[(r["CB"], r["UB"], r["GX"])].append(r["xf"])
    for unit, xfs in per_group.items():          # a discarded group, or one representative read and its duplicates
        assert set(xfs) == {19} or sorted(xfs) == [17] * (len(xfs) - 1) + [25], unit


def test_documented_rule_differs_from_the_matrix_in_the_two_source_pattern_entries(synth10x, recs):
    groups = defaultdict(Counter)
    for r in recs:
        if _in_group(r):
            groups[(r["CB"], r["UB"])][r["GX"]] += 1
    rule = Counter((cb, g) for (cb, _), g in _rule(groups).items() if g is not None)
    designed = Counter({(CELLS_10X[c], g): n for (c, g), n in MATRIX.items()})
    assert {k: (designed[k], rule[k]) for k in set(rule) | set(designed) if rule[k] != designed[k]} == {
        (CELLS_10X["c4"], "g3"): (0, 1),      # alone at its corrected UMI, xf 19
        (CELLS_10X["c5"], "g1"): (2, 1),      # three groups of one read, one with xf 25
    }


def test_designed_events_equal_the_rule_on_the_reads_that_survive(synth10x, recs):
    removed = synth10x.reads.set_index("qname")[TAGS]
    ev = synth10x.events
    assert list(ev.columns) == ["case", "cb", "ub", "tag", "before", "after", "lost", "gained", "switched"]
    assert len(ev) == len(TAGS) * synth10x.groups.groupby(["cb", "ub"]).ngroups == len(TAGS) * len(GROUPS)
    n0 = defaultdict(Counter)
    for r in recs:
        if _in_group(r):
            n0[(r["CB"], r["UB"])][r["GX"]] += 1
    before = _rule(n0)
    for tag in TAGS:
        nt = defaultdict(Counter)
        for r in recs:
            if _in_group(r):
                nt[(r["CB"], r["UB"])][r["GX"]] += not removed.loc[r["qname"], tag]
        after = _rule(nt)
        e = ev[ev.tag == tag]
        designed = {(r.cb, r.ub): (_opt(r.before), _opt(r.after)) for r in e.itertuples()}
        assert designed == {u: (before[u], after[u]) for u in before}, tag
    b, a = ev.before.notna(), ev.after.notna()
    differ = ev.before.fillna("") != ev.after.fillna("")
    assert (ev.lost == (b & differ)).all() and (ev.gained == (a & differ)).all() and (ev.switched == (b & a & differ)).all()
    got = {(r.case, TAGS.index(r.tag)): (r.lost, r.gained, r.switched) for r in ev.itertuples() if r.lost or r.gained}
    lost, gained, switched = (True, False, False), (False, True, False), (True, True, True)
    assert got == {
        ("one_inside", 0): lost, ("one_inside", 1): lost, ("one_inside", 2): lost, ("one_inside", 3): lost,
        ("two_one_inside", 2): lost, ("two_one_inside", 3): lost, ("spliced", 2): lost, ("spliced", 3): lost,
        ("intronic", 1): lost, ("intronic", 3): lost, ("exon_intron", 3): lost, ("pa_only", 2): lost, ("pa_only", 3): lost,
        # winner 3 against loser 1 on another reference: a tie in default mode, a switch with -i
        ("three_vs_one", 0): lost, ("three_vs_one", 1): switched, ("three_vs_one", 2): lost, ("three_vs_one", 3): switched,
        # winner that leads by one read: a tie while a read remains, a switch at the controls
        ("leads_by_one", 0): lost, ("leads_by_one", 1): lost, ("leads_by_one", 2): switched, ("leads_by_one", 3): switched,
        ("tie_broken", 1): gained, ("tie_broken", 3): gained,
        ("g3_two", 2): lost, ("g3_two", 3): lost,
        # 3, 2 and 2 reads: a tie of three in default mode, a switch with -i
        ("three_genes", 0): lost, ("three_genes", 1): switched, ("three_genes", 2): lost, ("three_genes", 3): switched,
        ("three_vs_two", 0): switched, ("three_vs_two", 1): switched, ("three_vs_two", 2): switched,
        ("three_vs_two", 3): switched,
        ("g2_inside", 0): lost, ("g2_inside", 1): lost, ("g2_inside", 2): lost, ("g2_inside", 3): lost,
        ("g2_partial", 2): lost, ("g2_partial", 3): lost,
    }
    # the switch of three_vs_two happens while a read remains on the old winner
    old = synth10x.groups[(synth10x.groups.case == "three_vs_two") & (synth10x.groups.gene == "g1")].iloc[0]
    assert all(old.n_reads - old[f"rm_{t}"] == 1 for t in TAGS)
    # with two genes on the cell and UMI, a setting can also take a read of the winner and leave the UMI where it is
    multi = synth10x.groups.groupby(["cb", "ub"]).filter(lambda g: len(g) > 1)
    winner = multi.merge(ev, left_on=["cb", "ub", "gene"], right_on=["cb", "ub", "before"])
    removed_and_kept = [(r.case_x, r.tag) for r in winner.itertuples() if getattr(r, f"rm_{r.tag}") > 0 and r.after == r.before]
    assert removed_and_kept == [("winner_keeps", t) for t in TAGS]
    for name in ("tie", "chain"):            # nothing counted before or after
        assert ev[ev.case == name].before.isna().all() and ev[ev.case == name].after.isna().all()


def _cell_ranger(reads):
    """Port of correct_umis and determine_low_support_umigenes for the reads of one barcode, [(raw UMI, gene)], to
    [(corrected UMI, low support)] (cellranger 9.0.1, lib/rust/tx_annotation/src/mark_dups.rs, lines 19 to 59, 86 to 108
    and 226 to 239: one read of each corrected group is moved before low support is decided)."""
    counts = Counter(reads)
    corr = {}
    for (umi, gene), n in counts.items():
        best, best_n = umi, n
        for pos in range(len(umi)):
            for base in "ACGT":
                test = umi[:pos] + base + umi[pos + 1:]
                k = counts.get((test, gene), 0)
                if base != umi[pos] and (k > best_n or (k == best_n and test > best)):
                    best, best_n = test, k
        if best != umi:
            corr[(umi, gene)] = best
    moved = Counter(counts)
    for (umi, gene), dest in corr.items():
        moved[(umi, gene)] -= 1
        moved[(dest, gene)] += 1
    low = set()
    for umi in {u for u, _ in moved}:
        per_gene = {g: n for (u, g), n in moved.items() if u == umi}
        top = max(per_gene.values())
        tied = sum(n == top for n in per_gene.values()) >= 2
        low |= {(umi, g) for g, n in per_gene.items() if tied or n < top}
    return [(corr.get(r, r[0]), (corr.get(r, r[0]), r[1]) in low) for r in reads]


def test_cell_ranger_code_gives_the_designed_ub_and_low_support_from_the_raw_umis(recs):
    by_cell = defaultdict(list)
    for r in recs:
        if "CB" in r and "UB" in r and r["xf"] & 1:      # valid barcode, valid UMI, confidently mapped to one gene
            by_cell[r["CB"]].append(r)
    assert set(by_cell) == set(CELL) | {UNCALLED_10X}
    for cb, rs in by_cell.items():
        for r, (ub, low) in zip(rs, _cell_ranger([(r["UR"], r["GX"]) for r in rs])):
            assert (r["UB"], bool(r["xf"] & 2)) == (ub, low), r["qname"]


def test_fixture_is_deterministic(synth10x, recs, tmp_path):
    again = build_10x(tmp_path)
    assert _records(again.bam) == recs
    assert _outs(again) == _outs(synth10x)
    with h5py.File(again.outs / "molecule_info.h5", "r") as f1, h5py.File(synth10x.outs / "molecule_info.h5", "r") as f2:
        names = []
        f1.visit(names.append)
        assert all(np.array_equal(f1[n][:], f2[n][:]) for n in names if isinstance(f1[n], h5py.Dataset))
    for name in ("reads", "groups", "events"):
        pd.testing.assert_frame_equal(getattr(again, name), getattr(synth10x, name))
