import numpy as np
import pandas as pd
import pytest

from pafbench.bd67 import accept_long_read, terminal_a_run
from pafbench.lrcontext import (BY_CLASS_COLUMNS, CLASSES, CONTEXT_COLUMNS, READ_COLUMNS, SITE_COLUMNS, SUB_CAUSES, Models,
                                attach_stage16, classify, context_tables, intron_sites_off_model, isoform_ends, link_1179,
                                load_models, pigeon_crosscheck, read_body, reduce_sites, top_class, totals)
from pafbench.sites import priming_site
from tests.synth_long import (ACCEPTOR, AS, DESIGNED, DONOR, G, IG, LOCUS, NEAR, NEITHER, NJ, NOT_JOINED, OPPOSITE, SHORT, ctx,
                              expected, long_read, write_gtf)

STRANDS = pytest.mark.parametrize("minus", [False, True], ids=["plus", "minus"])


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    return load_models(write_gtf(tmp_path_factory.mktemp("lrcontext") / "genes.gtf"))["chrS"]


def test_designed_reads_cover_every_class_and_are_accepted_at_their_site():
    assert {want.klass for _, _, want in DESIGNED.values()} == set(CLASSES)
    for name, (blocks, site, _) in DESIGNED.items():
        for minus in (False, True):
            got, aligned_a, clip_a, after = terminal_a_run(long_read(blocks, site, minus))
            assert got == (G - site if minus else site), (name, minus)
            assert accept_long_read(aligned_a, clip_a, after) and clip_a == 12, (name, minus)
            assert aligned_a == sum(e - max(s, site) for s, e in blocks if e > site), (name, minus)


def test_read_body_cuts_at_the_site_and_keeps_deletions_inside_a_block():
    blocks, site, _ = DESIGNED["full_chain_3UTR"]
    a = long_read(blocks, site)
    assert a.cigarstring == "7S23M2D3I25M100N48M2D3I50M100N48M2D3I50M100N104M2D3I106M45S"
    assert read_body(a, site) == [(1050, 1100), (1200, 1300), (1400, 1500), (1600, 1800)]
    # minus strand: the same body in transcript sense, a genomic interval [a, b) mirrored to [-b, -a)
    a = long_read(blocks, site, minus=True)
    assert a.reference_start == G - 1812 and a.cigarstring == "45S106M3I2D104M100N50M3I2D48M100N50M3I2D48M100N25M3I2D23M7S"
    assert read_body(a, G - site) == [(1050 - G, 1100 - G), (1200 - G, 1300 - G), (1400 - G, 1500 - G), (1600 - G, 1800 - G)]
    # a block that lies wholly in the terminal A run is not part of the body, nor is its junction
    blocks, site, _ = DESIGNED["run_across_junction"]
    for minus in (False, True):
        body = read_body(long_read(blocks, site, minus), G - site if minus else site)
        assert body == ([(1050 - G, 1100 - G), (1200 - G, 1300 - G), (1400 - G, 1490 - G)] if minus
                        else [(1050, 1100), (1200, 1300), (1400, 1490)])
    # the run starts the block after a CIGAR N: that N is the last intron of the body, which ends with an empty block at the site
    blocks, site, _ = DESIGNED["run_after_annotated_junction"]
    for minus in (False, True):
        body = read_body(long_read(blocks, site, minus), G - site if minus else site)
        assert body == ([(1250 - G, 1300 - G), (1400 - G, 1500 - G), (1600 - G, 1600 - G)] if minus
                        else [(1250, 1300), (1400, 1500), (1600, 1600)])
    # every aligned base in the run: no body
    a = long_read([(1050, 1090)], 1050)
    assert terminal_a_run(a)[0] == 1050 and read_body(a, 1050) == []
    a = long_read([(1050, 1090)], 1050, minus=True)
    assert terminal_a_run(a)[0] == G - 1050 and read_body(a, G - 1050) == []


def test_load_models_mirrors_the_minus_strand(models):
    assert [t.tid for t in models.tx] == list(LOCUS) + [t + "m" for t in LOCUS]
    ta, tam = models.tx[0], models.tx[6]
    assert (ta.start, ta.end, ta.cds, ta.start_codon, ta.end_nf) == (1000, 2000, (1050, 1700), 1050, False)
    assert ta.exons == [(1000, 1100), (1200, 1300), (1400, 1500), (1600, 2000)]
    # in transcript sense the mirrored transcript is the plus transcript shifted by -G; its start codon, 0-based 8947..8949
    # on the genome, begins at -8950
    assert (tam.start, tam.end, tam.cds, tam.start_codon) == (1000 - G, 2000 - G, (1050 - G, 1700 - G), 1050 - G)
    assert tam.exons == [(s - G, e - G) for s, e in ta.exons]
    assert models.tx[2].cds is None and [t.end_nf for t in models.tx] == [False] * 5 + [True] + [False] * 5 + [True]
    # TC has no CDS, TL a CDS without a start codon (tagged cds_start_NF)
    assert [t.start_codon for t in models.tx[:6]] == [1050, 1220, None, 3050, None, 3050]
    assert models.tx[4].cds == (3050, 3450) and models.tx[10].start_codon is None
    for minus, shift in ((False, 0), (True, -G)):
        assert sorted(models.introns[minus]) == sorted({(d + shift, a + shift) for d, a in (
            (1100, 1200), (1300, 1400), (1500, 1600), (950, 1200), (1500, 1700), (3200, 3400))})
        assert models.introns[minus][1300 + shift, 1400 + shift] == [(6 * minus, 1), (6 * minus + 1, 1)]
        assert models.donors[minus] == {x + shift for x in (1100, 1300, 1500, 950, 3200)}
        assert models.acceptors[minus] == {x + shift for x in (1200, 1400, 1600, 1700, 3400)}


def test_load_models_restricted_to_references(tmp_path):
    gtf = write_gtf(tmp_path / "genes.gtf")
    assert list(load_models(gtf)) == ["chrS"]
    got = load_models(gtf, refs=["chrE", "chrS"])
    assert list(got) == ["chrE", "chrS"] and got["chrE"].tx == [] and len(got["chrS"].tx) == 12
    assert load_models(gtf, refs=["chrE"])["chrE"].tx == []


@STRANDS
@pytest.mark.parametrize("name", list(DESIGNED))
def test_designed_reads(models, name, minus):
    blocks, site, _ = DESIGNED[name]
    a = long_read(blocks, site, minus)
    assert classify(models, minus, read_body(a, terminal_a_run(a)[0])) == expected(name, minus)


@STRANDS
def test_spans_overlapping_an_interval(models, minus):
    shift, k0 = (-G, 6) if minus else (0, 0)
    spans = lambda a, b: sorted(models.spans(minus, a + shift, b + shift))
    assert spans(1340, 1350) == [k0, k0 + 1] and spans(1340, 1351) == [k0, k0 + 1, k0 + 2]      # TC starts at 1350
    assert spans(2000, 2100) == [k0 + 1] and spans(2100, 2200) == [] and spans(899, 900) == []     # TB is 900..2100
    assert spans(0, 12000) == [k0 + k for k in range(6)]


@STRANDS
def test_suffix_search_agrees_with_full_compatibility_and_ignores_the_decoy(models, minus):
    shift, m = (-G, "m") if minus else (0, "")
    tid = lambda held: [models.tx[k].tid for k in held]
    chain = lambda *introns: [(d + shift, a + shift) for d, a in introns]
    full = chain((1100, 1200), (1300, 1400), (1500, 1600))
    n, held = models.suffix(minus, full)
    assert (n, tid(held)) == (3, ["TA" + m]) and tid(models.chain(minus, full)) == ["TA" + m]
    # the last intron is in TB and TC; only TB also has the one before it, and no transcript has the whole chain
    two = chain((1100, 1200), (1300, 1400), (1500, 1700))
    n, held = models.suffix(minus, two)
    assert (n, tid(held)) == (2, ["TB" + m]) and models.chain(minus, two) == []
    n, held = models.suffix(minus, chain((1100, 1200), (1300, 1420), (1500, 1700)))
    assert (n, tid(held)) == (1, ["TB" + m, "TC" + m])
    assert models.suffix(minus, chain((1100, 1200), (1500, 1650))) == (0, [])
    # a chain longer than the transcript's own: the walk stops at the first intron of TC
    n, held = models.suffix(minus, chain((1300, 1400), (1500, 1700)))
    assert (n, tid(held)) == (2, ["TB" + m])
    # the suffix is matched position by position: the intron 1300..1400 of TA is in the read, but not next to its last one
    n, held = models.suffix(minus, chain((1300, 1400), (1420, 1480), (1500, 1600)))
    assert (n, tid(held)) == (1, ["TA" + m])
    # a run of consecutive introns from the middle or the end of a transcript is compatible, a run with a gap is not
    assert tid(models.chain(minus, chain((1300, 1400)))) == ["TA" + m, "TB" + m]
    assert tid(models.chain(minus, chain((1300, 1400), (1500, 1600)))) == ["TA" + m]
    assert models.chain(minus, chain((1100, 1200), (1500, 1600))) == []
    assert models.chain(minus, chain((1500, 1600), (1700, 1800))) == []       # runs past the last intron of TA


@STRANDS
def test_sub_causes_in_their_order(models, minus):
    shift = -G if minus else 0
    cause = lambda d, a: models.sub_cause(minus, (d + shift, a + shift))
    assert cause(1100, 1400) == NOT_JOINED       # donor of intron 1 and acceptor of intron 2 of TA
    assert cause(1120, 1400) == ACCEPTOR
    assert cause(1500, 1650) == DONOR
    assert cause(1120, 1450) == NEITHER
    assert cause(1500, 1530) == SHORT            # 30 nt, and its donor is annotated
    assert cause(1103, 1198) == NEAR             # 3 and 2 nt from 1100..1200
    assert cause(1105, 1205) == NEAR and cause(1106, 1200) == ACCEPTOR and cause(1100, 1206) == DONOR
    assert cause(1095, 1195) == NEAR and cause(1094, 1200) == ACCEPTOR and cause(1100, 1194) == DONOR
    # an annotated intron of the other strand, where this strand has neither splice site
    assert models.sub_cause(minus, (-1200, -1100) if minus else (G - 1200, G - 1100)) == OPPOSITE


def test_sub_cause_precedence():
    # plus: intron 100..140 (40 nt). minus: intron at genomic 300..330 (30 nt), in transcript sense -330..-300
    m = Models([("P", False, (0, 200), [(0, 100), (140, 200)], [], None, False),
                ("M", True, (-400, -250), [(-400, -330), (-300, -250)], [], None, False)])
    assert m.sub_cause(False, (102, 141)) == NEAR         # also shorter than 50 nt
    assert m.sub_cause(False, (300, 330)) == SHORT        # also the intron of M on the other strand
    assert m.sub_cause(False, (100, 149)) == SHORT        # 49 nt; 9 nt off at the acceptor, so not near
    assert m.sub_cause(False, (100, 150)) == DONOR        # 50 nt is not short
    # the intron of P seen from the minus strand, made long enough: 100..160 on plus is -160..-100 on minus
    m = Models([("P", False, (0, 200), [(0, 100), (160, 200)], [], None, False),
                ("M", True, (-200, 0), [(-200, -160), (-100, 0)], [], None, False)])
    assert m.sub_cause(False, (100, 160)) == NEAR and m.sub_cause(True, (-160, -100)) == NEAR   # annotated on its own strand
    m = Models([("P", False, (0, 200), [(0, 100), (160, 200)], [], None, False)])
    assert m.sub_cause(True, (-160, -100)) == OPPOSITE


def test_choice_falls_back_to_the_transcript_id():
    rows = [(tid, False, (0, 1000), [(0, 400), (600, 1000)], [], None, False) for tid in ("T2", "T1", "T3")]
    assert classify(Models(rows), False, [(100, 200)]) == ctx("ncexon", "T1")
    assert classify(Models(rows[::-1]), False, [(100, 400), (600, 700)]) == ctx("ncexon", "T1", 1)


def test_choice_order_end_then_overlap_then_nearest_end():
    # one exon each; body 100..300, p = 299
    rows = [("far", False, (0, 1000), [(0, 1000)], [], None, False),          # all 200 nt exonic, end 700 nt away
            ("near", False, (0, 305), [(0, 150), (250, 305)], [], None, False)]   # 100 nt exonic, ends within the tolerance
    assert classify(Models(rows), False, [(100, 300)]) == ctx("annotated_end", "near", other_class=True)
    # no end within the tolerance: the larger exonic overlap wins against the nearer end
    rows[1] = ("near", False, (0, 400), [(0, 150), (250, 400)], [], None, False)
    assert classify(Models(rows), False, [(100, 300)]) == ctx("ncexon", "far")
    # equal overlap: the nearer annotated 3' end
    rows[1] = ("near", False, (0, 400), [(0, 400)], [], None, False)
    assert classify(Models(rows), False, [(100, 300)]) == ctx("ncexon", "near")
    # the overlap is counted over every exon a block touches: 50 + 200 nt of "two" against 70 + 50 nt of "wide"
    rows = [("wide", False, (0, 400), [(0, 120), (300, 400)], [], None, False),
            ("two", False, (0, 400), [(0, 100), (150, 400)], [], None, False)]
    assert classify(Models(rows), False, [(50, 350)]) == ctx("ncexon", "two", complete_5p=True)


def test_unspliced_read_longer_than_the_transcript_is_placed_by_its_end_with_the_tolerance():
    m = Models([("T", False, (100, 200), [(100, 200)], [], None, False)])
    # no span contains the read; its last base lies 10 nt past the last base of T, then 11 nt
    assert classify(m, False, [(50, 210)]) == ctx("annotated_end", "T", starts_outside=True, complete_5p=True)
    assert classify(m, False, [(50, 211)]) == ctx("downstream")


def test_end_tolerance_boundaries(models):
    # last base of TA is 1999: 10 nt before or after it is annotated_end, 11 nt is 3UTR or downstream
    klass = lambda body: classify(models, False, body).klass
    chain = [(1250, 1300), (1400, 1500)]
    ends = [klass(chain + [(1600, e)]) for e in (1989, 1990, 2010, 2011)]
    assert ends == ["3UTR", "annotated_end", "annotated_end", "downstream"]
    # unspliced: contained with the tolerance (annotated_end), then past it with the end outside every span (downstream)
    assert [classify(models, False, [(2050, e)]) for e in (2110, 2111)] == [ctx("annotated_end", "TB"), ctx("downstream")]
    # 5' completeness: 50 nt from the annotated 5' end counts, 51 does not
    tail = [(1200, 1300), (1400, 1500), (1600, 1700)]
    assert [classify(models, False, [(s, 1100)] + tail).complete_5p for s in (950, 949, 1050, 1051)] == [True, False, True, False]
    # the start is covered when the first CDS base of TA (1050) lies in an aligned block
    assert [classify(models, False, [(s, 1100)] + tail).covers_start for s in (1050, 1051)] == [True, False]
    assert not classify(models, False, [(1360, 1395)]).covers_start       # TC has no CDS


def test_cds_boundaries_and_the_end_nf_flag_only_at_the_end(models):
    # CDS of TA: first base 1050, last base 1699. TB and TC hold these reads in an intron or with one exonic base
    last = lambda s, p: classify(models, False, [(s, p + 1)])
    assert [last(1010, p).klass for p in (1049, 1050)] == ["5UTR", "CDS"]
    assert [last(1650, p).klass for p in (1699, 1700)] == ["CDS", "3UTR"]
    assert {last(1010, 1050).transcript, last(1650, 1700).transcript} == {"TA"}
    # TS, TL and TN contain the read with 40 exonic nt each; TN has the nearest end and is tagged, but the end is in its CDS
    assert last(3410, 3449) == ctx("CDS", "TN")


def _reads(rows):
    """READ_COLUMNS frame from (minus, site, class name, {other columns}) rows; a column not given is 0, False or no
    sub-cause, the length 1000."""
    out = []
    for minus, site, klass, other in rows:
        row = dict.fromkeys(READ_COLUMNS, 0) | {"minus": minus, "site": site, "klass": CLASSES.index(klass), "length": 1000}
        row |= other | {"sub_cause": SUB_CAUSES.index(other["sub_cause"]) if "sub_cause" in other else -1}
        out.append(row)
    return pd.DataFrame(out, columns=READ_COLUMNS).astype({c: bool for c in READ_COLUMNS[5:11] + ["minus", "no_splice"]})


# seven sites: (+,100) CDS by a majority that a chain_novel read makes; (+,200) novel junction with two sub-causes tied;
# (+,300) the three reasons tied; (+,400) antisense against intergenic; (+,500) a chain_novel read tied with a reason;
# (-,50) a class tied with a reason; (-,100) one chain_novel read
REDUCE_READS = [
    (False, 100, "CDS", dict(matched=3, complete_5p=True, covers_start=True)),
    (False, 100, "CDS", dict(matched=1, chain_novel=1, other_class=True, length=2000)),
    (False, 100, "3UTR", dict(starts_outside=True, no_splice=True, length=500)),
    (False, 200, NJ, dict(sub_cause=NEITHER, length=300)),
    (False, 200, NJ, dict(sub_cause=DONOR, length=400)),
    (False, 200, "CDS", dict(matched=2, chain_novel=2, retained_after_junction=True)),
    (True, 100, "3UTR", dict(matched=3, chain_novel=3)),
    (True, 50, AS, dict(no_splice=True)),
    (True, 50, "annotated_end", dict(matched=2, end_nf=True, complete_5p=True, covers_start=True, other_class=True)),
    (False, 300, IG, {}), (False, 300, AS, {}), (False, 300, NJ, dict(sub_cause=NEAR)),
    (False, 400, IG, {}), (False, 400, AS, {}),
    (False, 500, NJ, dict(sub_cause=SHORT)), (False, 500, "intron_after_junction", dict(matched=1, chain_novel=1)),
]


def test_top_class_takes_the_first_of_tied_classes():
    # the four sites of the prototype's self-test, and one where a class leads
    m = np.zeros((5, len(CLASSES)), int)
    m[0, [1, 2]] = 2                    # 3UTR and CDS
    m[1, [6, 8]] = 1                    # intron_after_junction and novel junction: the reason does not win the tie
    m[2, 8], m[2, 1] = 3, 2
    m[3, [8, 9]] = 4                    # novel junction and antisense
    m[4, 7], m[4, 0], m[4, 10] = 5, 2, 3
    top, tied, share = top_class(m)
    assert top.tolist() == ["3UTR", "intron_after_junction", NJ, NJ, "downstream"]
    assert tied.tolist() == [True, True, False, True, False] and share.tolist() == [0.5, 0.5, 0.6, 0.5, 0.5]


def test_reduce_sites_gives_both_versions_per_site():
    sites, by_class = reduce_sites(_reads(REDUCE_READS), "chrS")
    assert list(sites.columns) == SITE_COLUMNS and list(by_class.columns) == BY_CLASS_COLUMNS
    assert (sites.ref == "chrS").all() and (by_class.ref == "chrS").all()
    assert list(zip(sites.strand, sites.site)) == [("+", 100), ("+", 200), ("+", 300), ("+", 400), ("+", 500), ("-", 50),
                                                   ("-", 100)]
    assert sites.n_accepted.tolist() == [3, 3, 3, 2, 2, 2, 1]
    counts = sites[[f"n_{c}" for c in CLASSES]]
    assert (counts.sum(axis=1) == sites.n_accepted).all()
    want = {("n_CDS", 0): 2, ("n_3UTR", 0): 1, ("n_CDS", 1): 1, (f"n_{NJ}", 1): 2, (f"n_{NJ}", 2): 1, (f"n_{AS}", 2): 1,
            (f"n_{IG}", 2): 1, (f"n_{AS}", 3): 1, (f"n_{IG}", 3): 1, (f"n_{NJ}", 4): 1, ("n_intron_after_junction", 4): 1,
            ("n_annotated_end", 5): 1, (f"n_{AS}", 5): 1, ("n_3UTR", 6): 1}
    assert {(c, i): int(v) for c in counts for i, v in enumerate(counts[c]) if v} == want
    assert sites.n_starts_outside.tolist() == [1, 0, 0, 0, 0, 0, 0]
    assert sites.n_retained_after_junction.tolist() == [0, 1, 0, 0, 0, 0, 0]
    assert sites.n_chain_novel.tolist() == [1, 1, 0, 0, 1, 0, 1] and sites.n_chain_novel_suffix1.tolist() == [1, 0, 0, 0, 1, 0, 0]
    assert sites.n_end_NF.tolist() == [0, 0, 0, 0, 0, 1, 0] and sites.n_no_splice.tolist() == [1, 0, 0, 0, 0, 1, 0]
    # placed: the chain_novel reads count under their class; a tie goes to the first class of CLASSES
    assert sites.context.tolist() == ["CDS", NJ, NJ, AS, "intron_after_junction", "annotated_end", "3UTR"]
    assert sites.tied.tolist() == [False, False, True, True, True, True, False]
    assert sites.top_share.tolist() == [2 / 3, 2 / 3, 1 / 3, 0.5, 0.5, 0.5, 1.0]
    # strict: they count as novel junction, which takes (+,100) to a tie of three that 3UTR wins
    assert sites.context_without_chain_novel.tolist() == ["3UTR", NJ, NJ, AS, NJ, "annotated_end", NJ]
    assert sites.tied_without_chain_novel.tolist() == [True, False, True, True, False, True, False]
    assert sites.top_share_without_chain_novel.tolist() == [1 / 3, 1.0, 1 / 3, 0.5, 1.0, 0.5, 1.0]
    # the top class of the placed version holds chain_novel reads only
    assert sites.chain_novel_only.tolist() == [False, False, False, False, True, False, True]
    # main sub-cause where the placed top class is novel junction; donor_only and neither_site are tied at (+,200)
    assert sites.sub_cause.tolist() == ["", DONOR, NEAR, "", "", "", ""]
    assert sites.share_spliced.tolist() == [2 / 3, 1 / 3, 0, 0, 0.5, 0.5, 1.0]
    assert sites.share_5p_complete.tolist() == [1 / 3, 0, 0, 0, 0, 0.5, 0]
    assert sites.median_length.tolist() == [1000, 400, 1000, 1000, 1000, 1000, 1000]
    rows = by_class.drop(columns="ref").itertuples(index=False, name=None)
    # strand, site, klass, chain_novel, sub_cause, reads, spliced, complete_5p, covers_start, other_class
    assert list(rows) == [
        ("+", 100, "3UTR", False, "", 1, 0, 0, 0, 0), ("+", 100, "CDS", False, "", 1, 1, 1, 1, 0),
        ("+", 100, "CDS", True, "", 1, 1, 0, 0, 1),
        ("+", 200, "CDS", True, "", 1, 1, 0, 0, 0), ("+", 200, NJ, False, DONOR, 1, 0, 0, 0, 0),
        ("+", 200, NJ, False, NEITHER, 1, 0, 0, 0, 0),
        ("+", 300, NJ, False, NEAR, 1, 0, 0, 0, 0), ("+", 300, AS, False, "", 1, 0, 0, 0, 0),
        ("+", 300, IG, False, "", 1, 0, 0, 0, 0),
        ("+", 400, AS, False, "", 1, 0, 0, 0, 0), ("+", 400, IG, False, "", 1, 0, 0, 0, 0),
        ("+", 500, "intron_after_junction", True, "", 1, 1, 0, 0, 0), ("+", 500, NJ, False, SHORT, 1, 0, 0, 0, 0),
        ("-", 50, "annotated_end", False, "", 1, 1, 1, 1, 1), ("-", 50, AS, False, "", 1, 0, 0, 0, 0),
        ("-", 100, "3UTR", True, "", 1, 1, 0, 0, 0)]


def test_reduce_sites_without_reads():
    sites, by_class = reduce_sites(_reads([]), "chrE")
    assert list(sites.columns) == SITE_COLUMNS and list(by_class.columns) == BY_CLASS_COLUMNS
    assert not len(sites) and not len(by_class)


def _stage16(sites, label, region):
    """The stage 16 table for these sites, with one more site that has no accepted read."""
    s = sites[["ref", "strand", "site"]].assign(n_reads=sites.n_accepted + 1, n_accepted=sites.n_accepted, a20=0, label=label,
                                                a_band="0-4", hexamer=[True, False] * 3 + [True], region=region, gene="")
    return pd.concat([s, s.iloc[:1].assign(site=7, n_reads=1, n_accepted=0)], ignore_index=True).sample(frac=1, random_state=0)


LABELS = ["templated", "templated", "nontemplated", "templated", "nontemplated", "templated", "templated"]
REGION = ["CDS", "intron", "antisense", "intergenic", "intron", "3UTR", "3UTR"]


def test_attach_stage16_adds_the_site_columns_and_checks_the_acceptance():
    sites, _ = reduce_sites(_reads(REDUCE_READS), "chrS")
    s16 = _stage16(sites, LABELS, REGION)
    got = attach_stage16(sites, s16)
    assert list(got.columns) == CONTEXT_COLUMNS + [c for c in SITE_COLUMNS if c not in CONTEXT_COLUMNS]
    assert got.label.tolist() == LABELS and got.region.tolist() == REGION and got.hexamer.tolist() == [True, False] * 3 + [True]
    assert got.a_band.eq("0-4").all() and got.n_accepted.tolist() == sites.n_accepted.tolist()
    pd.testing.assert_frame_equal(got[SITE_COLUMNS], sites)
    # a site of stage 16 that this scan lacks, one that stage 16 lacks, and another number of accepted reads
    with pytest.raises(ValueError, match="1 sites of stage 16 without reads here, 0 sites here that stage 16 lacks"):
        attach_stage16(sites.iloc[1:], s16)
    with pytest.raises(ValueError, match="0 sites of stage 16 without reads here, 2 sites here that stage 16 lacks"):
        attach_stage16(sites, s16[~s16.site.isin([200, 300])])
    with pytest.raises(ValueError, match="1 sites with another number of accepted reads"):
        attach_stage16(sites, s16.assign(n_accepted=np.where(s16.site == 400, 3, s16.n_accepted)))
    # every row given counts: the caller keeps the references it scanned
    with pytest.raises(ValueError, match="7 sites of stage 16 without reads here"):
        attach_stage16(sites, pd.concat([s16, s16.assign(ref="chrE")], ignore_index=True))


def _attached():
    sites, by_class = reduce_sites(_reads(REDUCE_READS), "chrS")
    return attach_stage16(sites, _stage16(sites, LABELS, REGION)), by_class


def test_intron_sites_with_a_reason_other_than_novel_junction():
    sites, _ = _attached()
    assert not len(intron_sites_off_model(sites))    # the intron sites are (+,200) and (+,500): novel junction or a class
    bad = intron_sites_off_model(sites.assign(region="intron"))
    assert list(zip(bad.strand, bad.site, bad.context)) == [("+", 400, AS)]
    # intergenic when placed and novel junction when strict
    bad = intron_sites_off_model(sites.assign(region="intron", context=sites.context.replace(AS, IG),
                                              context_without_chain_novel=NJ))
    assert list(zip(bad.strand, bad.site, bad.context)) == [("+", 400, IG)]
    # the strict version is checked too
    sites.loc[0, ["region", "context_without_chain_novel"]] = ["intron", IG]
    assert intron_sites_off_model(sites).site.tolist() == [100]


def test_context_tables_count_sites_by_top_class_and_reads_by_their_own_class():
    sites, by_class = _attached()
    t = context_tables(sites, by_class, ["label", "a_band"])
    assert list(t.columns) == ["version", "label", "a_band", "context", "sites", "reads", "share_spliced", "share_5p_complete"]
    assert (t.a_band == "0-4").all()
    got = {(r.version, r.label, r.context): (r.sites, r.reads, r.share_spliced, r.share_5p_complete) for r in t.itertuples()}
    want = {
        # non-templated sites: (+,300) novel junction and (+,500) intron_after_junction
        ("placed", "nontemplated", "intron_after_junction"): (1, 1, 1.0, 0.0), ("placed", "nontemplated", NJ): (1, 2, 0.0, 0.0),
        ("placed", "nontemplated", AS): (0, 1, 0.0, 0.0), ("placed", "nontemplated", IG): (0, 1, 0.0, 0.0),
        # templated sites: (+,100) CDS, (+,200) novel junction, (+,400) antisense, (-,50) annotated_end, (-,100) 3UTR
        ("placed", "templated", "annotated_end"): (1, 1, 1.0, 1.0), ("placed", "templated", "3UTR"): (1, 2, 0.5, 0.0),
        ("placed", "templated", "CDS"): (1, 3, 1.0, 1 / 3), ("placed", "templated", NJ): (1, 2, 0.0, 0.0),
        ("placed", "templated", AS): (1, 2, 0.0, 0.0), ("placed", "templated", IG): (0, 1, 0.0, 0.0),
        # strict: the four chain_novel reads are novel junction reads without a matched intron
        ("strict", "nontemplated", NJ): (2, 3, 0.0, 0.0), ("strict", "nontemplated", AS): (0, 1, 0.0, 0.0),
        ("strict", "nontemplated", IG): (0, 1, 0.0, 0.0),
        ("strict", "templated", "annotated_end"): (1, 1, 1.0, 1.0), ("strict", "templated", "3UTR"): (1, 1, 0.0, 0.0),
        ("strict", "templated", "CDS"): (0, 1, 1.0, 1.0), ("strict", "templated", NJ): (2, 5, 0.0, 0.0),
        ("strict", "templated", AS): (1, 2, 0.0, 0.0), ("strict", "templated", IG): (0, 1, 0.0, 0.0),
    }
    # rows in the order of the versions, the labels and CLASSES; a class without a site and without a read has no row
    assert list(got) == list(want)
    for k in want:
        assert got[k] == pytest.approx(want[k]), k
    r = context_tables(sites, by_class, ["region"])
    assert list(r.columns) == ["version", "region", "context", "sites", "reads", "share_spliced", "share_5p_complete"]
    for version, col in (("placed", "context"), ("strict", "context_without_chain_novel")):
        v = r[r.version == version]
        assert v.sites.sum() == 7 and v.reads.sum() == 16
        assert v.groupby("region").sites.sum().to_dict() == sites.groupby("region").size().to_dict()
        assert v[v.sites > 0].groupby("context").sites.sum().to_dict() == sites.groupby(col).size().to_dict()
    intron = r[(r.version == "placed") & (r.region == "intron")]
    assert list(zip(intron.context, intron.sites, intron.reads)) == [("CDS", 0, 1), ("intron_after_junction", 1, 1), (NJ, 1, 3)]


def test_totals_of_the_json():
    sites, by_class = _attached()
    t = totals(sites, by_class)
    zero = dict.fromkeys(CLASSES, 0)
    # the sites of the region intron are (+,200), novel junction in both versions, and (+,500), novel junction when strict
    assert t["accepted_reads"] == 16 and t["sites"] == 7 and t["intron_sites"] == 2
    assert t["placed"] == {"reads_by_class": zero | {"annotated_end": 1, "3UTR": 2, "CDS": 3, "intron_after_junction": 1, NJ: 4,
                                                     AS: 3, IG: 2},
                           "sites_by_context": zero | {"annotated_end": 1, "3UTR": 1, "CDS": 1, "intron_after_junction": 1, NJ: 2,
                                                       AS: 1},
                           "sites_tied": 4, "sites_top_share_at_most_half": 4, "intron_sites_top_no_model": 1}
    assert t["strict"] == {"reads_by_class": zero | {"annotated_end": 1, "3UTR": 1, "CDS": 1, NJ: 8, AS: 3, IG: 2},
                           "sites_by_context": zero | {"annotated_end": 1, "3UTR": 1, NJ: 4, AS: 1},
                           "sites_tied": 4, "sites_top_share_at_most_half": 4, "intron_sites_top_no_model": 2}
    assert t["sites_top_class_on_chain_novel_only"] == 2
    assert t["flagged_reads"] == {"starts_outside": 1, "chain_novel": 4, "chain_novel_suffix1": 2, "retained_after_junction": 1,
                                  "end_NF": 1, "no_splice": 2}
    # per class of the placed version: its reads, then the chain_novel ones among them and those of them that had a
    # compatible transcript with another class, and the same for the other reads
    assert t["by_class"]["CDS"] == {"reads": 3, "matched_intron": 3, "complete_5p": 1, "covers_start": 1, "other_class": 0,
                                    "chain_novel": 2, "chain_novel_other_class": 1}
    assert t["by_class"]["annotated_end"] == {"reads": 1, "matched_intron": 1, "complete_5p": 1, "covers_start": 1,
                                              "other_class": 1, "chain_novel": 0, "chain_novel_other_class": 0}
    assert t["by_class"][NJ] == dict.fromkeys(("matched_intron", "complete_5p", "covers_start", "other_class", "chain_novel",
                                               "chain_novel_other_class"), 0) | {"reads": 4}
    assert list(t["by_class"]) == list(CLASSES)
    zero = dict.fromkeys(SUB_CAUSES, 0)
    assert t["novel_junction_reads_by_sub_cause"] == zero | {NEAR: 1, SHORT: 1, DONOR: 1, NEITHER: 1}
    assert t["novel_junction_sites_by_main_sub_cause"] == zero | {NEAR: 1, DONOR: 1}


def test_isoform_ends_are_priming_site_boundaries(tmp_path):
    gtf = tmp_path / "iso.gtf"
    gtf.write_text("##pacbio-collapse-version 1.0\n"
                   'chrS\tPacBio\ttranscript\t101\t500\t.\t+\t.\tgene_id "PB.1"; transcript_id "PB.1.1";\n'
                   'chrS\tPacBio\texon\t101\t200\t.\t+\t.\tgene_id "PB.1"; transcript_id "PB.1.1";\n'
                   'chrS\tPacBio\texon\t401\t500\t.\t+\t.\tgene_id "PB.1"; transcript_id "PB.1.1";\n'
                   'chrS\tPacBio\ttranscript\t101\t500\t.\t-\t.\tgene_id "PB.2"; transcript_id "PB.2.1";\n'
                   'chrE\tPacBio\ttranscript\t7\t90\t.\t-\t.\tgene_id "PB.3"; transcript_id "PB.3.2";\n')
    ends = isoform_ends(gtf)
    assert list(ends.columns) == ["isoform", "ref", "strand", "site"]
    # plus: the last base is 1-based 500, 0-based 499, and the boundary after it 500; minus: the last base is 0-based 100
    # and the boundary is 100, as sites.priming_site gives for a read
    assert list(ends.itertuples(index=False, name=None)) == [("PB.1.1", "chrS", "+", 500), ("PB.2.1", "chrS", "-", 100),
                                                             ("PB.3.2", "chrE", "-", 6)]
    assert isoform_ends(gtf, ["chrE"]).isoform.tolist() == ["PB.3.2"]
    a = long_read([(400, 500)], 500)
    assert priming_site(a) == 500
    a = long_read([(G - 500, G - 400)], G - 400, minus=True)       # the mirrored read covers 0-based 400..499 on minus
    assert (a.reference_start, priming_site(a)) == (400, 400)


def test_pigeon_crosscheck_matches_isoform_ends_to_sites():
    sites = pd.DataFrame({"ref": "chrS", "strand": ["+", "+", "-", "+"], "site": [500, 520, 100, 900],
                          "context": ["3UTR", "CDS", NJ, "annotated_end"], "n_accepted": [2, 1, 1, 5]})
    ends = pd.DataFrame({"isoform": ["a", "b", "c", "d", "e", "f"], "ref": ["chrS"] * 5 + ["chrE"],
                         "strand": ["+", "+", "-", "-", "+", "+"], "site": [510, 531, 90, 111, 895, 900]})
    cls = pd.DataFrame({"isoform": ["a", "b", "c", "d", "e", "x"],
                        "structural_category": ["full-splice_match"] * 3 + ["incomplete-splice_match", "full-splice_match",
                                                                            "genic"],
                        "subcategory": ["reference_match", "reference_match", "mono-exon", "3prime_fragment", "reference_match",
                                        "mono-exon"]})
    table, stats = pigeon_crosscheck(ends, cls, sites)
    assert list(table.columns) == ["structural_category", "subcategory", "context", "isoforms"]
    # a: 500 and 520 are both 10 away, the lower site wins (the stage 16 match); b: 520 is 11 away; c: 100 is 10 away;
    # d: 11 away; e: 900 is 5 away; f has no classification row and no site on its reference
    assert list(table.itertuples(index=False, name=None)) == [
        ("full-splice_match", "mono-exon", NJ, 1), ("full-splice_match", "reference_match", "annotated_end", 1),
        ("full-splice_match", "reference_match", "3UTR", 1), ("full-splice_match", "reference_match", "none", 1),
        ("incomplete-splice_match", "3prime_fragment", "none", 1), ("not_classified", "not_classified", "none", 1)]
    assert stats == {"isoform_ends": 6, "without_classification": 1, "matched": 3, "share_matched": 0.5}


def test_link_1179_gives_the_context_and_the_shifted_null():
    sites = pd.DataFrame({"ref": "chrS", "strand": ["+", "+", "-", "-", "+"], "site": [1000, 3000, 5000, 4000, 2000],
                          "label": ["templated", "nontemplated", "templated", "nontemplated", "ambiguous"],
                          "context": ["CDS", "annotated_end", "intron_unspliced", "3UTR", NJ], "n_accepted": [3, 1, 2, 1, 1]})
    mol = pd.DataFrame({"mol": [10, 11, 12, 13, 14], "ref": "chrS", "strand": ["+", "+", "-", "-", "+"],
                        "site": [1005, 2990, 5010, 3000, 4000],
                        "label": ["templated", "nontemplated", "nontemplated", "templated", "ambiguous"]})
    context, link = link_1179(mol, sites)
    # real: 1005 -> 1000 (CDS, same label), 2990 -> 3000 (annotated_end, same), 5010 on minus -> 5000 (other label),
    # 3000 on minus and 4000 on plus have no site within 10 nt
    assert context.tolist() == ["CDS", "annotated_end", "intron_unspliced", "none", "none"]
    assert link["real"]["matched"] == 3
    assert link["real"]["share_matched_by_label"] == {"templated": 0.5, "nontemplated": 1.0, "ambiguous": 0.0}
    assert link["real"]["agreement"] == {"templated": 1.0, "nontemplated": 0.5, "pooled": 2 / 3}
    # moved 1,000 nt upstream in transcript sense: plus sites go down (5, 1990, 3000), minus sites go up (6010, 4000).
    # 1990 -> 2000 (ambiguous site), 3000 -> 3000 (nontemplated site, the molecule is ambiguous), 4000 on minus -> 4000
    assert link["null_shift_nt"] == 1000 and link["null"]["matched"] == 3
    assert link["null"]["share_matched_by_label"] == {"templated": 0.5, "nontemplated": 0.5, "ambiguous": 1.0}
    assert link["null"]["agreement"] == {"templated": 0.0, "nontemplated": 0.0, "pooled": 0.0}
