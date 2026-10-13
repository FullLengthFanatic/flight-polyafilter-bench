import pytest

from pafbench.bd67 import accept_long_read, terminal_a_run
from pafbench.lrcontext import CLASSES, Models, classify, load_models, read_body
from tests.synth_long import (ACCEPTOR, DESIGNED, DONOR, G, LOCUS, NEAR, NEITHER, NOT_JOINED, OPPOSITE, SHORT, ctx, expected,
                              long_read, write_gtf)

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
    # every aligned base in the run: no body
    a = long_read([(1050, 1090)], 1050)
    assert terminal_a_run(a)[0] == 1050 and read_body(a, 1050) == []
    a = long_read([(1050, 1090)], 1050, minus=True)
    assert terminal_a_run(a)[0] == G - 1050 and read_body(a, G - 1050) == []


def test_load_models_mirrors_the_minus_strand(models):
    assert [t.tid for t in models.tx] == list(LOCUS) + [t + "m" for t in LOCUS]
    ta, tam = models.tx[0], models.tx[6]
    assert (ta.start, ta.end, ta.cds, ta.end_nf) == (1000, 2000, (1050, 1700), False)
    assert ta.exons == [(1000, 1100), (1200, 1300), (1400, 1500), (1600, 2000)]
    # in transcript sense the mirrored transcript is the plus transcript shifted by -G
    assert (tam.start, tam.end, tam.cds) == (1000 - G, 2000 - G, (1050 - G, 1700 - G))
    assert tam.exons == [(s - G, e - G) for s, e in ta.exons]
    assert models.tx[2].cds is None and [t.end_nf for t in models.tx] == [False] * 5 + [True] + [False] * 5 + [True]
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
    m = Models([("P", False, (0, 200), [(0, 100), (140, 200)], [], False),
                ("M", True, (-400, -250), [(-400, -330), (-300, -250)], [], False)])
    assert m.sub_cause(False, (102, 141)) == NEAR         # also shorter than 50 nt
    assert m.sub_cause(False, (300, 330)) == SHORT        # also the intron of M on the other strand
    assert m.sub_cause(False, (100, 149)) == SHORT        # 49 nt; 9 nt off at the acceptor, so not near
    assert m.sub_cause(False, (100, 150)) == DONOR        # 50 nt is not short
    # the intron of P seen from the minus strand, made long enough: 100..160 on plus is -160..-100 on minus
    m = Models([("P", False, (0, 200), [(0, 100), (160, 200)], [], False),
                ("M", True, (-200, 0), [(-200, -160), (-100, 0)], [], False)])
    assert m.sub_cause(False, (100, 160)) == NEAR and m.sub_cause(True, (-160, -100)) == NEAR   # annotated on its own strand
    m = Models([("P", False, (0, 200), [(0, 100), (160, 200)], [], False)])
    assert m.sub_cause(True, (-160, -100)) == OPPOSITE


def test_choice_falls_back_to_the_transcript_id():
    rows = [(tid, False, (0, 1000), [(0, 400), (600, 1000)], [], False) for tid in ("T2", "T1", "T3")]
    assert classify(Models(rows), False, [(100, 200)]) == ctx("ncexon", "T1")
    assert classify(Models(rows[::-1]), False, [(100, 400), (600, 700)]) == ctx("ncexon", "T1", 1)


def test_choice_order_end_then_overlap_then_nearest_end():
    # one exon each; body 100..300, p = 299
    rows = [("far", False, (0, 1000), [(0, 1000)], [], False),          # all 200 nt exonic, end 700 nt away
            ("near", False, (0, 305), [(0, 150), (250, 305)], [], False)]   # 100 nt exonic, ends within the tolerance
    assert classify(Models(rows), False, [(100, 300)]) == ctx("annotated_end", "near", other_class=True)
    # no end within the tolerance: the larger exonic overlap wins against the nearer end
    rows[1] = ("near", False, (0, 400), [(0, 150), (250, 400)], [], False)
    assert classify(Models(rows), False, [(100, 300)]) == ctx("ncexon", "far")
    # equal overlap: the nearer annotated 3' end
    rows[1] = ("near", False, (0, 400), [(0, 400)], [], False)
    assert classify(Models(rows), False, [(100, 300)]) == ctx("ncexon", "near")
    # the overlap is counted over every exon a block touches: 50 + 200 nt of "two" against 70 + 50 nt of "wide"
    rows = [("wide", False, (0, 400), [(0, 120), (300, 400)], [], False), ("two", False, (0, 400), [(0, 100), (150, 400)], [], False)]
    assert classify(Models(rows), False, [(50, 350)]) == ctx("ncexon", "two", complete_5p=True)


def test_unspliced_read_longer_than_the_transcript_is_placed_by_its_end_with_the_tolerance():
    m = Models([("T", False, (100, 200), [(100, 200)], [], False)])
    # no span contains the read; its last base lies 10 nt past the last base of T, then 11 nt
    assert classify(m, False, [(50, 210)]) == ctx("annotated_end", "T", starts_outside=True, complete_5p=True)
    assert classify(m, False, [(50, 211)]) == ctx("downstream")


def test_end_tolerance_boundaries(models):
    # last base of TA is 1999: 10 nt before or after it is annotated_end, 11 nt is 3UTR or downstream
    klass = lambda body: classify(models, False, body).klass
    chain = [(1250, 1300), (1400, 1500)]
    assert [klass(chain + [(1600, e)]) for e in (1989, 1990, 2010, 2011)] == ["3UTR", "annotated_end", "annotated_end", "downstream"]
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
