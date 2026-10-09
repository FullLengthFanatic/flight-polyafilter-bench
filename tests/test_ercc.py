import gzip
import importlib.util
import json
import math
import sys

import numpy as np
import pandas as pd
import pysam
import pytest

from pafbench.config import REPO
from pafbench.ercc import ercc_correlation, ercc_reads, ercc_settings, ercc_tables, parse_ercc_conc, umi_counts
from pafbench.flags import Setting, flag_reference, setting_tag
from pafbench.grid import published
from pafbench.transcripts import load_transcripts

# header and three data lines of cms_095046.txt (ERCC_CONC_URL, fetched 2026-10-08), CRLF as served
EXCERPT = ("Re-sort ID\tERCC ID\tsubgroup\tconcentration in Mix 1 (attomoles/ul)\tconcentration in Mix 2 (attomoles/ul)\t"
           "expected fold-change ratio\tlog2(Mix 1/Mix 2)\r\n"
           "1\tERCC-00130\tA\t30000\t7500\t4\t2\r\n"
           "4\tERCC-00108\tA\t937.5\t234.375\t4\t2\r\n"
           "92\tERCC-00048\tD\t0.01430512\t0.02861023\t0.5\t-1\r\n")


def test_parse_ercc_conc():
    conc = parse_ercc_conc(EXCERPT)
    assert conc.columns.tolist() == ["ercc_id", "mix1_amol_per_ul"]
    assert conc.ercc_id.tolist() == ["ERCC-00130", "ERCC-00108", "ERCC-00048"]
    assert conc.mix1_amol_per_ul.tolist() == [30000.0, 937.5, 0.01430512]
    assert conc.mix1_amol_per_ul.dtype == np.float64


def test_parse_ercc_conc_finds_columns_by_header():
    rows = [line.split("\t") for line in EXCERPT.replace("\r\n", "\n").strip().split("\n")]
    order = [4, 3, 6, 1, 0, 5, 2]  # Mix 2 before Mix 1, ERCC ID after them
    text = "\n".join("\t".join(r[k] for k in order) for r in rows) + "\n"
    pd.testing.assert_frame_equal(parse_ercc_conc(text), parse_ercc_conc(EXCERPT))


def _reads():
    return pd.DataFrame({"ref": ["E1", "E1", "E1", "E1", "E2", "E3"],
                         "cb": ["c1", "c1", "c1", "c2", "c1", "c1"],
                         "ub": ["u1", "u1", "u2", "u1", "u1", "u9"]})


def test_umi_counts():
    reads = _reads()
    # E1: UMIs (c1,u1) with two reads, (c1,u2), (c2,u1); E2: one; E3: one
    assert umi_counts(reads).to_dict() == {"E1": 3, "E2": 1, "E3": 1}
    removed = np.array([True, False, True, True, True, False])
    # (c1,u1) survives through its second read; (c1,u2), (c2,u1) and E2's UMI lose their only read; E2 stays at 0
    c = umi_counts(reads, removed)
    assert c.to_dict() == {"E1": 1, "E2": 0, "E3": 1}
    assert c.index.name == "ercc_id" and c.dtype == np.int64
    assert umi_counts(reads, np.ones(6, bool)).to_dict() == {"E1": 0, "E2": 0, "E3": 0}


def test_ercc_correlation():
    conc = pd.DataFrame({"ercc_id": ["E1", "E2", "E3", "E4", "E7"], "mix1_amol_per_ul": [10.0, 100.0, 1000.0, 10000.0, 5.0]})
    # E6 has no concentration and E7 no count: both left out; log10(UMI + 1) = 1, 2, 3, 4 = log10(conc)
    exact = pd.Series({"E1": 9, "E2": 99, "E3": 999, "E4": 9999, "E6": 5})
    r = ercc_correlation(exact, conc)
    assert r["n"] == 4 and r["pearson_log10"] == pytest.approx(1.0) and r["spearman"] == pytest.approx(1.0)
    # x = 1, 2, 3, 4 and y = 1, 2, 3, 1: pearson 0.5 / sqrt(5 * 2.75) = 0.134840; spearman on ranks
    # (1, 2, 3, 4) and (1.5, 3, 4, 1.5): 0.5 / sqrt(5 * 4.5) = 0.105409
    r = ercc_correlation(pd.Series({"E1": 9, "E2": 99, "E3": 999, "E4": 9}), conc)
    assert r["n"] == 4
    assert r["pearson_log10"] == pytest.approx(0.5 / math.sqrt(13.75)) and r["spearman"] == pytest.approx(0.5 / math.sqrt(22.5))
    # a zero count enters as log10(1) = 0
    r = ercc_correlation(pd.Series({"E1": 0, "E2": 99, "E3": 999}), conc)
    assert r["pearson_log10"] == pytest.approx(np.corrcoef([0, 2, 3], [1, 2, 3])[0, 1])


def test_ercc_correlation_constant_counts_is_none():
    conc = pd.DataFrame({"ercc_id": ["E1", "E2", "E3"], "mix1_amol_per_ul": [1.0, 10.0, 100.0]})
    assert ercc_correlation(pd.Series({"E1": 0, "E2": 0, "E3": 0}), conc) == {"n": 3, "pearson_log10": None, "spearman": None}


def test_ercc_tables():
    reads = _reads().assign(tA=[True, False, True, True, True, False], tB=False)
    conc = pd.DataFrame({"ercc_id": ["E1", "E2", "E3"], "mix1_amol_per_ul": [100.0, 1.0, 10.0]})
    counts, per = ercc_tables(reads, ["tA", "tB"], conc)
    assert counts.columns.tolist() == ["ercc_id", "setting", "umis"]
    assert list(zip(counts.ercc_id, counts.setting, counts.umis)) == [
        ("E1", "unfiltered", 3), ("E2", "unfiltered", 1), ("E3", "unfiltered", 1),
        ("E1", "tA", 1), ("E2", "tA", 0), ("E3", "tA", 1),
        ("E1", "tB", 3), ("E2", "tB", 1), ("E3", "tB", 1)]
    assert per["unfiltered"]["umis"] == 5 and per["unfiltered"]["umis_kept_share"] == 1.0
    assert per["tA"]["umis"] == 2 and per["tA"]["umis_kept_share"] == 0.4
    assert per["tB"] == per["unfiltered"]
    # unfiltered: x = log10(conc) = 2, 0, 1 and y = log10(4), log10(2), log10(2)
    assert per["unfiltered"]["correlation"]["pearson_log10"] == pytest.approx(
        np.corrcoef([2, 0, 1], np.log10([4, 2, 2]))[0, 1])


@pytest.fixture(scope="module")
def tagged(synth, tmp_path_factory):
    out = tmp_path_factory.mktemp("ercc") / "tagged.bam"
    with pysam.AlignmentFile(str(synth.bam)) as fin, pysam.AlignmentFile(str(out), "wb", template=fin) as fout:
        for i, a in enumerate(fin.fetch(until_eof=True)):
            tags = []
            if i % 5 != 4:
                tags.append(("CB", f"c{i % 4}"))   # c3 is not a called cell
            if i % 6 != 5:
                tags.append(("UB", f"u{i % 9}"))
            a.set_tags(tags)
            if i % 11 == 10 and not a.is_secondary:
                a.flag |= 0x800   # supplementary: dropped even with a called CB and a UB
            fout.write(a)
    pysam.index(str(out))
    return out


def test_ercc_reads(synth, tagged):
    tx = load_transcripts(synth.gtf)
    settings = [Setting(300, 10, 1, True), Setting(150, None, 0, False)]
    tags = [setting_tag(s) for s in settings]
    cells = {"c0", "c1", "c2"}
    with pysam.AlignmentFile(str(tagged)) as bam:
        by_name = {a.query_name: a for a in bam.fetch(until_eof=True)}  # qnames are unique in the synthetic BAM
    for ref in ("chrA", "chrB"):
        got = ercc_reads(tagged, synth.fasta, ref, tx, settings, cells)
        fl = flag_reference(tagged, synth.fasta, ref, tx, settings)
        keep = [not (by_name[q].flag & 0x900) and by_name[q].has_tag("UB") and by_name[q].has_tag("CB")
                and by_name[q].get_tag("CB") in cells for q in fl.qname]
        exp = pd.DataFrame({"ref": ref, "cb": [by_name[q].get_tag("CB") for q in fl.qname[keep]],
                            "ub": [by_name[q].get_tag("UB") for q in fl.qname[keep]]})
        exp = pd.concat([exp, fl.loc[keep, tags].reset_index(drop=True)], axis=1)
        pd.testing.assert_frame_equal(got, exp, check_dtype=False)
        assert got.columns.tolist() == ["ref", "cb", "ub"] + tags
        assert 0 < len(got) < len(fl)
    # chrA: secondaries and supplementaries are dropped even with CB and UB; both settings remove some kept reads
    got = ercc_reads(tagged, synth.fasta, "chrA", tx, settings, cells)
    assert got[tags].any().all() and (~got[tags]).any().all()
    with pysam.AlignmentFile(str(tagged)) as bam:
        sup = [a for a in bam.fetch("chrA")
               if a.flag & 0x800 and a.has_tag("UB") and a.has_tag("CB") and a.get_tag("CB") in cells]
    assert len(sup) > 3


def test_ercc_settings(tmp_path):
    pub = [setting_tag(s) for s in published()]
    assert pub == ["c300_l10_m1_e", "c300_l10_m1_i", "c275_l9_m1_e", "c275_l9_m1_i",
                   "c180_l11_m2_e", "c180_l11_m2_i", "c375_l3_m2_e", "c375_l3_m2_i"]
    settings, chosen = ercc_settings(tmp_path / "absent.json")
    assert [setting_tag(s) for s in settings] == pub and chosen is None
    sel = tmp_path / "selection.json"
    sel.write_text(json.dumps({"modes": {"e": {"introns": False, "selection": {"tag": "c250_l9_m1_e"}},
                                         "i": {"introns": True, "selection": {"tag": None}}}}))
    settings, chosen = ercc_settings(sel)
    assert chosen == {"e": "c250_l9_m1_e", "i": None}
    assert [setting_tag(s) for s in settings] == pub + ["c250_l9_m1_e"]
    assert settings[-1] == Setting(250, 9, 1, False)
    # a chosen tag that is already a published setting is not added twice
    sel.write_text(json.dumps({"modes": {"e": {"selection": {"tag": "c275_l9_m1_e"}},
                                         "i": {"selection": {"tag": "c150_l20_m0_i"}}}}))
    settings, chosen = ercc_settings(sel)
    assert chosen == {"e": "c275_l9_m1_e", "i": "c150_l20_m0_i"}
    assert [setting_tag(s) for s in settings] == pub + ["c150_l20_m0_i"]


SPIKE_NAMES = {"chrA": "ERCC-00130", "chrB": "ERCC-00108"}


def _spike_set(synth, d):
    """The synthetic genome as spike-in contigs: chrA as ERCC-00130, chrB as ERCC-00108 and again as SIRV1 (not an ERCC,
    so stage 15 must leave it out). Records get CB c((i // 3) % 4) (c3 is not a called cell; none when i % 7 == 6) and
    UB u(i // 3) (none when i % 11 == 10), so most UMIs have several reads. Returns (fasta, gtf, sorted indexed bam)."""
    d.mkdir()
    with pysam.FastaFile(str(synth.fasta)) as fa:
        seqs = {c: fa.fetch(c) for c in SPIKE_NAMES}
    fasta, gtf, bam = d / "spikes.fa", d / "spikes.gtf", d / "star_sorted.bam"
    fasta.write_text("".join(f">{SPIKE_NAMES[c]}\n{s}\n" for c, s in seqs.items()) + f">SIRV1\n{seqs['chrB']}\n")
    pysam.faidx(str(fasta))
    gtf.write_text("".join(SPIKE_NAMES[line[:4]] + line[4:] for line in synth.gtf.read_text().splitlines(True)))
    hdr = pysam.AlignmentHeader.from_dict({"SQ": [{"SN": SPIKE_NAMES[c], "LN": len(s)} for c, s in seqs.items()]
                                           + [{"SN": "SIRV1", "LN": len(seqs["chrB"])}]})
    sirv = []
    with pysam.AlignmentFile(str(synth.bam)) as fin, pysam.AlignmentFile(str(bam), "wb", header=hdr) as fout:
        for i, a in enumerate(fin.fetch(until_eof=True)):
            tags = [] if i % 7 == 6 else [("CB", f"c{(i // 3) % 4}")]
            if i % 11 != 10:
                tags.append(("UB", f"u{i // 3}"))
            rec = a.to_dict() | {"ref_name": SPIKE_NAMES[a.reference_name]}
            b = pysam.AlignedSegment.from_dict(rec, hdr)
            b.set_tags(tags)
            fout.write(b)
            if a.reference_name == "chrB":
                b = pysam.AlignedSegment.from_dict(rec | {"name": f"{a.query_name}_s", "ref_name": "SIRV1"}, hdr)
                b.set_tags(tags)
                sirv.append(b)
        for b in sirv:
            fout.write(b)
    pysam.index(str(bam))
    return fasta, gtf, bam


def _brute_umis(bam, fasta, gtf, settings, cells):
    """{(ercc_id, setting): UMIs} counted directly: distinct (CB, UB) over primary records with a called-cell CB and a
    UB, all of them ("unfiltered") or those flags.flag_reference keeps; records joined to flags by qname. Also the
    number of such records."""
    tx = load_transcripts(gtf)
    tags = [setting_tag(s) for s in settings]
    umis, n = {}, 0
    with pysam.AlignmentFile(str(bam)) as fh:
        for ref in SPIKE_NAMES.values():
            by_name = {a.query_name: a for a in fh.fetch(ref)}  # qnames are unique in the synthetic BAM
            fl = flag_reference(bam, fasta, ref, tx, settings)
            sets = {t: set() for t in ["unfiltered"] + tags}
            for q, removed in zip(fl.qname, fl[tags].to_numpy()):
                a = by_name[q]
                cb = a.get_tag("CB") if a.has_tag("CB") else None
                if a.flag & 0x900 or cb not in cells or not a.has_tag("UB"):
                    continue
                n += 1
                key = (cb, a.get_tag("UB"))
                sets["unfiltered"].add(key)
                for t, r in zip(tags, removed):
                    if not r:
                        sets[t].add(key)
            umis |= {(ref, t): len(v) for t, v in sets.items()}
    return umis, n


def _stage15(monkeypatch, tmp_path, synth):
    spec = importlib.util.spec_from_file_location("stage15", REPO / "scripts" / "15_ercc_counts.py")
    stage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage)
    fasta, gtf, bam = _spike_set(synth, tmp_path / "base")
    outs = tmp_path / "lib" / "outs"
    (outs / "filtered_feature_bc_matrix").mkdir(parents=True)
    with gzip.open(outs / "filtered_feature_bc_matrix" / "barcodes.tsv.gz", "wt") as fh:
        fh.write("c0\nc1\nc2\n")
    (tmp_path / "ref").mkdir()
    (tmp_path / "ref" / "ercc_conc_raw.txt").write_bytes(EXCERPT.encode())

    def no_network(*args, **kwargs):
        raise AssertionError("stage 15 downloaded the concentration table although it exists")

    monkeypatch.setattr(stage.urllib.request, "urlopen", no_network)
    monkeypatch.setattr(stage, "LIBRARIES", {"SYN": outs / "possorted_genome_bam.bam"})
    monkeypatch.setattr(stage, "SIRV_FASTA", fasta)
    monkeypatch.setattr(stage, "SIRV_GTF", gtf)
    return stage, fasta, gtf, bam


def test_stage15_on_synthetic_spikes(synth, tmp_path, monkeypatch, capsys):
    stage, fasta, gtf, bam = _stage15(monkeypatch, tmp_path, synth)
    out, ref = tmp_path / "out", tmp_path / "ref"
    argv = ["15", "SYN", "--base", str(bam.parent), "--out", str(out), "--ref", str(ref)]
    monkeypatch.setattr(sys, "argv", argv + ["--selection", str(tmp_path / "absent.json")])
    stage.main()
    printed = capsys.readouterr().out.strip().splitlines()
    assert printed[0] == f"{tmp_path / 'absent.json'} absent: published settings only"

    settings = list(published())
    tags = [setting_tag(s) for s in settings]
    umis, n_reads = _brute_umis(bam, fasta, gtf, settings, {"c0", "c1", "c2"})
    ids = sorted(SPIKE_NAMES.values())  # ERCC-00108 then ERCC-00130; SIRV1 is not an ERCC contig
    counts = pd.read_parquet(out / "ercc_counts.parquet")
    assert counts.columns.tolist() == ["ercc_id", "setting", "umis"]
    assert list(counts.itertuples(index=False, name=None)) == [
        (e, t, umis[(e, t)]) for t in ["unfiltered"] + tags for e in ids]
    # chrA holds records 0..144; chrB's 145 (c0, u48), 147 and 148 (c1, u49) are kept, 146 has no CB
    assert umis[("ERCC-00108", "unfiltered")] == 2
    assert umis[("ERCC-00130", "unfiltered")] > 2
    assert any(umis[(e, t)] < umis[(e, "unfiltered")] for e in ids for t in tags)
    pd.testing.assert_frame_equal(pd.read_csv(ref / "ercc_conc.tsv", sep="\t"), parse_ercc_conc(EXCERPT))

    js = json.loads((out / "ercc.json").read_text())
    assert list(js) == ["library", "provisional", "note", "concentration_source", "selection", "ercc_contigs", "reads",
                        "ercc_detected", "ercc_with_concentration", "settings"]
    assert (js["library"], js["provisional"], js["note"]) == ("SYN", True, stage.NOTE)
    assert js["concentration_source"] == stage.ERCC_CONC_URL
    assert js["selection"] is None
    assert (js["ercc_contigs"], js["reads"], js["ercc_detected"], js["ercc_with_concentration"]) == (2, n_reads, 2, 2)
    assert list(js["settings"]) == ["unfiltered"] + tags
    base = umis[("ERCC-00108", "unfiltered")] + umis[("ERCC-00130", "unfiltered")]
    for t in ["unfiltered"] + tags:
        lo, hi = umis[("ERCC-00108", t)], umis[("ERCC-00130", t)]
        # two ERCCs: log10(UMI + 1) against log10(937.5) and log10(30000) correlates +1 or -1, None when the counts tie
        r = None if lo == hi else pytest.approx(1.0 if hi > lo else -1.0)
        assert js["settings"][t]["correlation"] == {"n": 2, "pearson_log10": r, "spearman": r}
        assert js["settings"][t]["umis"] == lo + hi
        assert js["settings"][t]["umis_kept_share"] == pytest.approx((lo + hi) / base)
    assert js["settings"]["unfiltered"]["correlation"]["pearson_log10"] == pytest.approx(1.0)

    # with stage 17's selection: the chosen tag per mode (or None); the chosen tag that is not published is counted too
    sel = tmp_path / "selection.json"
    sel.write_text(json.dumps({"modes": {"e": {"introns": False, "selection": {"tag": "c250_l9_m1_e"}},
                                         "i": {"introns": True, "selection": {"tag": None}}}}))
    monkeypatch.setattr(sys, "argv", argv + ["--selection", str(sel)])
    stage.main()
    assert "absent" not in capsys.readouterr().out
    js = json.loads((out / "ercc.json").read_text())
    assert js["selection"] == {"e": "c250_l9_m1_e", "i": None}
    assert list(js["settings"]) == ["unfiltered"] + tags + ["c250_l9_m1_e"]
    extra, _ = _brute_umis(bam, fasta, gtf, [Setting(250, 9, 1, False)], {"c0", "c1", "c2"})
    counts = pd.read_parquet(out / "ercc_counts.parquet")
    assert list(counts[counts.setting == "c250_l9_m1_e"].itertuples(index=False, name=None)) == [
        (e, "c250_l9_m1_e", extra[(e, "c250_l9_m1_e")]) for e in ids]
