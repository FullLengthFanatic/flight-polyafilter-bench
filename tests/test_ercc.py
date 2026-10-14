import gzip
import importlib.util
import json
import math
import sys
from collections import Counter

import numpy as np
import pandas as pd
import pysam
import pytest

from pafbench.config import REPO
from pafbench.ercc import (ercc_correlation, ercc_mix, ercc_reads, ercc_settings, ercc_tables, mix_acceptance, parse_ercc_conc,
                          parse_ercc_mixes, residual_slope, umi_counts)
from pafbench.evaluate import resample_counts
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
    assert list(js) == ["library", "provisional", "note", "concentration_source", "selection", "barcodes", "ercc_contigs", "reads",
                        "ercc_detected", "ercc_with_concentration", "settings"]
    assert (js["library"], js["provisional"], js["note"]) == ("SYN", True, stage.NOTE)
    assert js["concentration_source"] == stage.ERCC_CONC_URL
    assert js["selection"] is None
    assert js["barcodes"] == "cells"
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


def test_parse_ercc_mixes():
    mixes = parse_ercc_mixes(EXCERPT)
    assert mixes.columns.tolist() == ["ercc_id", "subgroup", "mix1_amol_per_ul", "mix2_amol_per_ul", "log2_mix1_over_mix2"]
    assert list(mixes.itertuples(index=False, name=None)) == [("ERCC-00130", "A", 30000.0, 7500.0, 2.0),
                                                              ("ERCC-00108", "A", 937.5, 234.375, 2.0),
                                                              ("ERCC-00048", "D", 0.01430512, 0.02861023, -1.0)]
    rows = [line.split("\t") for line in EXCERPT.replace("\r\n", "\n").strip().split("\n")]
    order = [6, 4, 2, 3, 1, 0, 5]  # log2 ratio first, Mix 2 before Mix 1
    text = "\n".join("\t".join(r[k] for k in order) for r in rows) + "\n"
    pd.testing.assert_frame_equal(parse_ercc_mixes(text), mixes)


SUBGROUPS = (("A", "4", "2"), ("B", "1", "0"), ("C", "0.67", "-0.58"), ("D", "0.5", "-1"))


def _mix_table(per=23, drop=()):
    """A table shaped like cms_095046.txt: per ERCCs in each subgroup (none in the subgroups of drop), Mix 1 from 30,000
    amol/ul down 21 log2 units in equal steps, offset by a quarter step per subgroup, Mix 2 = Mix 1 / fold."""
    rows = [EXCERPT.split("\r\n")[0]]
    for k, (group, fold, log2r) in enumerate(SUBGROUPS):
        for j in range(per if group not in drop else 0):
            mix1 = 30000 * 2.0 ** -((j + k / 4) * 21 / (per - 1))
            rows.append(f"{len(rows)}\tERCC-{len(rows):05d}\t{group}\t{mix1}\t{mix1 / float(fold)}\t{fold}\t{log2r}")
    return parse_ercc_mixes("\r\n".join(rows) + "\r\n")


def _simulate(mixes, column, seed, slope=1.0, sd=0.5, length_power=0.0, lengths=None):
    """Poisson UMIs per ERCC (zeros kept) with log2 mean 0.6 + slope * log2(concentration of column) + normal noise of sd,
    plus length_power * log2(length) for a capture that depends on length. slope 1: UMIs proportional to input, the
    premise under which design 3.10 expects a residual slope of about -1 from Mix 2."""
    rng = np.random.default_rng(seed)
    log2_mu = 0.6 + slope * np.log2(mixes[column].to_numpy()) + rng.normal(0, sd, len(mixes))
    if length_power:
        log2_mu += length_power * np.log2(mixes.ercc_id.map(lengths).to_numpy(np.float64))
    return pd.Series(rng.poisson(2.0 ** log2_mu), index=mixes.ercc_id.to_numpy())


def _lengths(mixes, seed=7):
    """Lengths unrelated to subgroup, 250 to 2,000 nt like the ERCCs of the SIRV FASTA."""
    return dict(zip(mixes.ercc_id, np.random.default_rng(seed).integers(250, 2001, len(mixes)).tolist()))


def test_residual_slope_equals_least_squares_on_resampled_ercc():
    rng = np.random.default_rng(3)
    x, r, y = rng.normal(0, 4, 30), rng.choice([2.0, 0.0, -0.58, -1.0], 30), rng.normal(0, 4, 30)
    w = resample_counts(30, 5, 11).astype(float)
    b, e, s = residual_slope(x, r, y, np.vstack([np.ones(30), w]))
    for k, row in enumerate(np.vstack([np.ones(30), w]).astype(int)):
        idx = np.repeat(np.arange(30), row)
        fit = np.polyfit(x[idx], y[idx], 1)
        res = y - np.polyval(fit, x)
        assert b[k] == pytest.approx(fit[0])
        assert e[k] == pytest.approx(res)
        assert s[k] == pytest.approx(np.polyfit(r[idx], res[idx], 1)[0])


def test_ercc_mix_counts_from_mix1():
    mixes = _mix_table()
    counts = _simulate(mixes, "mix1_amol_per_ul", 0)
    counts["ERCC-99999"] = 50   # not in the table: left out
    got = ercc_mix(counts, mixes, _lengths(mixes))
    ids = [i for i in counts.index if counts[i] > 0 and i in set(mixes.ercc_id)]
    assert got["n"] == len(ids) and 40 < got["n"] < 92   # ERCCs with a UMI and a concentration, as stage 15 counts them
    assert got["verdict"] == "Mix 1" and got["reason"] is None
    rs = got["residual_slope"]
    assert rs["lo"] <= 0 <= rs["hi"] and rs["hi"] < 0.5 and rs["lo"] > -0.5 and rs["n_boot"] == 1000 and rs["seed"] == 0
    assert got["pearson_log2_mix1"] > got["pearson_log2_mix2"] > 0.8
    m = mixes.set_index("ercc_id").loc[ids]
    y = np.log2(counts[ids].to_numpy(np.float64))
    assert got["pearson_log2_mix1"] == pytest.approx(np.corrcoef(y, np.log2(m.mix1_amol_per_ul))[0, 1])
    assert got["pearson_log2_mix2"] == pytest.approx(np.corrcoef(y, np.log2(m.mix2_amol_per_ul))[0, 1])
    fit = np.polyfit(np.log2(m.mix1_amol_per_ul), y, 1)
    res = y - np.polyval(fit, np.log2(m.mix1_amol_per_ul))
    assert got["log2_slope_mix1"] == pytest.approx(fit[0]) and 0.9 < fit[0] < 1.0   # ERCCs without a UMI flatten the low end
    assert rs["estimate"] == pytest.approx(np.polyfit(m.log2_mix1_over_mix2, res, 1)[0])
    assert list(got["subgroups"]) == ["A", "B", "C", "D"]
    for group, fold, log2r in SUBGROUPS:
        g = got["subgroups"][group]
        sel = (m.subgroup == group).to_numpy()
        assert g["log2_ratio"] == float(log2r) and g["n"] == int(sel.sum())
        assert g["mean_residual"] == pytest.approx(res[sel].mean())
        assert g["median_length"] == float(np.median([_lengths(mixes)[i] for i in m.index[sel]]))
    lc = got["length_check"]
    assert lc["test"] == "kruskal" and lc["alpha"] == 0.05 and lc["p"] > 0.05 and not lc["related"]
    assert got["min_per_subgroup"] == 5


def test_ercc_mix_counts_from_mix2():
    mixes = _mix_table()
    got = ercc_mix(_simulate(mixes, "mix2_amol_per_ul", 0), mixes, _lengths(mixes))
    assert got["verdict"] == "Mix 2" and got["reason"] is None
    rs = got["residual_slope"]
    assert rs["lo"] <= -1 <= rs["hi"] and rs["hi"] < 0
    assert got["pearson_log2_mix2"] > got["pearson_log2_mix1"]
    # Mix 2 holds more of D than Mix 1 says and less of A: the residuals fall with the log2 ratio
    means = [got["subgroups"][g]["mean_residual"] for g in "ABCD"]
    assert means[0] < means[1] < means[3] and means[0] < means[2] < means[3]


def test_ercc_mix_verdicts_over_seeds():
    # 20 simulated libraries per mix: no verdict names the other mix. With UMIs proportional to input the right mix is
    # named in 17 (Mix 1) and 15 (Mix 2) of 20, the rest undecided. With log2 UMIs at 0.9 times log2 input the residuals of
    # Mix 2 counts fall by about 0.85 per unit of log2 ratio, and an interval that excludes -1 leaves 16 of 20 undecided
    mixes = _mix_table()
    verdicts = {(slope, column): Counter(ercc_mix(_simulate(mixes, column, 100 + s, slope=slope), mixes, _lengths(mixes, s),
                                                  seed=s)["verdict"] for s in range(20))
                for slope in (1.0, 0.9) for column in ("mix1_amol_per_ul", "mix2_amol_per_ul")}
    for slope in (1.0, 0.9):
        assert verdicts[slope, "mix1_amol_per_ul"]["Mix 2"] == 0 and verdicts[slope, "mix2_amol_per_ul"]["Mix 1"] == 0
        assert verdicts[slope, "mix1_amol_per_ul"]["Mix 1"] >= 15
    assert verdicts[1.0, "mix2_amol_per_ul"]["Mix 2"] >= 15
    assert verdicts[0.9, "mix2_amol_per_ul"]["undecided"] >= 15


def test_ercc_mix_too_few_ercc_is_undecided():
    # every ERCC counted, noise-free Mix 1: only the number of ERCCs per subgroup decides
    mixes = _mix_table(per=4)
    counts = pd.Series(np.round(2.0 ** (1 + np.log2(mixes.mix1_amol_per_ul.to_numpy()) + 10)), index=mixes.ercc_id.to_numpy())
    got = ercc_mix(counts, mixes, _lengths(mixes))
    assert got["n"] == 16 and got["verdict"] == "undecided" and got["reason"] == "too few ERCCs"
    assert {g: v["n"] for g, v in got["subgroups"].items()} == {"A": 4, "B": 4, "C": 4, "D": 4}
    assert got["residual_slope"] is None and got["pearson_log2_mix1"] is None and got["length_check"] is None
    assert got["subgroups"]["A"]["mean_residual"] is None
    # five per subgroup is enough
    mixes = _mix_table(per=5)
    counts = pd.Series(np.round(2.0 ** (np.log2(mixes.mix1_amol_per_ul.to_numpy()) + 10)), index=mixes.ercc_id.to_numpy())
    assert ercc_mix(counts, mixes, _lengths(mixes))["verdict"] == "Mix 1"
    # a subgroup with no ERCC counted
    mixes = _mix_table()
    counts = _simulate(mixes, "mix1_amol_per_ul", 0)
    counts.loc[mixes.ercc_id[mixes.subgroup == "C"]] = 0
    got = ercc_mix(counts, mixes, _lengths(mixes))
    assert got["verdict"] == "undecided" and got["reason"] == "too few ERCCs" and got["subgroups"]["C"]["n"] == 0


def test_ercc_mix_length_artefact_is_undecided():
    # Mix 1 in the tube, but capture grows with length and the subgroups of low log2 ratio hold the long ERCCs (log2 length
    # = 10 - log2 ratio): the residuals fall with the ratio as they would under Mix 2
    mixes = _mix_table()
    rng = np.random.default_rng(5)
    lengths = dict(zip(mixes.ercc_id, np.round(2.0 ** (10 - mixes.log2_mix1_over_mix2.to_numpy()
                                                        + rng.normal(0, 0.1, len(mixes)))).tolist()))
    got = ercc_mix(_simulate(mixes, "mix1_amol_per_ul", 0, length_power=1.0, lengths=lengths), mixes, lengths)
    rs = got["residual_slope"]
    assert rs["lo"] <= -1 <= rs["hi"] and rs["hi"] < 0   # without the guard: "Mix 2"
    assert got["length_check"]["related"] and got["length_check"]["p"] < 1e-6
    assert got["verdict"] == "undecided" and got["reason"] == "subgroup related to length"


def test_ercc_mix_interval_with_both_or_neither_is_undecided():
    mixes = _mix_table()
    lengths = _lengths(mixes)
    # residuals rising with the ratio (neither mix): the interval excludes 0 and -1
    log2_mu = 0.6 + np.log2(mixes.mix1_amol_per_ul.to_numpy()) + mixes.log2_mix1_over_mix2.to_numpy()
    counts = pd.Series(np.random.default_rng(0).poisson(2.0 ** log2_mu), index=mixes.ercc_id.to_numpy())
    got = ercc_mix(counts, mixes, lengths)
    assert got["residual_slope"]["lo"] > 0
    assert got["verdict"] == "undecided" and got["reason"] == "interval excludes 0 and -1"
    # noise so large that the interval holds 0 and -1
    got = ercc_mix(_simulate(mixes, "mix1_amol_per_ul", 0, sd=4.0), mixes, lengths)
    assert got["residual_slope"]["lo"] < -1 and got["residual_slope"]["hi"] > 0
    assert got["verdict"] == "undecided" and got["reason"] == "interval contains 0 and -1"


def test_ercc_mix_bootstrap_is_seeded():
    mixes = _mix_table()
    counts, lengths = _simulate(mixes, "mix1_amol_per_ul", 0), _lengths(mixes)
    a, b = ercc_mix(counts, mixes, lengths), ercc_mix(counts, mixes, lengths)
    assert a == b
    c = ercc_mix(counts, mixes, lengths, n=200, seed=1)
    assert c["residual_slope"]["n_boot"] == 200 and c["residual_slope"]["seed"] == 1
    assert c["residual_slope"]["estimate"] == a["residual_slope"]["estimate"]
    assert (c["residual_slope"]["lo"], c["residual_slope"]["hi"]) != (a["residual_slope"]["lo"], a["residual_slope"]["hi"])


def test_mix_acceptance():
    assert mix_acceptance({"n": 68}, {"ercc_with_concentration": 68}) == {"ercc_with_concentration": 68, "n_equal": True}
    assert mix_acceptance({"n": 67}, {"ercc_with_concentration": 68}) == {"ercc_with_concentration": 68, "n_equal": False}
