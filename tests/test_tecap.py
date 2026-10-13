import json
import subprocess

import pandas as pd
import pysam
import pytest

from pafbench import tecap
from pafbench.config import GTF, POLYASITE
from pafbench.tecap import TAG, classify_library, label_buckets, label_counts, merge_parts, run_classify, site_buckets, site_key
from tests.synth_long import HDR, long_read

CAPTURED, EXON = "Captured (>=50%)", "MechB-exon: upstream TE, priming on upstream exon"
INTRONIC = "MechB-aspecific: upstream TE, intronic/flanking"


def test_site_key():
    assert TAG == "XK" and site_key("chr21", False, 7668993) == "chr21:+:7668993" and site_key("chr1", True, 5) == "chr1:-:5"


def _tecap_json(path, cells):
    """<sample>_terminal_exon.json as tecap 0.3.2 writes it, cut down to the keys around per_cell_metrics (our
    observation: design_m3_draft/review/v3/lens_stage30_whole/tecap_site_tag_out/BD70_chr21_site_terminal_exon.json)."""
    per_cell = {k: {"total": sum(c.values()), "counts": c, "fractions": {b: n / sum(c.values()) for b, n in c.items()}}
                for k, c in cells.items()}
    total = {}
    for c in cells.values():
        for b, n in c.items():
            total[b] = total.get(b, 0) + n
    path.write_text(json.dumps({"sample": "S", "platform": "cdna-pacbio", "total": sum(total.values()), "counts": total,
                                "cell_barcode_tag": TAG, "per_cell_metrics": per_cell, "schema_version": "0.1"}, indent=2))
    return path


CELLS = {"chr1:+:100": {CAPTURED: 4, EXON: 1}, "chr1:-:100": {INTRONIC: 2}, "chr2_KI270:1:+:7": {CAPTURED: 1},
         "chr1:+:250": {EXON: 3}}
SITES = pd.DataFrame({"ref": ["chr1", "chr1", "chr1", "chr1", "chr2_KI270:1"], "strand": ["+", "+", "-", "+", "+"],
                      "site": [100, 250, 100, 900, 7],
                      "label": ["templated", "nontemplated", "templated", "ambiguous", "templated"],
                      "n_accepted": [6, 3, 5, 2, 1], "n_no_splice": [1, 0, 3, 2, 0]})


def test_site_buckets_from_per_cell_metrics(tmp_path):
    t = site_buckets(_tecap_json(tmp_path / "S_terminal_exon.json", CELLS))
    assert list(t.columns) == ["ref", "strand", "site", "bucket", "reads"]
    assert t.site.dtype == "int64" and t.reads.dtype == "int64"
    # a reference name may hold a colon: the key is split from the right
    assert list(t.itertuples(index=False, name=None)) == [
        ("chr1", "+", 100, CAPTURED, 4), ("chr1", "+", 100, EXON, 1), ("chr1", "-", 100, INTRONIC, 2),
        ("chr2_KI270:1", "+", 7, CAPTURED, 1), ("chr1", "+", 250, EXON, 3)]
    empty = site_buckets(_tecap_json(tmp_path / "E_terminal_exon.json", {}))
    assert list(empty.columns) == list(t.columns) and not len(empty)


def test_label_buckets_sum_the_sites_of_a_label(tmp_path):
    buckets = site_buckets(_tecap_json(tmp_path / "S_terminal_exon.json", CELLS))
    t = label_buckets(buckets, SITES)
    assert list(t.columns) == ["label", "bucket", "reads"]
    assert list(t.itertuples(index=False, name=None)) == [
        ("nontemplated", EXON, 3), ("templated", CAPTURED, 5), ("templated", INTRONIC, 2), ("templated", EXON, 1)]
    # per label: accepted reads written to the BAM, those without a CIGAR N, and those tecap put into a bucket
    assert label_counts(SITES, t) == {"ambiguous": {"written": 2, "without_splice": 2, "classified": 0},
                                      "nontemplated": {"written": 3, "without_splice": 0, "classified": 3},
                                      "templated": {"written": 12, "without_splice": 4, "classified": 8}}
    # a key that is not a site of the table
    with pytest.raises(ValueError, match="1 site keys of tecap are not among the sites"):
        label_buckets(buckets, SITES[SITES.site != 250])


def test_run_classify_calls_tecap_with_the_options_of_the_design(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: calls.append((cmd, kw)))
    out = tmp_path / "BD70" / "tecap" / "out"
    got = run_classify(tmp_path / "accepted.bam", "BD70", out, 6)
    assert got == out / "BD70_terminal_exon.json" and out.is_dir()
    (cmd, kw), = calls
    assert cmd == [str(tecap.TECAP), "classify", "--bam", str(tmp_path / "accepted.bam"), "--sample", "BD70", "--platform",
                   "cdna-pacbio", "--gtf", str(GTF), "--polya-sites", str(POLYASITE), "--cell-barcode-tag", "XK",
                   "--per-gene-table", "--threads", "6", "--out-dir", str(out)]
    assert str(tecap.TECAP) == "/home/simone.picelli/.local/bin/tecap"
    # tecap's own output goes to a log next to its folder, a failure stops the stage
    assert kw["check"] is True and kw["stdout"].name == str(out.parent / "classify.log") and kw["stderr"] == subprocess.STDOUT


def _parts(tmp_path):
    """Tagged BAMs of two references, given against the order of the header: e1 on chrE, s1 and s2 on chrS."""
    parts = {}
    for ref, names in (("chrE", ["e1"]), ("chrS", ["s1", "s2"])):
        parts[ref] = tmp_path / f"{ref}.bam"
        with pysam.AlignmentFile(str(parts[ref]), "wb", header=HDR) as out:
            for i, name in enumerate(names):
                a = long_read([(1000 + 100 * i, 1090 + 100 * i)], 1080 + 100 * i, name=name)
                a.reference_name = ref
                a.set_tag(TAG, site_key(ref, False, 1080 + 100 * i), "Z")
                out.write(a)
    return parts


def test_merge_parts_concatenates_in_header_order_and_indexes(tmp_path):
    parts = _parts(tmp_path)
    bam = merge_parts(parts, tmp_path / "tecap" / "accepted.bam")
    assert bam == tmp_path / "tecap" / "accepted.bam" and not any(p.exists() for p in parts.values())
    with pysam.AlignmentFile(str(bam)) as fh:
        header = fh.header.to_dict()
        assert {k: header[k] for k in ("HD", "SQ")} == HDR.to_dict() and list(header) == ["HD", "SQ", "PG"]   # cat adds its @PG
        assert [(a.reference_name, a.query_name, a.get_tag(TAG)) for a in fh.fetch(until_eof=True)] == [
            ("chrS", "s1", "chrS:+:1080"), ("chrS", "s2", "chrS:+:1180"), ("chrE", "e1", "chrE:+:1080")]
        assert [a.query_name for a in fh.fetch("chrE")] == ["e1"]


def test_classify_library_merges_runs_tecap_and_writes_both_tables(tmp_path, monkeypatch):
    out, calls = tmp_path / "BD70", []
    sites = pd.DataFrame({"ref": ["chrS", "chrS", "chrE"], "strand": "+", "site": [1080, 1180, 1080],
                          "label": ["templated", "ambiguous", "templated"], "n_accepted": [1, 1, 1], "n_no_splice": [1, 1, 0]})

    def fake(bam, sample, out_dir, threads):
        """Stands in for tecap: the first two tag values of the merged, indexed BAM get reads, the first in two buckets."""
        calls.append((bam, sample, out_dir, threads))
        with pysam.AlignmentFile(str(bam)) as fh:
            keys = [a.get_tag(TAG) for a in fh.fetch()]
        out_dir.mkdir(parents=True)
        return _tecap_json(out_dir / f"{sample}_terminal_exon.json", {keys[0]: {CAPTURED: 2, EXON: 1}, keys[1]: {CAPTURED: 2}})

    monkeypatch.setattr(tecap, "run_classify", fake)
    got = classify_library(_parts(tmp_path), "BD70", out, sites, 3)
    assert calls == [(out / "tecap" / "accepted.bam", "BD70", out / "tecap" / "out", 3)]
    result = out / "tecap" / "out" / "BD70_terminal_exon.json"
    assert pd.read_parquet(out / "tecap_site_buckets.parquet").to_dict("list") == {
        "ref": ["chrS"] * 3, "strand": ["+"] * 3, "site": [1080, 1080, 1180], "bucket": [CAPTURED, EXON, CAPTURED],
        "reads": [2, 1, 2]}
    assert pd.read_parquet(out / "tecap_buckets.parquet").to_dict("list") == {
        "label": ["ambiguous", "templated", "templated"], "bucket": [CAPTURED, CAPTURED, EXON], "reads": [2, 2, 1]}
    assert got.pop("seconds") >= 0
    assert got == {"bam": str(out / "tecap" / "accepted.bam"), "json": str(result), "json_bytes": result.stat().st_size,
                   "sites_with_a_classified_read": 2,
                   "per_label": {"ambiguous": {"written": 1, "without_splice": 1, "classified": 2},
                                 "templated": {"written": 2, "without_splice": 1, "classified": 3}}}
