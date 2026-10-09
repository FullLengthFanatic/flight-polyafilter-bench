import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pyarrow
import pyarrow.parquet as pq
import pysam

from pafbench.config import REPO
from pafbench.exons import load_collapsed_exons
from pafbench.flags import setting_tag
from pafbench.genes import load_gene_segments
from pafbench.grid import all_settings
from pafbench.molecules import MEMBER_COLUMNS, consolidate_sites, member_rows
from pafbench.regions import load_region_segments
from pafbench.sites import PA_COLUMNS
from pafbench.spikein import add_spike_labels, annotated_ends, merge_scans, spike_pa_row
from pafbench.stream import scan_reference
from pafbench.transcripts import load_transcripts
from pafbench.truth import label_truth_molecules, truth_reads


def _stage14(monkeypatch):
    """The stage script as module 'stage14' (registered so that pool workers can unpickle its functions)."""
    spec = importlib.util.spec_from_file_location("stage14", REPO / "scripts" / "14_spikein_truth.py")
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "stage14", mod)
    spec.loader.exec_module(mod)
    return mod


def test_stage14_on_synthetic_spikes(synth, tmp_path, monkeypatch, capsys):
    base, out = tmp_path / "base", tmp_path / "out"
    base.mkdir()
    n_pa = 0
    with pysam.AlignmentFile(str(synth.bam)) as fin, \
            pysam.AlignmentFile(str(base / "star_Aligned.out.bam"), "wb", template=fin) as fout:
        for i, a in enumerate(fin.fetch(until_eof=True)):
            tags = [("CB", "c0"), ("UB", f"u{i // 4}")]  # each pa read (i % 4 == 0) and the next three records share a UMI
            if i % 4 == 0 or a.is_secondary:   # secondaries carry pa too and must not become pa rows
                tags.append(("pa", 20))
                n_pa += not a.is_secondary
            a.set_tags(tags)
            fout.write(a)
    stage = _stage14(monkeypatch)
    monkeypatch.setattr(stage, "SIRV_FASTA", synth.fasta)
    monkeypatch.setattr(stage, "SIRV_GTF", synth.gtf)
    monkeypatch.setattr(sys, "argv", ["14", "SYN", "--base", str(base), "--out", str(out), "--workers", "2"])
    stage.main()
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])

    # the same steps run sequentially, contigs in header order
    bam = out / "star_sorted.bam"
    seg = load_gene_segments(synth.gtf)
    with pysam.AlignmentFile(str(bam)) as fh, pysam.FastaFile(str(synth.fasta)) as fa:
        refs = [s.contig for s in fh.get_index_statistics() if s.mapped > 0]
        pa = pd.DataFrame([spike_pa_row(a, fa, seg) for r in refs for a in fh.fetch(r)
                           if not a.is_secondary and a.has_tag("pa")], columns=PA_COLUMNS)
        pa[["gx", "re"]] = pa[["gx", "re"]].fillna("").astype("str")
        mol = consolidate_sites(pa)
        by_k3 = defaultdict(list)
        for r in mol.itertuples():
            by_k3[(r.cb, r.ub, r.strand)].append((r.mol, r.ref, r.site, r.genes))
        exons = load_collapsed_exons(synth.gtf)
        members = pd.DataFrame([row for r in refs for row in member_rows(fh.fetch(r), r, by_k3, seg, exons)],
                               columns=MEMBER_COLUMNS)
    labeled = add_spike_labels(label_truth_molecules(mol, pa, members, load_region_segments(synth.gtf)),
                               annotated_ends(synth.gtf))
    reads = truth_reads(mol, pa, members)
    settings = all_settings()
    tx = load_transcripts(synth.gtf)
    flags, region = merge_scans([scan_reference(bam, synth.fasta, r, tx, settings, set(reads.qname))[:2] for r in refs],
                                [setting_tag(s) for s in settings])

    assert refs == ["chrA", "chrB"] and len(pa) == n_pa and len(mol) > 20 and len(members) > 20
    for name, exp in (("pa_sites", pa), ("molecules", mol), ("members", members), ("truth/molecules_labeled", labeled),
                      ("truth/truth_reads", reads), ("grid/truth_flags", flags), ("grid/region_counts", region)):
        pd.testing.assert_frame_equal(pd.read_parquet(out / f"{name}.parquet"), exp, check_dtype=False, obj=name)
    assert len(flags) == reads.qname.nunique() and flags.shape[1] == 4 + 396
    assert printed["pa_reads"] == n_pa and printed["molecules"] == len(mol)
    assert printed["truth_reads"] == {"pa": int((reads.role == "pa").sum()), "member": int((reads.role == "member").sum())}
    assert sum(sum(v.values()) for v in printed["by_module"].values()) == len(mol)
    # the spike BAM has no GX or RE tags: string columns of "", not null-typed parquet columns
    for name, cols in (("pa_sites", ["gx", "re"]), ("molecules", ["re"]), ("truth/molecules_labeled", ["re"])):
        schema, df = pq.read_schema(out / f"{name}.parquet"), pd.read_parquet(out / f"{name}.parquet")
        for c in cols:
            assert schema.field(c).type in (pyarrow.string(), pyarrow.large_string()) and set(df[c]) == {""}, (name, c)

    # the sorted BAM is kept when its index exists and is newer than star_Aligned.out.bam
    mtime = bam.stat().st_mtime_ns
    stage.main()
    assert bam.stat().st_mtime_ns == mtime
    # and sorted again when star_Aligned.out.bam is newer than the index (scripts/09_spikein_star.sh run again)
    src, bai = base / "star_Aligned.out.bam", Path(f"{bam}.bai")
    old = src.stat().st_mtime_ns - 10**9
    os.utime(bai, ns=(old, old))
    stage.main()
    assert bam.stat().st_mtime_ns != mtime and bai.stat().st_mtime_ns > src.stat().st_mtime_ns


def _contig(job):
    return job[1]


def test_per_contig_returns_header_order(monkeypatch):
    stage = _stage14(monkeypatch)
    stats = [SimpleNamespace(contig="a", mapped=1), SimpleNamespace(contig="b", mapped=5), SimpleNamespace(contig="c", mapped=3)]
    assert stage._per_contig(_contig, "x.bam", stats, 2) == ["a", "b", "c"]
