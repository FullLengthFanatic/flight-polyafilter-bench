"""Stage 30, step 10 of design 3.8: tecap buckets per long-read site. The accepted reads go to one BAM with their site key in
a tag, `tecap classify` breaks its buckets down by that tag, and its per-tag counts become tables per site and per label.

tecap, `tecap classify --help` (copy in design_m3_draft/design/checks/tecap_help.txt): "--cell-barcode-tag CELL_BARCODE_TAG
Per-cell breakdown via this BAM tag (e.g. CB). Opt-in; no-ops with a warning if the tag is absent." Our observation on
tecap 0.3.2: the breakdown is the key per_cell_metrics of <sample>_terminal_exon.json, one entry per tag value with
total, counts (bucket name to reads) and fractions; only reads that tecap puts into a bucket are counted there.
"""
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pysam

from .config import GTF, POLYASITE

TECAP = Path("/home/simone.picelli/.local/bin/tecap")
# Tag of ours for the site key. Our observation, 2026-10-13: the first 100,000 records of the BD67 and of the BD70 BAM carry
# ML MM NM RG SA bc bl bq bt bx cx di dl dr ds ec ls mg np qe ql qs qt rm rq sn we ws zm and no tag that starts with X.
TAG = "XK"
_KEY = ["ref", "strand", "site"]


def site_key(ref, minus, site):
    return f"{ref}:{'-' if minus else '+'}:{site}"


def merge_parts(parts, out):
    """One indexed BAM from the BAMs of single references (parts: {ref: path}, same header), concatenated in the order of
    the header so that the result is coordinate-sorted; the parts are removed. Returns out."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with pysam.AlignmentFile(str(next(iter(parts.values())))) as fh:
        order = sorted(parts, key=fh.references.index)
    pysam.cat("-o", str(out), *(str(parts[ref]) for ref in order))
    pysam.index(str(out))
    for path in parts.values():
        path.unlink()
    return out


def run_classify(bam, sample, out_dir, threads):
    """`tecap classify` on one BAM with the options of design 3.8, step 10, the buckets broken down by TAG. tecap's own
    output goes to classify.log next to out_dir. Returns the path of <sample>_terminal_exon.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir.parent / "classify.log", "w") as log:
        subprocess.run([str(TECAP), "classify", "--bam", str(bam), "--sample", sample, "--platform", "cdna-pacbio",
                        "--gtf", str(GTF), "--polya-sites", str(POLYASITE), "--cell-barcode-tag", TAG, "--per-gene-table",
                        "--threads", str(threads), "--out-dir", str(out_dir)], check=True, stdout=log, stderr=subprocess.STDOUT)
    return out_dir / f"{sample}_terminal_exon.json"


def site_buckets(path):
    """ref, strand, site, bucket, reads from per_cell_metrics of a <sample>_terminal_exon.json whose tag values are site
    keys, in the order of the file."""
    cells = json.loads(Path(path).read_text())["per_cell_metrics"]
    rows = [(*key.rsplit(":", 2), bucket, n) for key, d in cells.items() for bucket, n in d["counts"].items()]
    return pd.DataFrame(rows, columns=_KEY + ["bucket", "reads"]).astype({"site": np.int64, "reads": np.int64})


def label_buckets(buckets, sites):
    """label, bucket, reads: site_buckets summed over the sites of each label (sites: ref, strand, site, label), buckets by
    falling reads within a label. ValueError when a site key is not among the sites."""
    j = buckets.merge(sites[_KEY + ["label"]], on=_KEY, how="left", validate="many_to_one")
    if j.label.isna().any():
        raise ValueError(f"{j[j.label.isna()].drop_duplicates(_KEY).shape[0]} site keys of tecap are not among the sites")
    t = j.groupby(["label", "bucket"]).reads.sum().reset_index()
    return t.sort_values(["label", "reads", "bucket"], ascending=[True, False, True], ignore_index=True)


def label_counts(sites, by_label):
    """Per label of the sites (label, n_accepted, n_no_splice): the accepted reads written to the BAM, those without a
    CIGAR N, and those tecap put into a bucket (label_buckets)."""
    g = sites.groupby("label")[["n_accepted", "n_no_splice"]].sum()
    classified = by_label.groupby("label").reads.sum()
    return {label: {"written": int(r.n_accepted), "without_splice": int(r.n_no_splice), "classified": int(classified.get(label, 0))}
            for label, r in g.iterrows()}
