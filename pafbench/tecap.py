"""Stage 30, step 10 of design 3.8: tecap buckets per long-read site. The accepted reads go to one BAM with their site key in
a tag, `tecap classify` breaks its buckets down by that tag, and its per-tag counts become tables per site and per label.

tecap 0.4.0 (`tecap --version`). `tecap classify --help` (copy in design_m3_draft/design/checks/tecap_help.txt):
"--cell-barcode-tag CELL_BARCODE_TAG  Per-cell breakdown via this BAM tag (e.g. CB). Opt-in; no-ops with a warning if the tag
is absent." Its source, src/tecap/bam.py: per_cell_summary returns the "Compact per-cell dict: {cb: {counts: {cat: n}, total:
N, fractions: ...}}", which cli.py writes as per_cell_metrics of <sample>_terminal_exon.json. A read is counted under its tag
value once it has a bucket (lines 200 to 203); a read without a CIGAR N, or that overlaps no gene, is skipped before that
(lines 149 to 154 and 167 to 168).
"""
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pysam

from .config import GTF, POLYASITE

TECAP = Path("/home/simone.picelli/.local/bin/tecap")
# Tag of ours for the site key. Our observation, 2026-10-13: the first 100,000 records of the BD67 BAM carry ML MM NM RG SA
# bc bl bq bt bx cx di dl dr ds ec ls mg np qe ql qs qt rm rq sn we ws zm and no tag that starts with X.
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
    return {label: {"written": int(r.n_accepted), "without_splice": int(r.n_no_splice),
                    "classified": int(classified.get(label, 0))} for label, r in g.iterrows()}


def classify_library(parts, sample, out, sites, threads):
    """The tecap step of one library: the tagged BAMs of its references (parts: {ref: path}) become out/tecap/accepted.bam
    (merge_parts), tecap runs on it (run_classify, output in out/tecap/out), and its buckets are written per site to
    out/tecap_site_buckets.parquet and per label of the sites (ref, strand, site, label, n_accepted, n_no_splice) to
    out/tecap_buckets.parquet. Returns the paths of the BAM and of tecap's JSON, the size of that JSON, the sites with a
    classified read, label_counts and the seconds taken."""
    t0 = time.time()
    bam = merge_parts(parts, out / "tecap" / "accepted.bam")
    result = run_classify(bam, sample, out / "tecap" / "out", threads)
    buckets = site_buckets(result)
    by_label = label_buckets(buckets, sites)
    buckets.to_parquet(out / "tecap_site_buckets.parquet", index=False)
    by_label.to_parquet(out / "tecap_buckets.parquet", index=False)
    return {"bam": str(bam), "json": str(result), "json_bytes": result.stat().st_size,
            "sites_with_a_classified_read": len(buckets.drop_duplicates(_KEY)), "per_label": label_counts(sites, by_label),
            "seconds": time.time() - t0}
