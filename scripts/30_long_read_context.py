"""Stage 30: context of long-read (BD67, BD70) 3' ends against the GTF, with the pigeon cross-check for BD70, the link to
1179 with its null, and tecap buckets per site (design of 2026-10-12, section 3.8).

Reads and sites are those of stage 16 (same references by default, same acceptance); each accepted read gets one class by
pafbench.lrcontext.classify. Inputs are read under --base: <lib>/long_read_sites.parquet of stage 16 and
1179/truth/molecules_labeled.parquet. Outputs go under --out (default --base):

<lib>/isoform_context_sites.parquet    one row per site with an accepted read (lrcontext.CONTEXT_COLUMNS). The n_<class>
                                       columns count the reads placed by their 3' part (chain_novel) under their class;
                                       context is the top class of that "placed" version, context_without_chain_novel of
                                       the "strict" one, which counts them as no_model_novel_junction; top_share and tied
                                       belong to context. share_spliced: reads with a matched intron; median_length: median
                                       query length of the records, clips included.
<lib>/isoform_context_summary.parquet  version, label, a_band, context, sites, reads, share_spliced, share_5p_complete:
                                       sites whose top class is the context, and reads whose own class it is with the
                                       shares among those reads (lrcontext.context_tables).
<lib>/region_by_context.parquet        version, region, context, sites, reads: the same by milestone 2 region.
<lib>/sites_no_model.parquet           ref, strand, site, region, label, n_accepted, reason, sub_cause: the sites whose
                                       placed top class is a no_model reason (novel_junction, antisense, intergenic), with
                                       the main sub-cause of a novel junction.
<lib>/isoform_context.json             totals in both versions (lrcontext.totals), record counts, the acceptance checks,
                                       the link to 1179 and its null, per reference counts and seconds, for BD70 the
                                       counts of the pigeon cross-check, and under "tecap" the counts of that step.
BD70/pigeon_crosscheck.parquet         structural_category, subcategory, context, isoforms: the ends of the min20 collapsed
                                       isoforms against the nearest site within 10 nt ("none" without one).
1179/truth/long_read_context.parquet   mol, context_<lib>: the placed top class of the long-read site within 10 nt of each
                                       truth molecule on the references scanned ("none" without one).
<lib>/tecap/accepted.bam (indexed)     the accepted reads with the site key ref:strand:site in tag XK, <lib>/tecap/out/ with
                                       tecap's own files and <lib>/tecap/classify.log with its output.
<lib>/tecap_site_buckets.parquet       ref, strand, site, bucket, reads, and <lib>/tecap_buckets.parquet: label, bucket, reads.
                                       tecap runs after every other output of both libraries is written; the key "tecap"
                                       is then added to isoform_context.json.

The stage stops before writing any table when the sites or their accepted reads differ from stage 16, or when a site of the
region intron has the top class no_model_antisense or no_model_intergenic; such sites are listed. The tagged BAMs of the
references (<lib>/tecap/part.<ref>.bam) are written during the scan and are then left behind.

The last line printed is JSON: per library the accepted reads and sites; per version the novel-junction reads, the sites with a
no_model top class, among them those of the region intron, the tied sites and the sites whose top class holds at most half of
the reads; the sites whose top class rests on chain_novel reads alone; the matched share and the label agreement of the 1179
molecules by label, real and with the sites moved NULL_SHIFT nt upstream; and per label the reads written to the tagged BAM,
those without a splice and those tecap classified.

Options: --libs, --refs (default: the 23 references of stage 16), --workers (tecap gets as many, at most one per reference:
`tecap classify --help` says "--threads THREADS  Number of worker processes (contig-level parallelism)."), --base, --out,
--no-tecap (no tagged BAM and no tecap run: the context tables alone).
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import pysam

from pafbench.config import (FIVE_PRIME_TOL, GTF, LONG_END_TOL, LONG_READS, MIN_INTRON, NEAR_INTRON_TOL, NULL_SHIFT, OUT,
                             PIGEON_DIR)
from pafbench.lrcontext import (CONTEXT_COLUMNS, NOVEL_JUNCTION, VERSIONS, attach_stage16, context_tables, intron_sites_off_model,
                                isoform_ends, link_1179, load_models, pigeon_crosscheck, reduce_sites, scan_context, totals)
from pafbench.pool import map_forked
from pafbench.tecap import classify_library

REFS = [f"chr{i}" for i in range(1, 23)] + ["chrX"]
PIGEON_GTF = PIGEON_DIR / "20260608_FLIGHT-seq_horg_BD70_BD70.trimmed.min20count.gtf"
PIGEON_CLASS = PIGEON_DIR / "classification" / "BD70.trimmed.min20count_classification.txt"
RECORD_KEYS = ("records", "secondary_supplementary", "low_mapq", "used", "accepted", "empty_body")
_MODELS = None


def _init(models):
    global _MODELS
    _MODELS = models


def _work(job):
    lib, ref, part = job
    t0 = time.monotonic()
    reads, counts = scan_context(LONG_READS[lib], ref, _MODELS[ref], tagged=part)
    return (lib, ref, *reduce_sites(reads, ref), counts | {"seconds": time.monotonic() - t0})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--libs", nargs="+", default=list(LONG_READS), choices=list(LONG_READS))
    ap.add_argument("--refs", nargs="+", default=REFS)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--base", type=Path, default=OUT, help="folder with the outputs of stages 12 and 16")
    ap.add_argument("--out", type=Path, default=None, help="output folder (default: --base)")
    ap.add_argument("--no-tecap", action="store_true", help="skip the tagged BAM and the tecap run")
    args = ap.parse_args()
    out = args.out or args.base
    t0 = time.time()
    size = {}
    for lib in args.libs:
        with pysam.AlignmentFile(str(LONG_READS[lib])) as fh:
            size |= {(lib, s.contig): s.mapped for s in fh.get_index_statistics()}
        if not args.no_tecap:
            (out / lib / "tecap").mkdir(parents=True, exist_ok=True)
    jobs = sorted(((lib, ref, None if args.no_tecap else out / lib / "tecap" / f"part.{ref}.bam")
                   for lib in args.libs for ref in args.refs), key=lambda j: -size.get(j[:2], 0))
    res = map_forked(_work, jobs, min(args.workers, len(jobs)), initializer=_init, initargs=(load_models(GTF, args.refs),))
    scans = {}
    for lib in args.libs:
        parts = sorted((r for r in res if r[0] == lib), key=lambda r: args.refs.index(r[1]))
        s16 = pd.read_parquet(args.base / lib / "long_read_sites.parquet")
        try:
            sites = attach_stage16(pd.concat([p[2] for p in parts], ignore_index=True), s16[s16.ref.isin(args.refs)])
        except ValueError as e:
            raise SystemExit(f"{lib}: {e}")
        off = intron_sites_off_model(sites)
        if len(off):
            listing = off[["ref", "strand", "site", "n_accepted", "context", "context_without_chain_novel"]].to_string(
                index=False)
            raise SystemExit(f"{lib}: {len(off)} sites of the region intron with a top class no_model_antisense or "
                             f"no_model_intergenic\n{listing}")
        scans[lib] = (sites, pd.concat([p[3] for p in parts], ignore_index=True), parts)
    mol = pd.read_parquet(args.base / "1179" / "truth" / "molecules_labeled.parquet",
                          columns=["mol", "ref", "strand", "site", "label"])
    mol = mol[mol.ref.isin(args.refs)].reset_index(drop=True)
    link, line, summaries = mol[["mol"]].copy(), {}, {}
    for lib, (sites, by_class, parts) in scans.items():
        d = out / lib
        d.mkdir(parents=True, exist_ok=True)
        sites[CONTEXT_COLUMNS].to_parquet(d / "isoform_context_sites.parquet", index=False)
        context_tables(sites, by_class, ["label", "a_band"]).to_parquet(d / "isoform_context_summary.parquet", index=False)
        context_tables(sites, by_class, ["region"])[["version", "region", "context", "sites", "reads"]].to_parquet(
            d / "region_by_context.parquet", index=False)
        no_model = sites[sites.context.str.startswith("no_model_")]
        no_model.assign(reason=no_model.context.str.removeprefix("no_model_"))[
            ["ref", "strand", "site", "region", "label", "n_accepted", "reason", "sub_cause"]].to_parquet(
            d / "sites_no_model.parquet", index=False)
        summary = {"library": lib, "bam": str(LONG_READS[lib]), "gtf": str(GTF), "refs": args.refs,
                   "tolerances": {"LONG_END_TOL": LONG_END_TOL, "NEAR_INTRON_TOL": NEAR_INTRON_TOL, "MIN_INTRON": MIN_INTRON,
                                  "FIVE_PRIME_TOL": FIVE_PRIME_TOL, "NULL_SHIFT": NULL_SHIFT},
                   "records": {k: sum(p[4][k] for p in parts) for k in RECORD_KEYS}} | totals(sites, by_class)
        # attach_stage16 and intron_sites_off_model have passed; the class counts of a site add up to its reads by construction
        summary["acceptance"] = {"sites_and_accepted_reads_equal_stage16": True,
                                 "one_class_per_accepted_read": summary["accepted_reads"] == summary["records"]["accepted"],
                                 "intron_sites_off_model": 0}
        link[f"context_{lib}"], summary["vs_1179"] = link_1179(mol, sites)
        if lib == "BD70":
            classification = pd.read_csv(PIGEON_CLASS, sep="\t", usecols=["isoform", "structural_category", "subcategory"])
            table, counts = pigeon_crosscheck(isoform_ends(PIGEON_GTF, args.refs), classification, sites)
            table.to_parquet(d / "pigeon_crosscheck.parquet", index=False)
            summary["pigeon"] = {"gtf": str(PIGEON_GTF), "classification": str(PIGEON_CLASS)} | counts
        line[lib] = {"accepted_reads": summary["accepted_reads"], "sites": summary["sites"],
                     "reads_novel_junction": {v: summary[v]["reads_by_class"][NOVEL_JUNCTION] for v in VERSIONS},
                     "sites_top_no_model": {v: sum(n for c, n in summary[v]["sites_by_context"].items()
                                                   if c.startswith("no_model_")) for v in VERSIONS},
                     "intron_sites": summary["intron_sites"]}
        line[lib] |= {k: {v: summary[v][k] for v in VERSIONS}
                      for k in ("intron_sites_top_no_model", "sites_tied", "sites_top_share_at_most_half")}
        line[lib]["sites_top_class_on_chain_novel_only"] = summary["sites_top_class_on_chain_novel_only"]
        line[lib]["vs_1179"] = {k: {x: summary["vs_1179"][k][x] for x in ("share_matched_by_label", "agreement")}
                                for k in ("real", "null")}
        summary["per_ref"] = {p[1]: p[4] for p in parts}
        (d / "isoform_context.json").write_text(json.dumps(summary, indent=1))
        summaries[lib] = summary
    (out / "1179" / "truth").mkdir(parents=True, exist_ok=True)
    link.to_parquet(out / "1179" / "truth" / "long_read_context.parquet", index=False)
    if not args.no_tecap:   # last, so that a failure of tecap leaves the context outputs complete
        for lib, (sites, _, parts) in scans.items():
            d = out / lib
            summaries[lib]["tecap"] = classify_library({p[1]: d / "tecap" / f"part.{p[1]}.bam" for p in parts}, lib, d, sites,
                                                       min(args.workers, len(args.refs)))
            line[lib]["tecap"] = summaries[lib]["tecap"]["per_label"]
            (d / "isoform_context.json").write_text(json.dumps(summaries[lib], indent=1))
    print(json.dumps(line | {"seconds": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
