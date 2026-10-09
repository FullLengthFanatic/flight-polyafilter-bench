import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pysam
from scipy import stats

from .evaluate import tag_setting
from .flags import flag_reference, setting_tag
from .grid import published

ID_HEADER = "ERCC ID"
MIX1_HEADER = "concentration in Mix 1 (attomoles/ul)"


def parse_ercc_conc(text):
    """(ercc_id, mix1_amol_per_ul) from Thermo Fisher's ERCC table (ERCC_CONC_URL, tab-separated); columns found by header."""
    t = pd.read_csv(io.StringIO(text), sep="\t", dtype=str)
    t.columns = [c.strip() for c in t.columns]
    return pd.DataFrame({"ercc_id": t[ID_HEADER].str.strip().to_numpy(),
                         "mix1_amol_per_ul": t[MIX1_HEADER].astype(np.float64).to_numpy()})


def ercc_settings(selection):
    """The 8 published settings, then the tags chosen per mode in stage 17's selection.json that are not among them; also
    the chosen tag per mode ({mode: tag or None}; None when the file is absent)."""
    settings = list(published())
    if not Path(selection).exists():
        return settings, None
    modes = json.loads(Path(selection).read_text())["modes"]
    chosen = {mode: m["selection"]["tag"] for mode, m in modes.items()}
    known = {setting_tag(s) for s in settings}
    return settings + [tag_setting(t) for t in dict.fromkeys(chosen.values()) if t and t not in known], chosen


def ercc_reads(bam_path, fasta_path, ref, transcripts, settings, cells):
    """flags.flag_reference on one contig joined by fetch order to (cb, ub) read in a second pass, keeping primary records
    with a called-cell CB and a UB: ref, cb, ub and one bool per setting tag (True = removed)."""
    fl = flag_reference(bam_path, fasta_path, ref, transcripts, settings)
    qname, cb, ub = [], [], []
    with pysam.AlignmentFile(str(bam_path)) as bam:
        for a in bam.fetch(ref):
            qname.append(a.query_name)
            cb.append(a.get_tag("CB") if a.has_tag("CB") else None)
            ub.append(a.get_tag("UB") if a.has_tag("UB") else None)
    assert qname == fl.qname.tolist(), f"{ref}: second pass is not in flag_reference order"
    keep = (((fl.flag.to_numpy() & 0x900) == 0) & np.array([c in cells for c in cb], bool)
            & np.array([u is not None for u in ub], bool))
    tags = [setting_tag(s) for s in settings]
    out = pd.DataFrame({"ref": ref, "cb": np.asarray(cb, object)[keep], "ub": np.asarray(ub, object)[keep]})
    return pd.concat([out, fl.loc[keep, tags].reset_index(drop=True)], axis=1)


def umi_counts(reads, removed=None):
    """UMIs (ref, cb, ub) per ERCC with at least one read not removed (removed None: unfiltered). Indexed by ercc_id
    over every ref in reads, zeros kept, so a filtered count covers the ERCCs detected unfiltered."""
    kept = reads if removed is None else reads[~np.asarray(removed, bool)]
    n = kept.drop_duplicates(["ref", "cb", "ub"]).groupby("ref").size()
    return n.reindex(np.unique(reads.ref.to_numpy()), fill_value=0).astype(np.int64).rename("umis").rename_axis("ercc_id")


def ercc_correlation(counts, conc):
    """Over the ERCCs of counts (UMIs indexed by ercc_id; from umi_counts, so those with a UMI unfiltered) that have a Mix 1
    concentration: pearson of log10(UMI + 1) against log10(concentration), spearman on raw values (both None when either
    side is constant), n."""
    c = conc.set_index("ercc_id").mix1_amol_per_ul
    ids = [i for i in counts.index if i in c.index]
    x, y = c.loc[ids].to_numpy(np.float64), counts.loc[ids].to_numpy(np.float64)
    const = np.ptp(y) == 0 or np.ptp(x) == 0
    return {"n": len(ids),
            "pearson_log10": None if const else float(stats.pearsonr(np.log10(y + 1), np.log10(x))[0]),
            "spearman": None if const else float(stats.spearmanr(y, x)[0])}


def ercc_tables(reads, tags, conc):
    """ercc_counts rows (ercc_id, setting, umis) for 'unfiltered' then each tag, and per setting the correlation, the UMIs
    and the share of unfiltered UMIs kept."""
    base = umi_counts(reads)
    rows, per = [], {}
    for name in ["unfiltered"] + list(tags):
        c = base if name == "unfiltered" else umi_counts(reads, reads[name].to_numpy())
        rows.append(pd.DataFrame({"ercc_id": c.index.to_numpy(), "setting": name, "umis": c.to_numpy()}))
        per[name] = {"correlation": ercc_correlation(c, conc), "umis": int(c.sum()), "umis_kept_share": float(c.sum() / base.sum())}
    return pd.concat(rows, ignore_index=True), per
