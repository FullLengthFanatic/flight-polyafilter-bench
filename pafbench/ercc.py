import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pysam
from scipy import stats

from .config import ERCC_LENGTH_ALPHA, ERCC_MIX_MIN
from .evaluate import interval, resample_counts, tag_setting
from .flags import flag_reference, setting_tag
from .grid import published

ID_HEADER = "ERCC ID"
MIX1_HEADER = "concentration in Mix 1 (attomoles/ul)"
MIX2_HEADER = "concentration in Mix 2 (attomoles/ul)"
SUBGROUP_HEADER = "subgroup"
LOG2_RATIO_HEADER = "log2(Mix 1/Mix 2)"


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


def parse_ercc_mixes(text):
    """(ercc_id, subgroup, mix1_amol_per_ul, mix2_amol_per_ul, log2_mix1_over_mix2) from Thermo Fisher's ERCC table
    (ERCC_CONC_URL); columns found by header. The ratio is the table's column "log2(Mix 1/Mix 2)", Mix 1 over Mix 2: 2 in
    subgroup A, whose Mix 1 holds 4 times the Mix 2 amount (30000 against 7500 amol/ul for ERCC-00130)."""
    t = pd.read_csv(io.StringIO(text), sep="\t", dtype=str)
    t.columns = [c.strip() for c in t.columns]
    return pd.DataFrame({"ercc_id": t[ID_HEADER].str.strip().to_numpy(),
                         "subgroup": t[SUBGROUP_HEADER].str.strip().to_numpy(),
                         "mix1_amol_per_ul": t[MIX1_HEADER].astype(np.float64).to_numpy(),
                         "mix2_amol_per_ul": t[MIX2_HEADER].astype(np.float64).to_numpy(),
                         "log2_mix1_over_mix2": t[LOG2_RATIO_HEADER].astype(np.float64).to_numpy()})


def residual_slope(x, r, y, w):
    """Per row of the weights w ([draws, ERCCs]; bootstrap draw counts, or a row of ones for ordinary least squares): the
    slope b of y on x, the residuals e of that fit for every ERCC ([draws, ERCCs]) and the slope s of e on r, all fits
    weighted by the row."""
    sw = w.sum(axis=1, keepdims=True)
    dx, dy, dr = (v - (w @ v)[:, None] / sw for v in (x, y, r))
    b = (w * dx * dy).sum(axis=1) / (w * dx * dx).sum(axis=1)
    e = dy - b[:, None] * dx
    return b, e, (w * dr * e).sum(axis=1) / (w * dr * dr).sum(axis=1)


def ercc_mix(counts, mixes, lengths, n=1000, seed=0):
    """Which ERCC mix is in the tube (design 3.10, step 1), over the ERCCs of counts (unfiltered UMIs indexed by ercc_id)
    with a UMI and a row in mixes (parse_ercc_mixes), as stage 15's ercc_with_concentration: log2 UMIs regressed on log2
    Mix 1, then the slope of the residuals on the subgroup's log2(Mix 1/Mix 2), with a bootstrap 95% interval over ERCCs
    (evaluate.resample_counts draws, both fits redone in each). Verdict "Mix 1" when the interval contains 0 and not -1,
    "Mix 2" when it excludes 0 and reaches the Mix 2 range, from -1 to minus the fitted log2 slope on Mix 1, else
    "undecided"; also "undecided", with the reason, when a subgroup has fewer than ERCC_MIX_MIN ERCCs (statistics then None)
    or when ERCC length (lengths: ercc_id to nt) differs between the subgroups (Kruskal-Wallis p < ERCC_LENGTH_ALPHA), so
    that a capture that depends on length cannot pass for a mix."""
    m = mixes.set_index("ercc_id")
    ids = [i for i in counts.index if counts[i] > 0 and i in m.index]
    t = m.loc[ids]
    groups = sorted(m.subgroup.unique())
    sub = t.subgroup.to_numpy()
    ratio_of = m.groupby("subgroup").log2_mix1_over_mix2.first()
    out = {"n": len(ids), "min_per_subgroup": ERCC_MIX_MIN, "pearson_log2_mix1": None, "pearson_log2_mix2": None,
           "log2_slope_mix1": None, "residual_slope": None,
           "subgroups": {g: {"log2_ratio": float(ratio_of[g]), "n": int((sub == g).sum()), "mean_residual": None,
                             "median_length": None} for g in groups},
           "length_check": None, "verdict": "undecided", "reason": "too few ERCCs"}
    if min(v["n"] for v in out["subgroups"].values()) < ERCC_MIX_MIN:
        return out
    y = np.log2(counts[ids].to_numpy(np.float64))
    x1, x2 = np.log2(t.mix1_amol_per_ul.to_numpy()), np.log2(t.mix2_amol_per_ul.to_numpy())
    r = t.log2_mix1_over_mix2.to_numpy()
    length = np.array([lengths[i] for i in ids], np.float64)
    # r is log2(Mix 1/Mix 2). Under Mix 2 an ERCC holds its Mix 1 amount divided by 2 ** r, so log2 input = log2 Mix 1 - r:
    # with UMIs proportional to input the residuals of the Mix 1 fit fall by r, a slope of -1, and with log2 UMIs growing by
    # b per log2 of input they fall by about b r, a slope of about -b (hence the Mix 2 range from -1 to -b); under Mix 1 they
    # do not depend on r, a slope of 0
    b, e, s = residual_slope(x1, r, y, np.ones((1, len(ids))))
    lo, hi, n_boot = interval(residual_slope(x1, r, y, resample_counts(len(ids), n, seed).astype(np.float64))[2])
    h, p = stats.kruskal(*[length[sub == g] for g in groups])
    out |= {"pearson_log2_mix1": float(stats.pearsonr(y, x1)[0]), "pearson_log2_mix2": float(stats.pearsonr(y, x2)[0]),
            "log2_slope_mix1": float(b[0]),
            "residual_slope": {"estimate": float(s[0]), "lo": lo, "hi": hi, "n_boot": n_boot, "seed": seed},
            "length_check": {"test": "kruskal", "h": float(h), "p": float(p), "alpha": ERCC_LENGTH_ALPHA,
                             "related": bool(p < ERCC_LENGTH_ALPHA)}}
    for g in groups:
        sel = sub == g
        out["subgroups"][g] |= {"mean_residual": float(e[0, sel].mean()), "median_length": float(np.median(length[sel]))}
    has0, has1 = lo <= 0 <= hi, lo <= -1 <= hi
    m2 = sorted((-1.0, -float(b[0])))
    if out["length_check"]["related"]:
        out["reason"] = "subgroup related to length"
    elif has0 and not has1:
        out["verdict"], out["reason"] = "Mix 1", None
    elif not has0 and lo <= m2[1] and hi >= m2[0]:
        out["verdict"], out["reason"] = "Mix 2", None
    else:
        out["reason"] = "interval contains 0 and -1" if has0 else "interval excludes 0 and the Mix 2 range"
    return out


def mix_acceptance(mix, ercc):
    """Acceptance of stage 32 (design 3.10): its n equals ercc_with_concentration of stage 15's ercc.json."""
    return {"ercc_with_concentration": ercc["ercc_with_concentration"], "n_equal": mix["n"] == ercc["ercc_with_concentration"]}
