import re
from collections import defaultdict

import numpy as np
import pandas as pd

from .bd67 import terminal_a_run
from .genes import genes_at
from .sites import PA_COLUMNS, has_hexamer, pa_row, sense_context
from .stream import merge_regions
from .truth import LABELS

MODULES = ("ERCC", "SIRV_E0", "long_SIRV")
SPIKE_LABELS = ("tail", "internal", "other")


def tag_survival(before, after, tags=("CB", "UB", "pa", "ts")):
    """Per tag: among reads present in both and carrying the tag before, how many keep the same value."""
    res = {}
    for t in tags:
        has = [q for q, d in before.items() if t in d and q in after]
        kept = sum(1 for q in has if after[q].get(t) == before[q][t])
        res[t] = {"n": len(has), "kept": kept, "fraction": kept / len(has) if has else None}
    return res


def module_of(contig):
    if contig.startswith("ERCC"):
        return "ERCC"
    if re.fullmatch(r"SIRV[1-7]", contig):
        return "SIRV_E0"
    return "long_SIRV"


def annotated_ends(gtf):
    """{(contig, strd): sorted unique int64 ndarray} of transcript 3' ends in the priming_site convention: the GTF end on
    plus (0-based exclusive end), the GTF start - 1 on minus (0-based last cDNA base)."""
    ends = defaultdict(list)
    with open(gtf) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t")
            if f[2] != "transcript":
                continue
            plus = f[6] == "+"
            ends[(f[0], plus)].append(int(f[4]) if plus else int(f[3]) - 1)
    return {k: np.unique(np.asarray(v, np.int64)) for k, v in ends.items()}


def label_spike_site(site, strd, ends, tail_tol=10, internal_min=30):
    """(label, dist) of a site against the annotated ends of its (contig, strand); d > 0 when the site lies upstream of an
    end in transcript sense. tail: some |d| <= tail_tol; internal: else some d >= internal_min; other: neither, or no ends.
    dist is the d with the smallest |d| (ties: the positive one), None without ends."""
    if ends is None or len(ends) == 0:
        return "other", None
    d = ends - site if strd else site - ends
    if (np.abs(d) <= tail_tol).any():
        label = "tail"
    elif (d >= internal_min).any():
        label = "internal"
    else:
        label = "other"
    return label, int(d[np.lexsort((d < 0, np.abs(d)))[0]])


def spike_pa_row(a, fa, segments):
    """sites.pa_row with the site at the start of the read's terminal A run (bd67.terminal_a_run, one non-A allowed,
    aligned A's included), since spike-in reads keep their poly(A); context, a20, hexamer and genes (at the last cDNA
    base: site - 1 on plus, site on minus) follow that site."""
    row = dict(zip(PA_COLUMNS, pa_row(a, fa, segments)))
    site = terminal_a_run(a, tolerance=1)[0]
    down, up = sense_context(fa, a.reference_name, site, a.is_reverse)
    last = site if a.is_reverse else site - 1
    row.update(site=site, down20=down, up40=up, a20=down.count("A"), hexamer=has_hexamer(up),
               genes=genes_at(segments, a.reference_name, not a.is_reverse, last))
    return [row[c] for c in PA_COLUMNS]


def add_spike_labels(mol, ends):
    """mol plus spike_label and dist_to_end (label_spike_site against ends, float with NaN when the strand has no ends)
    and module (module_of the contig)."""
    res = [label_spike_site(s, st == "+", ends.get((r, st == "+"))) for r, st, s in zip(mol.ref, mol.strand, mol.site)]
    dist = np.array([np.nan if d is None else d for _, d in res], np.float64)
    return mol.assign(spike_label=[lab for lab, _ in res], dist_to_end=dist, module=[module_of(r) for r in mol.ref])


def merge_scans(scans, tags):
    """(truth_flags, region_counts) from per-contig stream.scan_reference results [(truth_df, region_df)], merged as
    stream.merge_parts does: truth rows in the given contig order; region rows by stream.merge_regions."""
    return pd.concat([t for t, _ in scans], ignore_index=True), merge_regions([r for _, r in scans], tags)


def spike_summary(pa, labeled, members, reads):
    """Counts printed by stage 14: pa reads, molecules by module and spike_label, spike_label against the A-count label,
    member candidate rows, truth reads by role."""
    by_mod = labeled.groupby(["module", "spike_label"]).size()
    cross = labeled.groupby(["spike_label", "label"]).size()
    return {
        "pa_reads": len(pa),
        "molecules": len(labeled),
        "by_module": {m: {s: int(by_mod.get((m, s), 0)) for s in SPIKE_LABELS} for m in MODULES},
        "spike_label_by_label": {s: {k: int(cross.get((s, k), 0)) for k in LABELS} for s in SPIKE_LABELS},
        "members": len(members),
        "truth_reads": {k: int((reads.role == k).sum()) for k in ("pa", "member")},
    }
