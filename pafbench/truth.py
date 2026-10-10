import gzip
from pathlib import Path

import numpy as np
import pandas as pd

from .config import LABEL_CUTOFFS, MEMBER_CAP, SENS_CAPS
from .labels import label_molecules
from .molecules import select_members
from .regions import REGION_NAMES, regions_at

MOL_KEY = ["cb", "ub", "strand", "ref", "genes"]
TRUTH_COLUMNS = ["qname", "mol", "role", "dist5_exon", "dist_mode", "spliced"]
LABELS = ("templated", "nontemplated", "ambiguous")


def _pa_join(mol, pa):
    """pa reads of each molecule: rows with CB and UB, inner-joined to mol on the molecule key (pa row order kept)."""
    d = pa[pa.cb.notna() & pa.ub.notna()]
    return d.merge(mol[["mol"] + MOL_KEY], on=MOL_KEY, how="inner")


def load_called_cell_barcodes(barcodes_path):
    """Read barcodes from filtered_feature_bc_matrix/barcodes.tsv.gz."""
    barcodes = set()
    with gzip.open(barcodes_path, "rt") as f:
        for line in f:
            bc = line.rstrip("\n")
            barcodes.add(bc)
    return barcodes


def mark_in_cell(mol, barcodes_path):
    """Add in_cell boolean column: True if molecule's CB is in the called cells (from filtered_feature_bc_matrix/barcodes.tsv.gz)."""
    if not Path(barcodes_path).exists():
        return mol.assign(in_cell=False)
    called_barcodes = load_called_cell_barcodes(barcodes_path)
    in_cell = mol.cb.isin(called_barcodes).to_numpy(bool)
    return mol.assign(in_cell=in_cell)


def label_truth_molecules(mol, pa, members, segments, cutoffs=LABEL_CUTOFFS):
    """mol plus label, a_band, region at the last cDNA base, gene, has_ts and n_members (members selected at MEMBER_CAP)."""
    out = label_molecules(mol, cutoffs)
    plus = (out.strand == "+").to_numpy()
    last = out.site.to_numpy() - plus
    region = np.empty(len(out), object)
    for (ref, strand), idx in out.groupby(["ref", "strand"], sort=False).indices.items():
        region[idx] = regions_at(segments, ref, strand == "+", last[idx])
    j = _pa_join(mol, pa)
    has_ts = out.mol.isin(j.mol[j.ts > 0]).to_numpy()
    n = select_members(members, MEMBER_CAP).groupby("mol").size()
    n_members = out.mol.map(n).fillna(0).to_numpy(np.int64)
    gene = out.genes.str.split(";", n=1).str[0].to_numpy(object)
    return out.assign(region=region, gene=gene, has_ts=has_ts, n_members=n_members)


def truth_reads(mol, pa, members, cap=max(SENS_CAPS)):
    """Reads that carry a truth label: every pa read of a molecule (role pa), then members selected at cap (role member)."""
    j = _pa_join(mol, pa)
    p = pd.DataFrame({"qname": j.qname, "mol": j.mol.astype(np.int64), "role": "pa", "dist5_exon": np.nan,
                      "dist_mode": "", "spliced": j.spliced.astype(bool)})
    s = select_members(members, cap)
    m = pd.DataFrame({"qname": s.qname, "mol": s.mol.astype(np.int64), "role": "member",
                      "dist5_exon": s.dist5_exon.astype(np.float64), "dist_mode": s.dist_mode, "spliced": s.spliced.astype(bool)})
    return pd.concat([p, m], ignore_index=True)[TRUTH_COLUMNS]


def check_pa_counts(mol, reads):
    """Raise unless every molecule has exactly n_pa pa rows in reads."""
    got = mol.mol.map(reads[reads.role == "pa"].groupby("mol").size()).fillna(0).to_numpy(np.int64)
    bad = got != mol.n_pa.to_numpy()
    if bad.any():
        raise ValueError(f"{int(bad.sum())} molecules do not receive n_pa pa reads (first mol {int(mol.mol.to_numpy()[bad][0])})")


def truth_summary(labeled, reads):
    """Counts printed by stage 12; members within each sensitivity cap come from the member rows of reads."""
    member = reads[reads.role == "member"]
    with_members = labeled[labeled.n_members > 0].label.value_counts()
    out = {
        "molecules": len(labeled),
        "labels": {k: int((labeled.label == k).sum()) for k in LABELS},
        "regions": {k: int((labeled.region == k).sum()) for k in REGION_NAMES},
        "has_ts": int(labeled.has_ts.sum()),
        "truth_reads": {k: int((reads.role == k).sum()) for k in ("pa", "member")},
        "members_within_cap": {str(c): int((member.dist5_exon <= c).sum()) for c in SENS_CAPS},
        "molecules_with_members": {k: int(with_members.get(k, 0)) for k in LABELS},
    }
    if "in_cell" in labeled:
        out["in_cell"] = int(labeled.in_cell.sum())
    return out
