"""Stage 28: selection bias, labeled against unlabeled molecules (design of 2026-10-12, section 3.6).

Population: the molecules of called cells that the counting rule of stage 24 counts before filtering
(recount/counted_molecules.parquet, one row per UMI of M0). A counted molecule is labeled when a truth molecule joins it
(eval/truth_to_counted.parquet of stage 27, joined): the join goes through the read group (cell, UMI, GX of a pa read), so
the gene is GX, not our own gene. The pass's n_pa above 0 is a second definition of labeled, cross-tabulated with it.

Five axes (AXES), each a table of category shares among labeled and among unlabeled molecules, and a record table by RE.
An axis fires when a share differs by more than BIAS_POINTS points between the two groups (rule); what that decides is
decision.
"""
import numpy as np
import pandas as pd

from .config import BIAS_POINTS

AXES = ("region", "cell_class", "expr_bin", "position", "family_size")   # axes 1 to 5 of design 3.6, in its order
REWEIGHT_AXES = ("region", "cell_class", "expr_bin", "family_size")       # the merge order of the pooling
REGIONS = ("E", "N", "EN")      # re_class 1, 2 and 3 of stage 23: exonic records only, intronic only, both
FAMILY = ("1", "2", "3+")       # n_reads
RULE_STRATUM = {"position": "1"}  # the rule reads axis 4 among molecules with one read without a pa tag, the others in "all"
AXIS_COLUMNS = ["lib", "axis", "stratum", "category", "n_labeled", "n_unlabeled", "share_labeled", "share_unlabeled",
                "diff_points"]
# the accuracy table of a fired axis that moves from the details block into the main text (design 3.6, point 3)
PLACEMENT = {"region": "eval/metrics.parquet", "cell_class": "eval/strata_m3.parquet", "expr_bin": "eval/strata_m3.parquet"}
_TOL = 1e-9  # a difference of exactly BIAS_POINTS, up to floating point rounding, does not fire


def _share(n):
    n = np.asarray(n, np.float64)
    total = n.sum()
    return n / total if total else np.full(len(n), np.nan)


def axis_rows(lib, axis, stratum, categories, n_labeled, n_unlabeled):
    """Rows of one axis and stratum in AXIS_COLUMNS: molecules per category in each group, the category's share of the
    group (NaN when the group is empty) and the difference share_labeled - share_unlabeled in percentage points."""
    sl, su = _share(n_labeled), _share(n_unlabeled)
    return pd.DataFrame({"lib": lib, "axis": axis, "stratum": stratum, "category": list(categories),
                         "n_labeled": np.asarray(n_labeled, np.int64), "n_unlabeled": np.asarray(n_unlabeled, np.int64),
                         "share_labeled": sl, "share_unlabeled": su, "diff_points": 100 * (sl - su)})


def _largest(diff, names, points):
    d = np.abs(np.asarray(diff, np.float64))
    if not np.isfinite(d).any():
        return {"fired": False, "max_abs_diff_points": None}, None
    k = int(np.nanargmax(d))
    return {"fired": bool(d[k] - points > _TOL), "max_abs_diff_points": float(d[k])}, names[k]


def rule(table, records, points=BIAS_POINTS):
    """The 5 point rule of design 3.6 on one library's axis table (AXIS_COLUMNS) and record table (records_table, or None).
    An axis fires when a category of its rule stratum (RULE_STRATUM, else "all") differs by more than points between labeled
    and unlabeled molecules; the record table fires when the share of labeled molecules of an RE differs by more than points
    from the share of duplicate-free units (diff_points_units). Per axis found in table: the stratum read, fired, the
    largest |diff_points| and its category; fired lists the fired axes in the order of AXES."""
    out = {"axes": {}}
    for axis in [a for a in AXES if a in set(table.axis)]:
        stratum = RULE_STRATUM.get(axis, "all")
        t = table[(table.axis == axis) & (table.stratum == stratum)]
        res, cat = _largest(t.diff_points, t.category.tolist(), points)
        out["axes"][axis] = {"stratum": stratum, **res, "category": cat}
    out["fired"] = [a for a, r in out["axes"].items() if r["fired"]]
    if records is not None:
        res, re = _largest(records.diff_points_units, records.re.tolist(), points)
        out["records"] = {**res, "re": re}
    return out


def decision(fired, records_fired):
    """What the fired axes (names of AXES) and the record table decide, and nothing else (design 3.6, points 1 to 3):
    wording, every headline sentence on burden and accuracy says "among molecules with a polyA-trimmed read" and names
    axes_named; burden_numbers, the headline and the unweighted share among counted labeled molecules always, the share
    reweighted over reweight_axes (the fired axes among 1, 2, 3 and 5) when there is one; main_text_accuracy, the accuracy
    tables of the fired axes among region, cell class and expression that move into the main text."""
    named = [a for a in AXES if a in fired]
    reweight = [a for a in REWEIGHT_AXES if a in fired]
    return {"wording": bool(named) or bool(records_fired), "axes_named": named + (["records"] if records_fired else []),
            "burden_numbers": ["headline", "counted_unweighted"] + (["counted_reweighted"] if reweight else []),
            "reweight_axes": reweight,
            "main_text_accuracy": [{"axis": a, "table": PLACEMENT[a], "stratum_kind": a} for a in named if a in PLACEMENT]}
