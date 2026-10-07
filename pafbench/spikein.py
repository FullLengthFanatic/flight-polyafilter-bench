import re


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
