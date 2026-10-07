import numpy as np
import pandas as pd


def hexamer_curve(mol):
    g = mol.groupby("a20").hexamer.agg(["size", "mean"])
    g = g.reindex(range(21), fill_value=0)
    return pd.DataFrame({"a20": g.index, "n": g["size"].to_numpy(), "rate": g["mean"].to_numpy()})


def _pooled(c, lo, hi):
    sel = c[(c.a20 >= lo) & (c.a20 <= hi)]
    return float((sel.rate * sel.n).sum() / sel.n.sum())


def choose_cutoffs(curve, frac=0.25, min_n=200):
    tail = _pooled(curve, 0, 4)
    bg = _pooled(curve, 18, 20)
    span = tail - bg
    if span <= 0:
        raise ValueError("hexamer rate is not higher at low A counts; cut-offs cannot be calibrated")
    ok = curve[curve.n >= min_n]
    templ = next(k for k in range(22) if bool((ok[ok.a20 >= k].rate <= bg + frac * span).all()))
    nont = max(k for k in range(-1, 21) if bool((ok[ok.a20 <= k].rate >= tail - frac * span).all()))
    if templ > 20 or nont < 0 or nont >= templ:
        raise ValueError(f"no usable cut-offs (templated_min={templ}, nontemplated_max={nont})")
    return {"templated_min": int(templ), "nontemplated_max": int(nont), "tail_rate": tail,
            "background_rate": bg, "frac": frac, "min_n": min_n}


def templated_share(mol, templated_min, nontemplated_max):
    cls = np.where(mol.a20 >= templated_min, "templated", np.where(mol.a20 <= nontemplated_max, "nontemplated", "ambiguous"))
    t = mol.assign(cls=cls).groupby("re").cls.value_counts().unstack(fill_value=0)
    for col in ("templated", "nontemplated", "ambiguous"):
        if col not in t:
            t[col] = 0
    t["share"] = t.templated / (t.templated + t.nontemplated)
    return t


def compare_shares(a, b, threshold=0.05):
    j = a[["share"]].join(b[["share"]], lsuffix="_a", rsuffix="_b", how="inner")
    j["diff"] = j.share_b - j.share_a
    j["flag"] = j["diff"].abs() > threshold
    return j
