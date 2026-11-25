"""Stage 29b: the internal-priming share under the conventional rule (Balázs et al. 2019; SQANTI3's default is its
12-of-20 half, Pardo-Palacios et al. 2024), for comparison with the 8/4 labels. A molecule is flagged when the 20 nt
window downstream of its site (down20, transcript sense) holds RUN_MIN or more consecutive adenines anywhere, or
COUNT_MIN or more adenines in all. This is scNoiseMeter's `either` criterion at the benchmark site. The rule is binary:
no ambiguous class."""
import numpy as np

RUN_MIN = 6
COUNT_MIN = 12


def conventional_flag(mol):
    """True for the molecules the conventional rule calls internally primed (down20 and a20 of molecules_labeled)."""
    return (mol.down20.str.contains("A" * RUN_MIN) | (mol.a20 >= COUNT_MIN)).to_numpy(bool)


def conventional_share(mol):
    """n molecules, n_conventional flagged by the rule, share = n_conventional / n, and share_all_8_4 = templated / n,
    the share_all of stage 18 on the same molecules. Every molecule is in the denominator, whatever its label."""
    n = len(mol)
    n_conventional = int(np.count_nonzero(conventional_flag(mol)))
    return {"n": n, "n_conventional": n_conventional, "share": n_conventional / n,
            "share_all_8_4": int((mol.label.to_numpy() == "templated").sum()) / n}
