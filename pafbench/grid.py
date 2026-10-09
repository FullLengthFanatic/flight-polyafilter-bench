from .flags import Setting

COVLENS = (150, 200, 250, 300, 375, 450, 600)
MINLENS = (3, 5, 7, 9, 10, 11, 13, 15, 20)
MISMS = (0, 1, 2)
PUBLISHED = ((300, 10, 1), (275, 9, 1), (180, 11, 2), (375, 3, 2))


def grid():
    """392 settings: per mode and COVLEN, the remove-all control (MINSNRLEN None) then every MINSNRLEN x MISM."""
    out = []
    for introns in (False, True):
        for c in COVLENS:
            out.append(Setting(c, None, 0, introns))
            out.extend(Setting(c, l, m, introns) for l in MINLENS for m in MISMS)
    return out


def published():
    """README example and the paper's Fig. 4A optima, each in both modes."""
    return [Setting(c, l, m, introns) for c, l, m in PUBLISHED for introns in (False, True)]


def all_settings():
    """grid() followed by the published settings it lacks (396)."""
    out = grid()
    seen = set(out)
    out.extend(s for s in published() if s not in seen)
    return out
