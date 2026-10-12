import sys


def polyafilter_modules():
    """Import the pinned, unmodified polyAfilter modules."""
    # imported here: config builds IMPACT_SETTINGS from flags, and flags imports this module through transcripts
    from .config import VENDOR_POLYAFILTER
    path = str(VENDOR_POLYAFILTER)
    if path not in sys.path:
        sys.path.insert(0, path)
    import BAMfilter
    import RNAseqAnalysis
    import SNRanalysis
    return BAMfilter, RNAseqAnalysis, SNRanalysis
