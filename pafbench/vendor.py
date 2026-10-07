import sys

from .config import VENDOR_POLYAFILTER


def polyafilter_modules():
    """Import the pinned, unmodified polyAfilter modules."""
    path = str(VENDOR_POLYAFILTER)
    if path not in sys.path:
        sys.path.insert(0, path)
    import BAMfilter
    import RNAseqAnalysis
    import SNRanalysis
    return BAMfilter, RNAseqAnalysis, SNRanalysis
