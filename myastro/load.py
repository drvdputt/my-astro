"""Module for loading all kinds of astro data. E.g., a default thing
that globs and sorts the 12 MIRI cubes

"""

from specutils import Spectrum1D
from pathlib import Path

def get_cubes(d, suffix="s3d.fits"):
    """Get all cubes in a directory, sorted by wavelength

    Useful for loading all 12 miri MRS segments.
    """
    cubes = sorted(
        [Spectrum1D.read(f) for f in Path(d).glob("*"+suffix)],
        key=lambda x: x.spectral_axis.value[0],
    )
    if len(cubes) == 0:
        raise f"No cubes found in {d}"
    return cubes
