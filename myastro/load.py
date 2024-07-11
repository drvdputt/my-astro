"""Module for loading all kinds of astro data. E.g., a default thing
that globs and sorts the 12 MIRI cubes

"""

from specutils import Spectrum1D
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.nddata import StdDevUncertainty


def get_cubes(d, suffix="s3d.fits"):
    """Get all cubes in a directory, sorted by wavelength

    Useful for loading all 12 miri MRS segments.
    """
    cubes = sorted(
        [Spectrum1D.read(f) for f in Path(d).glob("*" + suffix)],
        key=lambda x: x.spectral_axis.value[0],
    )
    if len(cubes) == 0:
        raise FileExistsError(f"No cubes found in {d}")
    return cubes


def stis_sx2(fn):
    with fits.open(fn) as hdul:
        print(fn)
        print(hdul[0].header["TARGNAME"])
        hdul.info()
        sci = hdul["SCI"]
        bunit = u.Unit(sci.header["BUNIT"])
        s = Spectrum1D(
            sci.data * bunit,
            wcs=WCS(sci),
            uncertainty=StdDevUncertainty(hdul["ERR"].data * bunit),
        )
        s.meta["s_header"] = sci.header
        s.meta["p_header"] = hdul[0].header
        s.meta["file"] = fn
    return s
