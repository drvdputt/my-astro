"""Module for loading all kinds of astro data. E.g., a default thing
that globs and sorts the 12 MIRI cubes

"""

from specutils import Spectrum
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.nddata import StdDevUncertainty
from astropy.table import Table


def get_cubes(d, suffix="s3d.fits"):
    """Get all cubes in a directory, sorted by wavelength

    Useful for loading all 12 miri MRS segments.
    """
    cubes = sorted(
        [Spectrum.read(f) for f in Path(d).glob("*" + suffix)],
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
        s = Spectrum(
            sci.data * bunit,
            wcs=WCS(sci),
            uncertainty=StdDevUncertainty(hdul["ERR"].data * bunit),
        )
        s.meta["s_header"] = sci.header
        s.meta["p_header"] = hdul[0].header
        s.meta["file"] = fn
    return s


def iue(fn):
    """Load fits file from IUE

    Tested for SWP mxlo

    Returns
    -------
    Spectrum

    """
    # The astropy table reader does something useful here, but the
    # format is a bit weird. We repackage the spectrum here as a
    # Spectrum objec.t
    t = Table.read(fn)
    flux = t["FLUX"].quantity[0]
    spectral_axis = t["WAVE"].quantity[0]
    # ensure that flux and uncertainty are in the same unit here
    uncertainty = StdDevUncertainty(t["SIGMA"].quantity[0].to(flux.unit))
    return Spectrum(flux, spectral_axis, uncertainty=uncertainty)
