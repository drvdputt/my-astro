"""Tools that work on both regular spectra and cubes"""

from specutils import Spectrum1D
import numpy as np
from astropy import units as u
from astropy.table import Table
from astropy.nddata import StdDevUncertainty


def mask_wavelength_range(s1d, wmin, wmax):
    return np.logical_and(wmin < s1d.spectral_axis, s1d.spectral_axis < wmax)


def slice_spectral_axis(s1d, mask):
    """Arbitrary slice of spectral axis.

    Need to rebuild object to do this
    """
    s1d_new = Spectrum1D(
        s1d.flux[mask],
        s1d.spectral_axis[mask],
        uncertainty=None if s1d.uncertainty is None else s1d.uncertainty[mask],
    )
    s1d_new.meta = s1d.meta
    return s1d_new


def coadd(s1ds, new_spectral_axis=None):
    """Co-add any list of Spectrum1D objects.

    If cubes, they need to have the same shapes (not supported for now)

    Parameters
    ----------

    s1ds: list of Spectrum1D

    new_spectral_axis: new wavelength grid. In the future, I want this
    to be somewhat automatic if None is chosen.

    """
    pass


def take_s1d_or_quantities(func):
    def decorated_func(
        *args, s1d=None, wavelength=None, flux=None, uncertainty=None, **kwargs
    ):
        if s1d is not None:
            w = s1d.spectral_axis
            f = s1d.flux
            u = s1d.uncertainty
        else:
            w = wavelength
            f = flux
            u = uncertainty

        return func(*args, wavelength=w, flux=f, uncertainty=u)

    return decorated_func


@take_s1d_or_quantities
def write_ascii_etc(fn, wavelength, flux, uncertainty):
    """Write a spectrum1d in ascii compatible with HST ETCs"""
    flux_etc = flux.to(u.erg * u.cm**-2 * u.s**-1 * u.angstrom**-1).value
    wavelength_etc = wavelength.to(u.angstrom).value
    good = (flux_etc > 0) & (wavelength_etc > 0) & np.isfinite(flux_etc)
    np.savetxt(fn, np.column_stack((wavelength_etc[good], flux_etc[good])))


@take_s1d_or_quantities
def write_ecsv(fn, wavelength, flux, uncertainty):
    """Write spectrum (in astropy quantities) to ecsv file

    s1d: Spectrum1D
        If not given, then the individual wavelength, flux, uncertainty
        arguments should be used.

    wavelength: Quantity

    flux: Quantity

    uncertainty: array like or e.g. StdDevUncertainty
        (will be converted to array and multiplied with flux unit)

    fn: str
        file name, best is to use ECSV

    """
    t = Table()
    t.add_column(wavelength, name="wavelength")
    t.add_column(flux, name="flux")
    if uncertainty is not None:
        t.add_column(uncertainty.array * flux.unit)
    t.write(fn)


def read_ecsv(fn):
    t = Table.read(fn)
    return Spectrum1D(
        t["flux"].quantity,
        t["wavelength"].quantity,
        uncertainty=StdDevUncertainty(t["uncertainty"]) if "uncertainty" in t.colnames else None,
    )
