"""Tools that work on both regular spectra and cubes"""

from specutils import Spectrum1D
import numpy as np
from astropy import units as u
from astropy.table import Table
from astropy.nddata import StdDevUncertainty


def mask_wavelength_range(s1d, wmin, wmax):
    return np.logical_and(wmin < s1d.spectral_axis, s1d.spectral_axis < wmax)


def slice_spectral_axis(s1d, mask):
    """Arbitrary slice of spectral axis using boolean mask

    Remove data points from a Spectrum1D object according to the given
    wavelength mask.

    Parameters
    ----------
    spec: Spectrum1D

    mask: bool array same size as spec wavelength axis
        True means to keep the data for that wavelength

    Returns
    -------

    Spectrum1D object with sliced flux, spectral axis, uncertainty, and a copy of the original metadata.

    """
    s1d_new = Spectrum1D(
        s1d.flux[mask],
        s1d.spectral_axis[mask],
        uncertainty=None if s1d.uncertainty is None else s1d.uncertainty[mask],
    )
    s1d_new.meta = s1d.meta
    return s1d_new


def remove_wavelength_ranges(s1d, wmin_wmax_pairs):
    """Remove a list of wavelength ranges from a Spectrum1D.

    This is a common task applied before fitting a spectrum.

    Parameters
    ----------

    s1d: Spectrum1D

    wmin_wmax_pairs: list of (float, float)
        Every wavelength range (in micron) that needs to be removed.

    """
    to_remove = np.full(s1d.spectral_axis.shape, False)
    for wmin, wmax in wmin_wmax_pairs:
        to_remove = to_remove | mask_wavelength_range(
            s1d, wmin * u.micron, wmax * u.micron
        )
    print(f"Removing {np.count_nonzero(to_remove)} wavelength points")
    return slice_spectral_axis(s1d, ~to_remove)


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
    """
    Decorate function so that it can take either s1d or (wavelength, flux, uncertainty).
    """

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

        return func(*args, wavelength=w, flux=f, uncertainty=u, **kwargs)

    return decorated_func


@take_s1d_or_quantities
def write_ascii_etc(fn, wavelength, flux, uncertainty):
    """Write a spectrum1d in ascii compatible with HST ETCs

    args: fn (str): filename
    """
    flux_etc = flux.to(u.erg * u.cm**-2 * u.s**-1 * u.angstrom**-1).value
    wavelength_etc = wavelength.to(u.angstrom).value
    good = (flux_etc > 0) & (wavelength_etc > 0) & np.isfinite(flux_etc)
    np.savetxt(fn, np.column_stack((wavelength_etc[good], flux_etc[good])))


@take_s1d_or_quantities
def write_ecsv(fn, wavelength, flux, uncertainty, **kwargs):
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

    kwargs: arguments passed through to write() function of
        astropy.table.Table.

    """
    t = Table()
    t.add_column(wavelength, name="wavelength")
    t.add_column(flux, name="flux")
    if uncertainty is not None:
        t.add_column(uncertainty.array * flux.unit, name="uncertainty")
    t.write(fn, **kwargs)


def read_ecsv(fn):
    t = Table.read(fn)
    return Spectrum1D(
        t["flux"].quantity,
        t["wavelength"].quantity,
        uncertainty=(
            StdDevUncertainty(t["uncertainty"]) if "uncertainty" in t.colnames else None
        ),
    )


def normalize(w, flux, wnorm, wnorm_bottom=None, ax=None):
    """
    Normalize by amplitude and shift spectrum

    wnorm: float or 2-tuple of float
        Wavelength at which flux will be chosen for normalization. When tuple:
        range in which the maximum flux is used instead.

    wnorm_bottom: float
        Wavelength at which the flux will be used as the offset.
        Fmax - amplitude is the normalization factor. The spectrum is shifted
        down by offset before normalization.

    ax: if given, plot the points used for normalization (unimplemented)

    Returns
    -------
    dict: {'flux': normalized flux
            'inorm': index of 'top' data point used to normalize
            'inorm_bottom': index of 'bottom data point}

    """
    # as normalization factor, choose amplitude of feature at wnorm?
    # I.e. the difference between 16.4 flux and continuum?

    if wnorm_bottom is None:
        inorm_bottom = None
        offset = np.percentile(flux, 1)
    else:
        inorm_bottom = np.searchsorted(w, wnorm_bottom)
        offset = flux[inorm_bottom]

    if hasattr(wnorm, "__len__"):
        istart = np.searchsorted(w, wnorm[0])
        istop = np.searchsorted(w, wnorm[1])
        inorm = np.argmax(flux[istart:istop]) + istart
    else:
        inorm = np.searchsorted(w, wnorm)
    fnorm = flux[inorm]

    factor = fnorm - offset

    # probably better to have some sort of continuum subtraction option,
    # so that we can compare the amplitudes of the features.
    return {
        "flux": (flux - offset) / factor,
        "inorm": inorm,
        "inorm_bottom": inorm_bottom,
    }
