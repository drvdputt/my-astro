"""Tools that work on both regular spectra and cubes"""

from specutils import Spectrum1D
import numpy as np

def mask_wavelength_range(s1d, wmin, wmax):
    return np.logical_and(wmin < s1d.spectral_axis, s1d.spectral_axis < wmax)

def slice_spectral_axis(s1d, mask):
    """Arbitrary slice of spectral axis.

    Need to rebuild object to do this
    """
    s1d_new = Spectrum1D(s1d.flux[mask], s1d.spectral_axis[mask], uncertainty=None if s1d.uncertainty is None else s1d.uncertainty[mask])
    s1d_new.meta = s1d.meta
    return s1d_new
    
