"""Some basic tools for plotting filters and integrating over them.

Also has stuff to download filters and load them as objects of my simple
classes.

"""

import stsynphot
import numpy as np
from scipy.interpolate import interp1d
from astropy import units as u


class FilterCurve:
    def __init__(self, central):
        self.w = central
        self.f = lambda x: self.evaluate(x)  # this is for backwards compatibility
        self.weight = self._weight()

    def _weight(self):
        return None

    def eval(self, wavs):
        return None

    def apply(self, wavs, flux):
        transmission = self.evaluate(wavs)
        integral = np.trapz(flux * transmission, wavs)
        return integral / self.weight


class DataFilterCurve(FilterCurve):
    def __init__(self, wav, throughput):
        self.lo = np.amin(wav)
        self.hi = np.amax(wav)
        self.integral = np.trapz(throughput, wav)
        central = np.trapz(throughput * wav, wav) / self.integral
        self.function = interp1d(wav, throughput, bounds_error=False, fill_value=0)
        super().__init__(central)

    def _weight(self):
        return self.integral

    def evaluate(self, wavs):
        return self.function(wavs)


def get_HST_filter_curve(stsynphotstring, wmin, wmax):
    """Example usage
    uv_real = {'SBC_F150LP': get_HST_filter_curve("acs,sbc,f150lp"),
               'SBC_F165LP': get_HST_filter_curve("acs,sbc,f165lp"),
               'WFC3_F225W': get_HST_filter_curve("wfc3,uvis1,f225w"),
               'WFC3_F275W': get_HST_filter_curve("wfc3,uvis1,f275w")}

    wmin and wmax slice the spectrum1d filter object before returning as
    the more efficient "DataFilterCurve"

    """
    bp = stsynphot.band(stsynphotstring).to_spectrum1d()
    s = bp[wmin * u.micron : wmax * u.micron]
    return DataFilterCurve(s.wavelength.to(u.micron).value, s.flux.value)


def calc_sed(wavs, flux, filters):
    """Calculate SED for given flux spectrum and set of filters.

    wavs and flux are just arrays (no units)

    filters is a list of FilterCurve objects
    """
    M = len(filters)
    wavcenters = np.zeros(M)
    sed = np.zeros(M)
    for i, f in enumerate(filters):
        wavcenters[i] = f.w
        sed[i] = f.apply(wavs, flux)

    return wavcenters, sed
