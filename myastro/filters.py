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
        self.f = lambda x: self.evaluate(x) # this is for backwards compatibility
        self.weight = self._weight()
            
    def _weight(self):
        return None

    def eval(self, wavs):
        return None

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


def get_HST_filter_curve(stsynphotstring):
    bp = stsynphot.band(stsynphotstring).to_spectrum1d()
    s = bp[0.1 * u.micron:0.3*u.micron]
    return DataFilterCurve(s.wavelength.to(u.micron).value, s.flux.value)
