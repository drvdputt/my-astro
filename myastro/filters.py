"""Some basic tools for plotting filters and integrating over them.

Also has stuff to download filters and load them as objects of my simple
classes.

"""
from itertools import cycle
import stsynphot
import numpy as np
from scipy.interpolate import interp1d
from astropy import units as u
from matplotlib import pyplot as plt


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


def get_stsynphot_filter_curve(string, wmin, wmax):
    bp = stsynphot.band(string).to_spectrum1d()
    s = bp[wmin * u.micron : wmax * u.micron]
    return DataFilterCurve(s.wavelength.to(u.micron).value, s.flux.value)


def calc_sed(wavs, flux, filters):
    """Calculate SED for given flux spectrum and set of filters.

    wavs and flux are just arrays (no units)

    filters: list of FilterCurve objects
    """
    M = len(filters)
    wavcenters = np.zeros(M)
    sed = np.zeros(M)
    for i, f in enumerate(filters):
        wavcenters[i] = f.w
        sed[i] = f.apply(wavs, flux)

    return wavcenters, sed


def plot_spectrum_filters_and_sed(
        ax_sed, ax_filter, wavs, flux, filters, extra_fluxes=[], text_pos=None, text_rot=0
):
    """ax_filter: recommendation: a twin axis of ax_sed

    wavs: wavelengths in the units supported by the given filter objects

    flux: flux array (units don't matter)

    filters: dict {str: FilterCurve}
        labels and FilterCurve objects

    extra fluxes: list of array
        extra curves, will use a line style cycle

    text_pos: list of float
        where to put the filter labels on the filter axes (vertical
        position). Defaults to the average flux.

    """
    wavcenters, sed = calc_sed(wavs, flux, filters.values())

    # filters
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    for i, name in enumerate(filters):
        t = filters[name].evaluate(wavs)
        mask = t != 0
        ax_filter.plot(wavs[mask], t[mask], color=colors[i], alpha=0.5)

        if text_pos is None:
            text_y = np.average(flux)
        else:
            text_y = text_pos[i]

        ax_sed.text(
            wavcenters[i],
            text_y,
            name,
            # name.split("_")[-1],
            color=colors[i],
            ha="center",
            rotation=text_rot,
        )

    # spectrum
    ax_sed.plot(wavs, flux, color="k")

    # sed
    ax_sed.scatter(wavcenters, sed, c=colors[: len(sed)])

    # same with an extra spectrum (plotted at low alpha or dotted)
    styles = cycle(("dotted", "dashed"))
    for ef in extra_fluxes:
        _, ref_sed = calc_sed(wavs, ef, filters.values())
        ax_sed.plot(wavs, ef, color="k", alpha=0.5, linestyle=next(styles))
        ax_sed.scatter(wavcenters, ref_sed, c=colors[: len(sed)], alpha=0.5)

    ax_sed.set_xlabel("wavelength (micron)")
    ax_filter.set_ylabel("Filter transmission")
