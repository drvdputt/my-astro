"""Plotting functions specific for spectra"""

from myastro.plot._colors import MY_COLOR_CYCLE
from myastro import spectrum_general

import numpy as np
from astropy import units as u


def s1d(ax, s, add_labels=True, offset=None, wav_unit=u.micron, **kwargs):
    """My favorite default way to plot a spectrum

    s : Spectrum1D

    add_labels : bool
        Add wavelength and flux labels to ax, with units

    wav_unit : Unit
        Change spectral axis unit before plotting
    """
    default_kwargs = dict(linewidth=0.5, drawstyle="steps")

    w = s.spectral_axis.to(wav_unit).value
    if offset is not None:
        w -= offset
    ax.plot(w, s.flux.value, **(default_kwargs | kwargs))

    if add_labels:
        ax.set_xlabel(f"wavelength ({s.spectral_axis.unit:latex_inline})")
        ax.set_ylabel(f"flux ({s.flux.unit:latex_inline})")


def compare_profiles(
    ax, wavelength, fluxes, labels, wmin, wmax, wnorm, wnorm_bottom=None
):
    """Compare profiles of two spectra.

    Zoom in on an emission complex plot the spectrum normalized
    according to a certain wavelength, to compare profiles of two
    spectra.

    See arguments of spectrum_general.normalize

    TODO: describe parameters and allow more than two 'flux' arguments
    (need to work on color/linestyle stuff for that)

    """
    w = wavelength
    w = w[np.logical_and(wmin < w, w < wmax)]

    fs_norm = []

    for i in range(len(fluxes)):
        f_norm = spectrum_general.normalize(w, fluxes[i], wnorm, wnorm_bottom)["flux"]
        ax.plot(w, f_norm, label=labels[i], color=next(MY_COLOR_CYCLE), alpha=0.8)
        fs_norm.append(f_norm)

    ax.set_xlabel("wavelength")
    ax.set_ylabel("$(F_\\nu(\\lambda) - c)\\ /\\ (F_\\nu(\\mathrm{peak}) - c)$")
    ax.set_ylim(np.amin(fs_norm) - 0.05, 1.05)
    ax.set_xlim(wmin, wmax)
