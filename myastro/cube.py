"""
Some common things, mostly based on specutils.

E.g. Collapsing, but also collapsing the uncertainty.
"""

import numpy as np
from astropy.nddata import StdDevUncertainty
from copy import deepcopy


def collapse_flux_and_unc(s3d):
    """

    s3d: 3D Spectrum1D cube with uncertainty

    returns: Spectrum1D
        Single spectrum

    """
    spec_avg = s3d.collapse(method=np.nanmean, axis="spatial")

    N = np.count_nonzero(s3d.uncertainty.array, axis=(0, 1))
    spec_avg_unc = np.sqrt(
        np.nansum(np.square(s3d.uncertainty.array), axis=(0, 1))
    ) / np.count_nonzero(np.isfinite(s3d.uncertainty.array), axis=(0, 1))
    spec_avg.uncertainty = StdDevUncertainty(spec_avg_unc * s3d.flux.unit)
    return spec_avg


def clean_spikes(s3d):
    """Some rules to mask out the worst of the spikes in the data"""
    spec2 = deepcopy(s3d)
    f = spec2.flux.value
    u = spec2.uncertainty.array
    median_per_slice = np.array(
        [np.nanmedian(f[..., i][f[..., i] > 0]) for i in range(f.shape[2])]
    )
    too_big = np.abs(f) > 125 * np.abs(median_per_slice)[np.newaxis, np.newaxis, :]
    zero_unc = u <= 0
    bad_unc = ~np.isfinite(u)

    spec2.flux.value[too_big | zero_unc | bad_unc] = np.nan
    spec2.uncertainty.array[too_big | zero_unc | bad_unc] = np.nan
    spec2.mask = ~(np.isfinite(f) & np.isfinite(u))
    spec2.meta = s3d.meta

    return spec2
