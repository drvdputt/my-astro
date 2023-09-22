import numpy as np


def find_overlap_ranges(ss):
    """Find the wavelength overlap regions of a list of spectra.

    Parameters
    ----------

    ss: list of Spectrum1D
        Assumes that the spectra are already sorted, and that each
        spectral segment overlaps only with the previous and
        the next in the list.

    Returns
    -------

    list of 2-tuples representing the ranges where spectral overlap occurs.
        typically [(min1, max0), (min2, max1), ...]

    """
    wav_overlap_ranges = []
    for i in range(1, len(ss)):
        pr = ss[i - 1]
        cr = ss[i]
        v1 = cr.spectral_axis[0]
        v2 = pr.spectral_axis[-1]
        if v2 > v1:
            wav_overlap_ranges.append((v1, v2))

    wav_channel_span = []
    for i in range(0, len(ss), 3):
        short = ss[i]
        long = ss[i + 2]
        wav_channel_span.append((short.spectral_axis[0], long.spectral_axis[-1]))

    return wav_overlap_ranges


def extract_overlapping_data(ss):
    """Extract flux data of overlap regions.

    Returned in order of occurrence.

    Returns
    -------
    list of 2-tuple of array-like
    [(overlap1_fluxes_left, overlap1_fluxes_right),
     (overlap2_fluxes_left, overlap2_fluxes_right),
    ...]
    """
    pairs = []
    for i, (v1, v2) in enumerate(find_overlap_ranges(ss)):
        s_left = ss[i]
        s_right = ss[i + 1]

        flux_left = s_left.flux[
            ..., (s_left.spectral_axis >= v1) & (s_left.spectral_axis <= v2)
        ].value
        flux_right = s_right.flux[
            ..., (s_right.spectral_axis >= v1) & (s_right.spectral_axis <= v2)
        ].value
        pairs.append((flux_left, flux_right))
    return pairs


def overlap_shifts(ss):
    """Find the ideal shifts to match spectral segments

    Can be used as an alternative when using ratios doesn't make sense.
    """
    shifts = []
    median_left = []
    median_right = []
    noise = []
    for left, right in extract_overlapping_data(ss):
        med_left = np.nanmedian(
            left,
            axis=-1,
        )
        median_left.append(med_left)
        med_right = np.nanmedian(
            right,
            axis=-1,
        )
        median_right.append(med_right)
        factors.append(med_left / med_right)
        noise.append(
            np.sqrt(np.var(flux_left, axis=-1) + np.var(flux_right, axis=-1)) / 2
        )

    if full_output:
        return (
            np.array(factors),
            np.array(median_left),
            np.array(median_right),
            np.array(noise),
        )
    else:
        return np.array(factors)


def overlap_ratios(ss, full_output=False):
    """Get the offsets, i.e. stitching factors, for a list of spectral elements.

    The values are the ratios of medians in the wavelength overlap
    regions.

    ss : list of spectrum1D or FastSpectrum

    Returns
    -------
    ratio: array of same spatial shape as flux without spectral axis
        Multiplying ss[i+1] by ratio[i], will match it to order ss[i]

    if full_output is True:
    ratio: array of same spatial as flux without spectral axis
        Multiplying ss[i+1] by ratio[i], will match it to order ss[i]
    median_left: array where each element is median of left segment in each overlap region
    median_right: array where each element is median of right segment in each overlap region
    noise: array of sqrt(std(flux1)**2 + std(flux2)**2) / 2
        measure of the noise in the overlap region

    """
    factors = []
    median_left = []
    median_right = []
    noise = []
    for i, (v1, v2) in enumerate(find_overlap_ranges(ss)):
        s_left = ss[i]
        s_right = ss[i + 1]

        flux_left = s_left.flux[
            ..., (s_left.spectral_axis >= v1) & (s_left.spectral_axis <= v2)
        ].value
        flux_right = s_right.flux[
            ..., (s_right.spectral_axis >= v1) & (s_right.spectral_axis <= v2)
        ].value

        med_left = np.nanmedian(
            flux_left,
            axis=-1,
        )
        median_left.append(med_left)
        med_right = np.nanmedian(
            flux_right,
            axis=-1,
        )
        median_right.append(med_right)
        factors.append(med_left / med_right)
        noise.append(
            np.sqrt(np.var(flux_left, axis=-1) + np.var(flux_right, axis=-1)) / 2
        )

    if full_output:
        return (
            np.array(factors),
            np.array(median_left),
            np.array(median_right),
            np.array(noise),
        )
    else:
        return np.array(factors)
