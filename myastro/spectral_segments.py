import numpy as np
from scipy.interpolate import interp1d
from astropy.nddata import StdDevUncertainty
from specutils import Spectrum1D
from myastro import plot
from itertools import cycle


def plot_merge_1d(ax, segments, merge_result, add_segments_with_offset=None):
    """Plot both the input segments and the final product

    Parameters
    ----------

    segments: list of Spectrum1D

    merge_result: Spectrum1D

    add_segments_with_offset: None or flux quantity
        plot the segments a second time, with an added flux offset for readability

    """
    color_it = cycle(("#66CCEE", "#228833", "#EE6677"))
    for s in segments:
        plot.s1d(ax, s) #, color=next(color_it), alpha=0.9)
        print("plotting segment")

    plot.s1d(ax, merge_result, color="k", linewidth=1)


def sort(ss):
    """Put the given spectra in the right order.

    Works by sorting by the lowest wavelength

    """
    return sorted(ss, key=lambda x: x.spectral_axis.value[0])


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


def overlap_shifts(ss, percentile=50, full_output=False):
    """Find the ideal shifts to match spectral segments

    Can be used as an alternative when using ratios doesn't make sense.

    Parameters
    ----------

    percentile : float 0 to 100
        Which percentile to use to calculate the shift
    """
    shifts = []
    median_left = []
    median_right = []
    noise = []
    for left, right in extract_overlapping_data(ss):
        med_left = np.nanpercentile(
            left,
            percentile,
            axis=-1,
        )
        median_left.append(med_left)
        med_right = np.nanpercentile(
            right,
            percentile,
            axis=-1,
        )
        median_right.append(med_right)
        shifts.append(med_left - med_right)
        noise.append(np.sqrt(np.var(left, axis=-1) + np.var(right, axis=-1)) / 2)

    if full_output:
        return (
            np.array(shifts),
            np.array(median_left),
            np.array(median_right),
            np.array(noise),
        )
    else:
        return np.array(shifts)


def shifts_to_offsets(shifts, reference_segment_index):
    """From N - 1 shifts between N segments, derive the N offsets

    Offsets[i] can be applied to segment[i] before merging the segments.

    Parameters
    ----------

    shifts: array
        output of overlap_shifts

    reference_segment_index: int
        the offset for this segment will be set to 0 by construction

    """
    offsets = np.full(len(shifts) + 1, 0, dtype=float)
    offsets[1:] = np.cumsum(shifts)
    offsets -= offsets[reference_segment_index]
    return offsets


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
    for flux_left, flux_right in extract_overlapping_data(ss):
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


def merge_1d(ss):
    """Merge a list of sorted Spectrum1D (no 2D or 3D supported for now)

    Uses a sliding merge for a smooth transition, and assuming that the
    orders are already flux-matched.

    For determining the new wavelength grid, the shortest segment is
    chosen in the overlap regions.

    Parameters
    ----------
    ss: sorted segments (list of Spectrum1D)

    Returns
    -------
    Spectrum1D

    """
    # find the wavelength regions where the segments overlap
    overlap_ranges = find_overlap_ranges(ss)

    # merge everything
    new_spectral_axis = np.sort(np.concatenate([s.spectral_axis for s in ss]))

    # replace the points in the overlap ranges by the finest of the two
    # relevant segments
    for ileft, (wmin, wmax) in enumerate(overlap_ranges):
        # keep the points outside the overlap region
        wavs_outside = new_spectral_axis[
            (new_spectral_axis < wmin) | (new_spectral_axis > wmax)
        ]
        # replace the points inside the overlap region with points from the left segment
        left_wavs = ss[ileft].spectral_axis
        wavs_inside = left_wavs[(left_wavs > wmin) & (left_wavs < wmax)]
        new_spectral_axis = np.sort(np.concatenate([wavs_outside, wavs_inside]))

    flux_new = [
        interp1d(
            s.spectral_axis.value,
            s.flux.value,
            bounds_error=False,
            fill_value=np.nan,
        )(new_spectral_axis.value)
        for s in ss
    ]
    unc_new = [
        interp1d(
            s.spectral_axis.value,
            s.uncertainty.array,
            bounds_error=False,
            fill_value=np.nan,
        )(new_spectral_axis.value)
        for s in ss
    ]

    # first, naively combine into one big spectrum without caring about the jumps
    flux_merged = np.nanmedian(flux_new, axis=0)

    for ileft, (wmin, wmax) in enumerate(overlap_ranges):
        f_left = flux_new[ileft]
        f_right = flux_new[ileft + 1]
        overlap = np.logical_and(new_spectral_axis > wmin, new_spectral_axis < wmax)

        # 0 at wmin, 1 at wmax
        sliding_f = (new_spectral_axis[overlap] - wmin) / (wmax - wmin)
        flux_merged[overlap] = (1 - sliding_f) * f_left[overlap] + sliding_f * f_right[
            overlap
        ]

    new_uncertainty_sum = np.sqrt(np.nansum([a**2 for a in unc_new], axis=0))
    new_uncertainty_count = np.count_nonzero(np.isfinite([a for a in unc_new]), axis=0)
    new_uncertainty = StdDevUncertainty(new_uncertainty_sum / new_uncertainty_count)

    return Spectrum1D(
        flux_merged * ss[0].flux.unit, new_spectral_axis, uncertainty=new_uncertainty
    )


def merge_nd(ss):
    """Merge a list of sorted (by wavelength) Spectrum1D segments

    This is a generalized version of merge_1d

    Parameters
    ----------
    ss: sorted segments (list of Spectrum1D), of same spatial shape

    Returns
    -------
    Spectrum1D

    """
    # find the wavelength regions where the segments overlap
    overlap_ranges = find_overlap_ranges(ss)

    # merge everything
    new_spectral_axis = np.sort(np.concatenate([s.spectral_axis for s in ss]))

    # replace the points in the overlap ranges by the finest of the two
    # relevant segments
    for ileft, (wmin, wmax) in enumerate(overlap_ranges):
        # keep the points outside the overlap region
        wavs_outside = new_spectral_axis[
            (new_spectral_axis < wmin) | (new_spectral_axis > wmax)
        ]
        # replace the points inside the overlap region with points from the left segment
        left_wavs = ss[ileft].spectral_axis
        wavs_inside = left_wavs[(left_wavs > wmin) & (left_wavs < wmax)]
        new_spectral_axis = np.sort(np.concatenate([wavs_outside, wavs_inside]))

    flux_new = [
        interp1d(
            s.spectral_axis.value,
            s.flux.value,
            axis=-1,  # this is default, but I'm being explicit here for clarity
            bounds_error=False,
            fill_value=np.nan,
        )(new_spectral_axis.value)
        for s in ss
    ]
    unc_new = [
        interp1d(
            s.spectral_axis.value,
            s.uncertainty.array,
            axis=-1,
            bounds_error=False,
            fill_value=np.nan,
        )(new_spectral_axis.value)
        for s in ss
    ]

    # first, naively combine into one big spectrum without caring about the jumps
    # At this point, flux_new has indices [segment, space, space, wavelength]
    flux_merged = np.nanmedian(flux_new, axis=0)

    for ileft, (wmin, wmax) in enumerate(overlap_ranges):
        f_left = flux_new[ileft]
        f_right = flux_new[ileft + 1]
        overlap = np.logical_and(new_spectral_axis > wmin, new_spectral_axis < wmax)

        # f(w) = 0 at wmin, 1 at wmax
        sliding_f = (new_spectral_axis[overlap] - wmin) / (wmax - wmin)
        flux_merged[..., overlap] = (1 - sliding_f) * f_left[
            ..., overlap
        ] + sliding_f * f_right[..., overlap]

    new_uncertainty_sum = np.sqrt(np.nansum([a**2 for a in unc_new], axis=0))
    new_uncertainty_count = np.count_nonzero(np.isfinite([a for a in unc_new]), axis=0)
    new_uncertainty = StdDevUncertainty(new_uncertainty_sum / new_uncertainty_count)

    return Spectrum1D(
        flux_merged * ss[0].flux.unit, new_spectral_axis, uncertainty=new_uncertainty
    )


def merge_nd_memfriendly(ss):
    """Memory-friendly version.

    WARNING: the old version is still there because this one has not
    been tested properly.

    It is also much faster, and the uncertainty algorithm works slightly
    differently.

    Merge a list of sorted (by wavelength) Spectrum1D segments

    This is a generalized version of merge_1d

    Parameters
    ----------
    ss: sorted segments (list of Spectrum1D), of same spatial shape

    Returns
    -------
    Spectrum1D

    """
    # find the wavelength regions where the segments overlap
    overlap_ranges = find_overlap_ranges(ss)

    # merge everything
    new_spectral_axis = np.sort(np.concatenate([s.spectral_axis for s in ss]))

    # replace the points in the overlap ranges by the finest of the two
    # relevant segments
    for ileft, (wmin, wmax) in enumerate(overlap_ranges):
        # keep the points outside the overlap region
        wavs_outside = new_spectral_axis[
            (new_spectral_axis < wmin) | (new_spectral_axis > wmax)
        ]
        # replace the points inside the overlap region with points from the left segment
        left_wavs = ss[ileft].spectral_axis
        wavs_inside = left_wavs[(left_wavs > wmin) & (left_wavs < wmax)]
        new_spectral_axis = np.sort(np.concatenate([wavs_outside, wavs_inside]))

    # we will sum everthing together, divide by weight to get average
    flux_merged = np.zeros((ss[0].shape[0], ss[0].shape[1], len(new_spectral_axis)))
    # do the same for unc (but averaging equation is quadratic)
    unc2_merged = np.zeros(flux_merged.shape)

    def new_wav_mask(s):
        return np.logical_and(
            new_spectral_axis >= s.spectral_axis[0],
            new_spectral_axis <= s.spectral_axis[-1],
        )

    def interp_f(s, wmask):
        return interp1d(
            s.spectral_axis.value,
            s.flux.value,
            axis=-1,  # this is default, but I'm being explicit here for clarity
            bounds_error=False,
            fill_value=np.nan,
        )(new_spectral_axis[wmask])

    def interp_u2(s, wmask):
        interp_u = interp1d(
            s.spectral_axis.value,
            s.uncertainty.array,
            axis=-1,  # this is default, but I'm being explicit here for clarity
            bounds_error=False,
            fill_value=np.nan,
        )(new_spectral_axis[wmask])
        return np.square(interp_u)

    # for memory reasons, we step through the whole thing in pairs.
    # start like this, already filling in the leftmost part.

    # initial state
    # xxx | ooo | ooo | ooo

    # first loop
    # L     R write here
    # xxx X xxx | ooo | ooo
    #     ^- and fix overlap

    # next
    #       L     R write here
    # xxx X xxx X xxx | ooo
    #           ^- and fix overlap

    ileft = 0
    iright = 1
    wmask_left = new_wav_mask(ss[ileft])
    flux_left = interp_f(ss[ileft], wmask_left)
    unc2_left = interp_u2(ss[ileft], wmask_left)
    flux_merged[..., wmask_left] = flux_left
    unc2_merged[..., wmask_left] = unc2_left
    for wmin, wmax in overlap_ranges:
        # calculate and write right part
        wmask_right = new_wav_mask(ss[iright])
        flux_right = interp_f(ss[iright], wmask_right)
        unc2_right = interp_u2(ss[iright], wmask_right)
        flux_merged[..., wmask_right] = flux_right
        unc2_merged[..., wmask_right] = unc2_right

        # overwrite overlap part
        wmask_overlap = wmask_left & wmask_right

        # sliding weight weight(w) = 0 at wmin, 1 at wmax
        sliding_weight = (new_spectral_axis[wmask_overlap] - wmin) / (wmax - wmin)
        N_overlap = len(sliding_weight)

        flux_merged[..., wmask_overlap] = (
            # the last N points of the left part
            (1 - sliding_weight) * flux_left[..., -N_overlap:]
            # the first N points of the right part
            + sliding_weight * flux_right[..., :N_overlap]
        )
        unc2_merged[..., wmask_overlap] = (1 - sliding_weight) * unc2_left[
            ..., -N_overlap:
        ] + sliding_weight * unc2_right[..., :N_overlap]

        # at the end of the loop, right becomes left...
        ileft = iright
        wmask_left = wmask_right
        flux_left = flux_right
        unc2_left = unc2_right
        # and new right will be calculated next loop
        iright += 1

    return Spectrum1D(
        flux_merged * ss[0].flux.unit,
        new_spectral_axis,
        uncertainty=StdDevUncertainty(np.sqrt(unc2_merged)),
    )
