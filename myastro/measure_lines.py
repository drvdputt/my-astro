"""Functions to measure lines.

I have two methods available: one based on gaussian+polynomial fitting
(line_continuum_and_flux), the other based on linear continuum
subtraction + numerical integration (measure_complex).

"""

from specutils.fitting import fit_lines
import numpy as np
import astropy.units as u
from pahfit.instrument import fwhm
from astropy.modeling.functional_models import Gaussian1D
from astropy.modeling.polynomial import Polynomial1D
from myastro import plot
from copy import deepcopy

UWAV = u.micron
UFWHM = u.micron
USPECTRUM = u.MJy / u.sr
UPERLAMBDA = u.erg / u.s / u.cm**2 / u.sr / u.micron
ULINEFLUX = u.erg / (u.s * u.cm**2 * u.sr)


def integrate_spectrum(s, wmin=None, wmax=None):
    """Integrates over wavelength, works for arbitrary spatial dimensions

    wmin wmax are Quantities
    """
    wavs = s.spectral_axis.value
    wunit = s.spectral_axis.unit
    if wmin is None:
        imin = 0
    else:
        imin = np.searchsorted(wavs, wmin.to(wunit).value)
    if wmax is None:
        imax = -1
    else:
        imax = np.searchsorted(wavs, wmax.to(wunit).value)

    integral = np.trapz(s.flux.value[..., imin:imax], wavs[imin:imax], axis=-1)
    return integral * s.flux.unit * wunit


def theoretical_fwhm(centers):
    return fwhm("jwst.miri.mrs.*", centers, as_bounded=True)[:, 0]


def line_flux_to_spectral_amplitude(fluxes, wavs, per_lambda=False):
    """Convert gaussian flux to amplitude in my units.

    Parameters
    ----------

    fluxes: fluxes to convert, should be dimensionless (will be
    multiplied with line flux unit).

    wavs: wavelengths in micron

    per_lambda: bool
    If False, use MJy / sr (per nu). If true, use UPERLAMBDA (per
    lambda) as output unit.

    """
    stddevs = theoretical_fwhm(wavs) / 2.35482004503 * UFWHM
    return (fluxes * ULINEFLUX / (stddevs * np.sqrt(2 * np.pi))).to(
        UPERLAMBDA if per_lambda else USPECTRUM,
        equivalencies=u.spectral_density(wavs * UWAV),
    )


def spectral_amplitude_to_line_flux(amps, wavs):
    """Convert gaussian amplitude to flux in my units.

    Uses PAHFIT instrument model to determine FWHM.

    """
    stddevs = theoretical_fwhm(wavs) / 2.35482004503 * UFWHM

    return (amps * UPERLAMBDA * stddevs * np.sqrt(2 * np.pi)).to(
        ULINEFLUX, equivalencies=u.spectral_density(wavs * UWAV)
    )


def s1d_sorted_slice(s1d, wmin, wmax):
    """Spectrum1D slicing is slow. Make it faster by assuming that the
    wavelengths are sorted"""
    imin = np.searchsorted(s1d.spectral_axis.value, wmin)
    imax = np.searchsorted(s1d.spectral_axis.value, wmax)
    return s1d[..., imin:imax]


def line_neighbourhood_unc(
    s1d_per_lambda,
    center,
    wcont_min,
    wcont_max,
    wcore_min,
    wcore_max,
    num_samples=10,
    ax=None,
):
    """Simple uncertainty estimate around line: take 10 adjacent windows
    (5 on each side), each of the same width as the line measurement
    window. For each of those windows, calculate the standard deviation
    of the flux values. Then take the median of the standard deviations
    to make it more robust against outliers (the line itself).

    Recommendation in case of uncertainty being blown up by other nearby
    lines: remove all lines from spectrum before passing it here.

    Then, the estimated flux standard deviation is converted to an
    uncertainty on the line flux by multiplying with the window width,
    and converting to our line unit. (it's interpreted as uncertainty on
    the constant continuum level)

    Parameters
    ----------
    s1d_per_lambda : Spectrum1D
        Again, in per-wavelength units. Otherwise integrating doesn't
        make sense.

    center : float
        Line central wavelength in micron

    wcont_min, wcont_max : float
        the limits for the continuum estimation window in micron

    wcore_min, wcore_max : float
        the limits for the line integration window in micron

    num_samples : int
        Number of windows to use. They are adjacent, starting at (center
        - num_samples // 2 * wsize)

    ax : plot a demonstration on the given ax

    Returns
    -------
    Quantity

    """
    w = s1d_per_lambda.spectral_axis.to(u.micron).value
    f = s1d_per_lambda.flux.value
    r = num_samples // 2
    centers = []
    left_edges = []
    right_edges = []
    cont_models = []
    integral_samples = []
    csamples = []
    wsamples = []

    # for a set of different wavelength offsets, get continuum models
    # and some values for plotting.
    wdelta = (wcore_max - wcore_min) * 0.5
    for i in range(-r, r + 1):  # inclusive interval
        # avoid samples close to line
        if i == -1 or i == 0 or i == 1:
            continue

        shift = i * wdelta
        wcont_min_shift = wcont_min + shift
        wcont_max_shift = wcont_max + shift
        wcore_min_shift = wcore_min + shift
        wcore_max_shift = wcore_max + shift

        # avoid samples that go outside of the wavelength range
        if wcont_min_shift < w[0] or wcore_max_shift > w[-1]:
            continue

        centers.append(center + shift)
        left_edges.append(wcont_min_shift)
        right_edges.append(wcont_max_shift)

        # create linear continuum function
        cont_linear = left_right_median_interp(
            w, f, wcont_min_shift, wcore_min_shift, wcore_max_shift, wcont_max_shift
        )
        cont_models.append(cont_linear)

        # subtract the linear continuum from each sample
        core_interval = (w > wcore_min_shift) & (w < wcore_max_shift)
        wsample = w[core_interval]
        fsample_cont_sub = f[..., core_interval] - cont_linear(w[core_interval])
        integral_sample = np.trapz(fsample_cont_sub, wsample)
        integral_samples.append(integral_sample)
        csamples.append(cont_linear(wsample))
        wsamples.append(wsample)

    # outlier-resistent standard deviation of integral samples, with 0
    # the assumed mean (continuum subtraction of continuum should be 0,
    # but isn't because of noise over which we integrate)
    # root mean square for starters, or median of absolutes?
    mad = np.nanmedian(np.abs(integral_samples))
    line_unc = mad * s1d_per_lambda.flux.unit * s1d_per_lambda.spectral_axis.unit

    # make plot if requested
    if ax is not None:
        if len(s1d_per_lambda.shape) > 1:
            raise ValueError("s1d can only be 1D for making this plot")

        plot.s1d(ax, s1d_per_lambda, color="k")
        # convert integral offset to average in flux unit?
        conts = np.array([func(c) for func, c in zip(cont_models, centers)])
        flux_offset = np.array(integral_samples) / (wcore_max - wcore_min)
        mad_flux_offset = mad / (wcore_max - wcore_min)
        ax.errorbar(
            centers,
            conts,
            mad_flux_offset,
            marker=".",
            label="linear continuum",
            color="xkcd:blue",
            capsize=0,
            linestyle="none",
        )
        ax.scatter(
            centers,
            flux_offset + conts,
            color="xkcd:orange",
            label="continuum integral",
            marker="x",
        )
        for ws, cs in zip(wsamples, csamples):
            ax.plot(ws, cs, color="xkcd:blue")

    return line_unc


def left_right_median_interp(x, y, left_min, left_max, right_min, right_max):
    """Estimate continuum by taking medians of windows left and right of
    the line. Then interpolate linearly between these medians"""
    left_interval = (x > left_min) & (x < left_max)
    right_interval = (x > right_min) & (x < right_max)
    x_left = np.average(x[left_interval])
    y_left = np.nanmedian(y[..., left_interval], axis=-1)
    x_right = np.average(x[right_interval])
    y_right = np.nanmedian(y[..., right_interval], axis=-1)

    # simple linear interpolation continuum
    def _cont_linear(w):
        return (y_left * (x_right - w) + y_right * (w - x_left)) / (x_right - x_left)

    return _cont_linear


def line_continuum_and_flux(s1d_per_lambda, center, fwhm_micron=None, s1d_for_unc=None):
    """Naive (but likely sufficient) continuum + line flux estimate

    s1d_per_lambda

    Uncertainty estimate works by taking standard deviation of window
    with same size as the measurement window.

    Parameters
    ----------

    s1d_per_lamba: Spectrum1D
        needs to be in per micron units for everything to make sense!

    center: float
        central wavelength of the line in micron

    fwhm_micron: float
        width of the line to assume. By default, a suitable FWHM will be
        derived using the MIRI resolution curve model from PAHFIT.

    s1d_for_unc: Spectrum1D
        use a different spectrum (e.g. one with lines removed) to
        estimate the uncertainty in windows near the line. By default, uses a copy of s1d_per_lambda

    remove_nearby_lines: list of float
        list of wavelengths in micron. Windows around the lines at these
        wavelengths will be cut out of the spectrum to improve the
        uncertainty estimation (avoid it from blowing up).

    Returns
    -------
    output: dict
        {"line_flux": Quantity,
         "line_unc": Quantity,
         "cont_model": function that can be evaluated at wavelength to get continuum,
         "wavs": wavelengths of the lines
         "peak_wav": measured wavelength of peak of line

    """
    if fwhm_micron is None:
        fwhm = theoretical_fwhm([center])[0]
    else:
        fwhm = fwhm_micron

    if not s1d_per_lambda.flux.unit.is_equivalent(UPERLAMBDA):
        raise ValueError("s1d is not in per lambda units! Result won't make sense!")

    # median estimate left and right of line
    w = s1d_per_lambda.spectral_axis.to(u.micron).value
    f = s1d_per_lambda.flux.value
    wcont_min = center - 4 * fwhm
    wcont_max = center + 4 * fwhm
    wcore_min = center - 2 * fwhm
    wcore_max = center + 2 * fwhm
    cont_linear = left_right_median_interp(
        w, f, wcont_min, wcore_min, wcore_max, wcont_max
    )

    # continuum-subtracted flux
    cont_sub_s1d = s1d_per_lambda - cont_linear(w) * s1d_per_lambda.unit

    flux = integrate_spectrum(
        cont_sub_s1d, wmin=wcore_min * u.micron, wmax=wcore_max * u.micron
    )

    # uncertainty estimation (with optional modified spectrum)
    the_s1d_for_unc = s1d_per_lambda if s1d_for_unc is None else s1d_for_unc
    unc = line_neighbourhood_unc(
        the_s1d_for_unc, center, wcont_min, wcont_max, wcore_min, wcore_max, 10
    )

    # also measure observed wavelength, not just theoretical?
    mask = (w > wcore_min) & (w < wcore_max)
    idx = np.argmax(cont_sub_s1d.flux.value[..., mask], axis=-1)
    peak_wav = w[mask][idx]
    # make the window a bit narrower.  (this will limit how much offset we can measure though)
    wcentroid_min = center - 1 * fwhm
    wcentroid_max = center + 1 * fwhm
    centroid_wav = (
        integrate_spectrum(
            cont_sub_s1d * cont_sub_s1d.spectral_axis,
            wmin=wcentroid_min * u.micron,
            wmax=wcentroid_max * u.micron,
        )
        / integrate_spectrum(
            cont_sub_s1d,
            wmin=wcentroid_min * u.micron,
            wmax=wcentroid_max * u.micron,
        )
    ).to(u.micron)

    return {
        "line_flux": flux.to(ULINEFLUX),
        "line_unc": unc,
        # cont_model:cont,
        "cont_model": lambda w: cont_linear(w.value) * s1d_per_lambda.unit,
        "wavs": s1d_per_lambda.spectral_axis[(w > wcont_min) & (w < wcont_max)],
        "peak_wav": peak_wav,
        "centroid_wav": centroid_wav,
    }


def measure_all(s1d, centers):
    """Naive measurement of all the lines at the given centers.

    Will use pahfit instrument model to set FWHM

    Some lines will need followup because of bad continuum.

    Parameters
    ----------
    centers: list of float
        Central wavelengths in micron

    Returns
    -------
    fluxes: flux for every line (Quantity)

    uncertainties: rough estimates for the uncertainty based on the
    local noise in the spectrum (see line_neighborhood_unc function) (Quantity)

    avg_conts: continuum level subtracted from each line. If continuum
        was a curve model instead of single value, the average value of
        the curve is returned. Can be used to check for bad continuum values.

    """
    # calculate or set up storage for line parameters
    fwhms = theoretical_fwhm(centers)
    fluxes = np.zeros(len(centers))
    uncs = np.zeros(len(centers))
    avg_conts = np.zeros(len(centers))
    wav_obs = np.zeros(len(centers))

    # do the conversion once here for efficiency
    s1d_per_lambda = s1d.new_flux_unit(UPERLAMBDA)

    # modified copy of the spectrum with lines removed, better for uncertainty estimation.
    s1d_for_unc = deepcopy(s1d_per_lambda)
    for c, fw in zip(centers, fwhms):
        window = np.logical_and(
            s1d_for_unc.spectral_axis.value > c - 2 * fw,
            s1d_for_unc.spectral_axis.value < c + 2 * fw,
        )
        s1d_for_unc.flux.value[window] = np.nan

    for i in range(len(centers)):
        output = line_continuum_and_flux(
            s1d_per_lambda, centers[i], fwhms[i], s1d_for_unc
        )
        fluxes[i] = output["line_flux"].value
        uncs[i] = output["line_unc"].value
        flux_unit = output["line_flux"].unit
        cont_model_eval = output["cont_model"](output["wavs"])
        cont_model_unit = cont_model_eval.unit
        avg_conts[i] = np.average(cont_model_eval.value)
        wav_obs[i] = output["centroid_wav"].value

    avg_conts *= cont_model_unit

    return {
        "flux": fluxes * flux_unit,
        "unc": uncs * flux_unit,
        "cont": avg_conts.to(
            s1d.unit, equivalencies=u.spectral_density(np.asarray(centers) * u.micron)
        ),
        "wav_obs": wav_obs,
    }


def measure_complex(
    s1d_per_lambda,
    centers,
    fwhms,
    wave_bounds_fractional=0.001,
    continuum_degree=1,
    alt_continuum=False,
    window_fwhm=4,
):
    """Measure slightly overlapping lines.

    Simultaneously fits line amplitude, width, center, and continuum.

    Parameters
    ----------

    centers: array
        wavelengths of the lines to fit (needs to be sorted ascending)

    fwhms: array
        fwhm to use for each line as an initial guess, and to determine
        the fitting window.

    wave_bounds_fractional: width of the interval in which the fitter is
    given freedom to fit the central wavelength. In fractions of
    wavelength.

    continuum_degree: int
        Degree of the polynomial to use for the continuum fit.

    alt_continuum: array
        Manually specify polynomial parameters

    window_fwhm: float
        Width of the window for the fit. The continuum is measured from
        -window_fwhm * fwhm to -(window_fwhm - 1) * fwhm. Can be
        adjusted to avoid big peaks.

    Returns
    -------
    dict :
        "fluxes" : list of luxes
        "fit" : fitted model object
        "window" : 2 tuple (wmin, wmax)

    """
    # reasonable numbers
    avgflux = np.average(s1d_per_lambda.flux)
    stddevs = fwhms / 2.35482004503
    centers_um = np.array(centers) * u.micron
    window = (
        min(centers_um) - window_fwhm * fwhms[0],
        max(centers_um) + window_fwhm * fwhms[-1],
    )

    # set up models
    gauss_names = []
    gauss_components = []
    for i, (w, s) in enumerate(zip(centers_um, stddevs)):
        name = f"g{i}"
        gauss_names.append(name)
        gauss_components.append(
            Gaussian1D(
                amplitude=avgflux,
                mean=w,
                stddev=s,
                fixed=dict(mean=False, stddev=False),
                bounds=dict(
                    mean=(
                        w * (1 - wave_bounds_fractional),
                        w * (1 + wave_bounds_fractional),
                    ),
                    stddev=(s * 0.7, s * 1.3),
                ),
                name=name,
            )
        )

    # add everything together with linear continuum
    model = gauss_components[0]
    if len(gauss_components) > 1:
        for c in gauss_components[1:]:
            model += c
    # model += (slope=0, intercept=avgflux)

    # try fixed continuum
    f = s1d_per_lambda.flux
    w = s1d_per_lambda.spectral_axis
    wlo = window[0]
    whi = window[1]

    x1 = window[0] + fwhms[0]
    x2 = window[1] - fwhms[-1]

    y1 = np.median(f[np.logical_and(w > wlo, w < wlo + 2 * fwhms[0])])
    y2 = np.median(f[np.logical_and(w > whi - 2 * fwhms[-1], w < whi)])
    slope = (y2 - y1) / (x2 - x1)
    intercept = y2 - slope * x2
    print(x1, y1)
    print(x2, y2)
    print(slope, intercept)
    cont = Polynomial1D(
        continuum_degree,
        name="cont",
        fixed={"c0": alt_continuum, "c1": alt_continuum},
    )
    if continuum_degree == 1:
        cont.c0 = intercept
        cont.c1 = slope

    model += cont

    fit_result = fit_lines(s1d_per_lambda, model, window=window)

    gauss_results = [fit_result[name] for name in gauss_names]
    amps = (
        np.array([g.amplitude.value for g in gauss_results])
        * gauss_results[0].amplitude.unit
    )
    stddevs = (
        np.array([g.stddev.value for g in gauss_results]) * gauss_results[0].stddev.unit
    )
    wavs = np.array([g.mean.value for g in gauss_results]) * gauss_results[0].mean.unit

    fluxes = (amps * stddevs * np.sqrt(2 * np.pi)).to(
        ULINEFLUX, equivalencies=u.spectral_density(wavs)
    )
    return {
        "fluxes": fluxes,
        "wavs": wavs,
        "fit": fit_result,
        "window": window,
        "cont_values": fit_result["cont"](centers * u.micron),
    }


def cube_line_map(ax, s3d, line_wav):
    """Apply line measurement algorithm to every point in cube, and plot
    it

    WATCH out for unit! Should be per_lambda"""
    result = line_continuum_and_flux(s3d, line_wav)
    v = result["line_flux"].value
    ax.imshow(v, vmin=0, vmax=np.percentile(v, 99.5))
