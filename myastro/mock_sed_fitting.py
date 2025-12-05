import numpy as np
from dataclasses import dataclass
from astropy import units as u
from scipy.optimize import minimize
from matplotlib import pyplot as plt


@dataclass
class Spectrum:
    """Simple container because Spectrum is too specialized and slow"""

    w: np.ndarray
    f: np.ndarray


class SEDMockFitter:
    def __init__(
        self,
        filters,
        stellar_spectrum,
        extinction_physical,
        extinction_feature_only=None,
    ):
        """filters: list of FilterCurve objects

           extinction: extinction model from dust_extinction that
           describes the feature of interest (can do .extinguish(av))

        Once set up with a number of filters, the function of this class
        take, apply those filters and noise to make a mock sed. Then,
        the mock SEDs are repeatedly generated and fit to characterize the
        uncertainty on the fit results.

        For now, this is specialized in AV fitting (assume an extinction
        shape and then determine slope + intercept + av)

        stellar_spectrum: provide the stellar spectrum to make things
        easier (UNUSED FOR NOW, it's passed as arguments)

        extinction_physical: the extinction model that is applied to the
        intrinsic spectrum to create the observed spectrum + observed
        SEDs.

        extinction_feature_only: the extinction model that is fit. Can
        be different from the above. E.g., create a model spectrum that
        is extinguished by linear part + bump, but then fit only the
        bump (since the linear part is already included in the slope +
        intercept fit, roughly. Does not always work of course
        (sometimes blackbody fit would be better), but it's kind of the
        point of this class to figure out how accurate we are with
        simple techniques.

        """
        self.filters = filters
        self.extinction = extinction_physical
        if extinction_feature_only is None:
            self.extinction_feature = self.extinction
        else:
            self.extinction_feature = extinction_feature_only

        self.xmin = min(f.lo for f in self.filters)
        self.xmax = max(f.hi for f in self.filters)
        self.spec_model_wavs = np.linspace(self.xmin, self.xmax, 1024)

    def observed_spectrum(self, spectrum, av):
        """Create the underlying physical model on a grid. It is a
        linear continuum extinguished by the extinction function stored
        in self.extinction"""
        flux = spectrum.f * self.extinction.extinguish(spectrum.w * u.micron, av)
        return Spectrum(w=spectrum.w, f=flux)

    def model_spectrum(self, slope, intercept, av):
        """The thing that's actually fitted (feature strength should be
        independent from continuum shape)"""
        flux = (
            slope * self.spec_model_wavs + intercept
        ) * self.extinction_feature.extinguish(self.spec_model_wavs * u.micron, av)
        return Spectrum(w=self.spec_model_wavs, f=flux)

    def mock_sed(self, spectrum):
        """Generate one mock sed based on given spectrum"""
        result = np.zeros(len(self.filters))
        for i, f in enumerate(self.filters):
            transmission = f.f(spectrum.w)
            # Jy * micron
            flux = np.trapz(spectrum.f * transmission, spectrum.w)
            # Jy
            avg_flux_dens = flux / f.weight
            # print(avg_flux_dens)
            result[i] = avg_flux_dens

        return Spectrum(w=np.array([f.w for f in self.filters]), f=result)

    @staticmethod
    def add_noise(sed, snrs):
        """SNR should be same size as SED"""
        delta = np.zeros(len(sed.f))
        for i in range(len(sed.f)):
            delta[i] = np.random.normal(loc=0, scale=sed.f[i] / snrs[i])
        return Spectrum(w=sed.w, f=sed.f + delta)

    def chi2(self, sed_obs, noise_model, slope, intercept, av):
        """sed_obs: observed sed (usually a mock sed)

        noise model: same size as sed_obs, every noise value

        wavs: wavs on which the model spectrum is generated

        """
        s = self.model_spectrum(slope, intercept, av)
        sed_model = self.mock_sed(s)
        return np.sum(np.square((sed_model.f - sed_obs.f) / noise_model))

    def test_mock_sed(self, slope, intercept, av, snrs=None):
        """Quick plot to see if the mocked sed makes sense"""
        s = self.observed_spectrum(slope, intercept, av)
        sed = self.mock_sed(s)
        self.plot_spectrum_and_sed(s, sed)
        if snrs is not None:
            sed = self.add_noise(sed, snrs)
            plt.scatter(sed.w, sed.f, marker=".")

    def fit_obs(self, sed_obs, snrs):
        """Do a single fit to the SED"""
        # need a decent initial guess
        slope = (sed_obs.f[-1] - sed_obs.f[0]) / (sed_obs.w[-1] - sed_obs.w[0])
        intercept = sed_obs.f[-1] - slope * sed_obs.w[-1]

        noise_model = 1 / np.asarray(snrs)
        result = minimize(
            lambda x: self.chi2(
                sed_obs, noise_model, slope=x[0], intercept=x[1], av=x[2]
            ),
            x0=[slope, intercept, 2],
            method="Nelder-Mead",
        )
        return result

    def test_one_fit(self, spectrum, av, snrs):
        s = self.observed_spectrum(spectrum, av)
        sed = self.mock_sed(s)
        mock_sed_with_noise = self.add_noise(sed, snrs)
        result = self.fit_obs(mock_sed_with_noise, snrs)
        print("showing original data and fitted curve")
        s2 = self.model_spectrum(result.x[0], result.x[1], result.x[2])
        sed2 = self.mock_sed(s2)
        self.plot_spectrum_and_sed(s, sed, label="mock spectrum")
        self.plot_spectrum_and_sed(s, mock_sed_with_noise, color="none")
        self.plot_spectrum_and_sed(s2, sed2, label="(linear * ext) fit")
        plt.legend()
        return result

    # check noise level for many mocked seds
    def mock_fit_uncertainty(self, spectrum: Spectrum, av, snrs):
        """spectrum: Spectrum object
               unextinguished Spectrum to be turned into mock SEDs.

        av : float
            av to apply the extinction law

        snrs : float array-like
            snr in every filter

        """
        # noise free spectrum
        s = self.observed_spectrum(spectrum, av)
        # noise free sed
        sed = self.mock_sed(s)

        # number of realizations
        N = 100
        slopes = np.zeros(N)
        intercepts = np.zeros(N)
        av_fits = np.zeros(N)
        for i in range(N):
            mock_sed_with_noise = self.add_noise(sed, snrs)
            result = self.fit_obs(mock_sed_with_noise, snrs)
            slopes[i] = result.x[0]
            intercepts[i] = result.x[1]
            av_fits[i] = result.x[2]

        av_avg = np.average(av_fits)
        av_sigma = np.std(av_fits)
        print(
            f"SNR {snrs} av {av_avg:.2f} sigma {av_sigma:.2f} ratio {av_avg / av_sigma:.2f}"
        )
        plt.hist(av_fits)
        plt.axvline(av)
        plt.axvline(av_avg)

    def plot_spectrum_and_sed(self, spectrum, sed, **kwargs):
        """Both arguments are Spectrum objects"""
        plt.plot(spectrum.w, spectrum.f, **kwargs)
        plt.scatter(sed.w, sed.f, marker="+")
