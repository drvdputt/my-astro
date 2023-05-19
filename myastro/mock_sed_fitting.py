import numpy as np
from dataclasses import dataclass
from astropy import units as u
from scipy.optimize import minimize
from matplotlib import pyplot as plt

@dataclass
class Spectrum:
    """Simple container because Spectrum1D is too specialized and slow"""
    w: np.ndarray
    f: np.ndarray

class SEDMockFitter:
    def __init__(self, filters, extinction_physical, extinction_feature_only=None):
        """filters: list of FilterCurve objects

           extinction: extinction model from dust_extinction that
           describes the feature of interest (can do .extinguish(av))

        """
        self.filters = filters
        self.extinction = extinction_physical
        if extinction_feature_only is None:
            self.extinction_feature = self.extinction
        else:
            self.extinction_feature = extinction_feature_only
        # ideal range based on effective width of the filters
        self.xmin = min(f.lo for f in self.filters)
        self.xmax = max(f.hi for f in self.filters)
        self.x = np.linspace(self.xmin, self.xmax, 1024)        

    def starting_spectrum(self, slope, intercept, av):
        """Create the underlying physical model on a grid. It is a
        linear continuum extinguished by the extinction function stored
        in self.extinction"""
        flux = (slope * self.x + intercept) * self.extinction.extinguish(self.x * u.micron, av)
        return flux

    def model_spectrum(self, slope, intercept, av):
        """The thing that's actually fitted (feature strength should be
        independent from continuum shape)"""
        flux = (slope * self.x + intercept) * self.extinction_feature.extinguish(self.x * u.micron, av)
        return flux

    def mock_sed(self, flux_array):
        """Generate one mock sed based on given spectrum"""
        result = np.zeros(len(self.filters))
        for i, f in enumerate(self.filters):
            transmission = f.f(self.x)
            # Jy * micron
            flux = np.trapz(flux_array * transmission, self.x)
            # Jy
            avg_flux_dens = flux / f.weight
            # print(avg_flux_dens)
            result[i] = avg_flux_dens
        
        return Spectrum(w=np.array([f.w for f in self.filters]),
                        f=result)

    @staticmethod
    def add_noise(sed, snr):
        delta = np.zeros(len(sed.f))
        for i in range(len(sed.f)):
            delta[i] = np.random.normal(loc=0, scale=sed.f[i] / snr)
        return Spectrum(w=sed.w, f=sed.f + delta)
 
    def chi2(self, sed_obs, snr, slope, intercept, av):
        # forward model use differnet extinction, for which feature
        # strength is independent of rest of shape
        s = self.model_spectrum(slope, intercept, av)
        sed_model = self.mock_sed(s)
        # noise model (for chi 2)
        noise_model = sed_model.f / snr
        return np.sum(np.square((sed_model.f - sed_obs.f) / noise_model))

    def fit_obs(self, sed_obs, snr):
        # need a decent initial guess
        slope = (sed_obs.f[-1] - sed_obs.f[0]) / (sed_obs.w[-1] - sed_obs.w[0])
        intercept = sed_obs.f[-1] - slope * sed_obs.w[-1]
    
        result = minimize(
            lambda x: self.chi2(sed_obs, snr, slope=x[0], intercept=x[1], av=x[2]),
            x0=[slope, intercept, 2],
            method = 'Nelder-Mead'
        )
        return result

    def test_mock_sed(self, slope, intercept, av, snr=None):
        """Quick plot to see if the mocked sed makes sense"""
        s = self.starting_spectrum(slope, intercept, av)
        sed = self.mock_sed(s)
        self.plot_spectrum_and_sed(s, sed)
        if snr is not None:
            sed = self.add_noise(sed, snr)
            plt.scatter(sed.w, sed.f, marker='.')

    def test_one_fit(self, slope, intercept, av, snr):
        s = self.starting_spectrum(slope, intercept, av)
        sed = self.mock_sed(s)
        mock_sed_with_noise = self.add_noise(sed, snr)
        result = self.fit_obs(mock_sed_with_noise, snr)
        print("showing original data and fitted curve")
        s2 = self.model_spectrum(result.x[0], result.x[1], result.x[2])
        sed2 = self.mock_sed(s2)
        self.plot_spectrum_and_sed(s, sed)
        self.plot_spectrum_and_sed(s, mock_sed_with_noise)
        self.plot_spectrum_and_sed(s2, sed2)
        return result

    # check noise level for many mocked seds
    def mock_fit_uncertainty(self, slope, intercept, av, snrs=[10, 20, 50, 100, 200, 500]):
        # noise free spectrum
        s = self.starting_spectrum(slope, intercept, av)
        # noise free sed
        sed = self.mock_sed(s)

        # try different noise levels
        M = len(snrs)
        av_avgs = np.zeros(M)
        av_sigmas = np.zeros(M)

        # for each snr, generate mock seds with noise added
        for j, snr in enumerate(snrs):
            # number of realizations for each SNR
            N = 100
            slopes = np.zeros(N)
            intercepts = np.zeros(N)
            av_fits = np.zeros(N)
            for i in range(N):
                mock_sed_with_noise = self.add_noise(sed, snr)
                result = self.fit_obs(mock_sed_with_noise, snr)
                # print(i, result)
                slopes[i] = result.x[0]
                intercepts[i] = result.x[1]
                av_fits[i] = result.x[2]
            av_avgs[j] = np.average(av_fits)
            av_sigmas[j] = np.std(av_fits)
            print(f"SNR {snr} av {av_avgs[j]:.2f} sigma {av_sigmas[j]:.2f} ratio {1/(av_sigmas[j]/av_avgs[j]):.2f}")

        plt.errorbar(snrs, av_avgs, av_sigmas)
        plt.xlabel('snr')
        plt.ylabel('A(V) fit')

        # pd.DataFrame(
        #     data={
        #         "av_fixed": np.full(len(snrs), av),
        #         "snr": snrs,
        #         "av_fit_avg": av_avgs,
        #         "av_fit_std": av_sigmas,
        #     }
        # ).to_csv(f"av_{av:.2f}_MIRI.csv")


    def plot_spectrum_and_sed(self, s, sed):
        plt.plot(self.x, s)
        plt.scatter(sed.w, sed.f, marker='+')
