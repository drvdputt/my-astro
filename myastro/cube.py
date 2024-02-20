"""
Some common things, mostly based on specutils.

E.g. Collapsing, but also collapsing the uncertainty.
"""

import numpy as np
from astropy.nddata import StdDevUncertainty
from copy import deepcopy
from jwst.datamodels import IFUCubeModel
from jwst.assign_wcs.pointing import create_fitswcs


def collapse_flux_and_unc(s3d):
    """

    s3d: 3D Spectrum1D cube with uncertainty

    returns: Spectrum1D
        Single spectrum

    """
    spec_avg = s3d.collapse(method=np.nanmean, axis="spatial")
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


def write_cube_wavetab_jwst_s3d_format(
    fits_fn, flux_array, unc_array, wav_array, celestial_wcs
):
    """Write cube in same format as jwst pipeline.

    The resulting fits file will be loadable by
    specutils.Spectrum1D.read with format="JWST s3d"

    fits_fn: str

    flux_array: array of shape (num_wav, num_y, num_x)
       In units MJy / sr (hardcoded)

    unc_array: array of shape (num_wav, num_y, num_x)
       In units MJy / sr (hardcoded)

    wav_array: array of shape (num_wav,)
       In units micron (hardcoded)

    celestial_wcs: 2D wcs (astropy.wcs.WCS class)

    """
    # first set up all the wcsinfo. This info will be used by a utility
    # function from the jwst package to set up a GWCS object. Once this
    # object and some metadata have set as members of the IFUCubeModel,
    # the latter will be written out in a format that specutils
    # understands.
    # first, the crucial part for the wavelength table

    ifucube_model = IFUCubeModel(
        data=flux_array,
        err=unc_array,
        wavetable=np.array(
            [(wav_array[None].T,)], dtype=[("wavelength", "<f4", (len(wav_array), 1))]
        ),
    )
    ifucube_model.meta.wcsinfo.ctype3 = "WAVE-TAB"
    ifucube_model.meta.wcsinfo.ps3_0 = "WCS-TABLE"
    ifucube_model.meta.wcsinfo.ps3_1 = "wavelength"
    ifucube_model.meta.wcsinfo.crval3 = 1.0
    ifucube_model.meta.wcsinfo.crpix3 = 1.0
    ifucube_model.meta.wcsinfo.cdelt3 = None
    ifucube_model.meta.wcsinfo.cunit3 = "um"
    ifucube_model.meta.wcsinfo.pc3_1 = 0.0
    ifucube_model.meta.wcsinfo.pc1_3 = 0.0
    ifucube_model.meta.wcsinfo.pc3_2 = 0.0
    ifucube_model.meta.wcsinfo.pc2_3 = 0.0
    ifucube_model.meta.wcsinfo.pc3_3 = 1.0
    ifucube_model.meta.wcsinfo.wcsaxes = 3
    ifucube_model.wavedim = "(1,{:d})".format(len(wav_array))

    # then, the celestial wcs info
    ifucube_model.meta.wcsinfo.crval1 = celestial_wcs.wcs.crval[0]
    ifucube_model.meta.wcsinfo.crval2 = celestial_wcs.wcs.crval[1]
    ifucube_model.meta.wcsinfo.crpix1 = celestial_wcs.wcs.crpix[0]
    ifucube_model.meta.wcsinfo.crpix2 = celestial_wcs.wcs.crpix[1]
    ifucube_model.meta.wcsinfo.cdelt1 = celestial_wcs.wcs.cdelt[0]
    ifucube_model.meta.wcsinfo.cdelt2 = celestial_wcs.wcs.cdelt[1]
    ifucube_model.meta.wcsinfo.ctype1 = "RA---TAN"
    ifucube_model.meta.wcsinfo.ctype2 = "DEC--TAN"
    ifucube_model.meta.wcsinfo.cunit1 = "deg"
    ifucube_model.meta.wcsinfo.cunit2 = "deg"

    pc = celestial_wcs.wcs.get_pc()
    ifucube_model.meta.wcsinfo.pc1_1 = pc[0, 0]
    ifucube_model.meta.wcsinfo.pc1_2 = pc[0, 1]
    ifucube_model.meta.wcsinfo.pc2_1 = pc[1, 0]
    ifucube_model.meta.wcsinfo.pc2_2 = pc[1, 1]

    ifucube_model.meta.ifu.flux_extension = "SCI"
    ifucube_model.meta.ifu.error_extension = "ERR"
    ifucube_model.meta.ifu.error_type = "ERR"

    ifucube_model.meta.bunit_data = "MJy / sr"
    ifucube_model.meta.bunit_err = "MJy / sr"

    ifucube_model.meta.wcs = create_fitswcs(ifucube_model)
    # seems like the reader also needs this one vvv. In the jwst package
    # code, it uses NAXIS1, NAXIS2, NAXIS3.
    ifucube_model.meta.wcs.bounding_box = (
        (0, flux_array.shape[2] - 1),
        (0, flux_array.shape[1] - 1),
        (0, flux_array.shape[0] - 1),
    )

    ifucube_model.write(fits_fn)
