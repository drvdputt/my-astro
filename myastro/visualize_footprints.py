"""Stuff that comes from the SMC imaging photometry proposal. Should
probably split this up to the right places."""

from astropy.wcs import WCS
from astropy.io import fits
from argparse import ArgumentParser
from matplotlib import pyplot as plt
import numpy as np


def wcs_image(ax, image_array, image_wcs, percentile=None, **kwargs):
    # make the wcs plot

    if percentile is None:
        vmax_default = np.nanpercentile(image_array, 99.9)
    else:
        vmax_default = np.nanpercentile(image_array, percentile)

    im = ax.imshow(
        image_array,
        origin="lower",
        cmap="Blues",
        vmin=kwargs.get("vmin", 0),
        vmax=kwargs.get("vmax", vmax_default),
    )
    ax.set_xlabel("RA")
    ax.set_ylabel("Dec")
    # ax.coords[0].set_major_formatter('d.d')
    # ax.coords[0].set_format_unit(u.degree)
    # ax.coords[1].set_format_unit(u.degree)
    return im

ap = ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--image", help="fits file containing background image with WCS")
args = ap.parse_args()
files = args.files
image_fn = args.image

if image_fn is not None:
    with fits.open(image_fn) as hdu:
        image_array = hdu["SCI"].data
        image_wcs = WCS(hdu["SCI"])
else:
    image_array = None
    image_wcs = None

fig = plt.figure()
ax = fig.add_subplot(111)  # projection=image_wcs)

N = len(files)
centers = np.zeros((2, N))
# collect the pointing labels
labels = []
for i, f in enumerate(files):
    with fits.open(f) as hdulist:
        header = hdulist["SCI"].header
        centers[0][i] = header["RA_REF"]
        centers[1][i] = header["DEC_REF"]
        # act_ids.append(hdulist['PRIMARY'].header['ACT_ID'])
        labels.append(f.split("_")[1][:2])

catalog = {
    "RA": centers[0],
    "DEC": centers[1],
}

scatter_kwargs = dict(marker="+")
if image_array is not None:
    im = wcs_image(ax, image_array, image_wcs, percentile=10)
    show_sources_on_image(
        ax,
        catalog,
        image_array,
        image_wcs,
        scatter_kwargs=scatter_kwargs,
        labels=labels,
    )
    fig.colorbar(im, ax=ax)
else:
    ax.scatter(centers[0], centers[1], **scatter_kwargs)
    ax.set_xlabel("RA")
    ax.set_ylabel("DEC")

plt.show()

