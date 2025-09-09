"""Some very common things to do with images, including plotting and comparing.

The functions here deal with monochromatic image data (simple 2d
arrays). Anything related to combining images into colors, should go
into rgb.py.

Idea: write them so that they all use the CCDData class from
astropy.nddata? They have a wcs property which is useful.

"""

from myastro import plot
from matplotlib import pyplot as plt
import numpy as np


def plot_many(images, ncols=1, **kwargs):
    nrows = len(images) // ncols
    if nrows * ncols < len(images):
        nrows += 1
    fig, axs = plt.subplots(nrows, ncols)
    for image, ax in zip(images, axs.flatten()):
        plot.nice_imshow(ax, image.data, **kwargs)

    return fig, axs


def plot_residual(a1, a2):
    """Show two arrays and their difference.

    Layout is reasonable for square images.

    Note: this is put in `image`, rather than `plot`, because it creates
    it's own layout. The stuff in `plot`, needs to operate on a single
    axes ideally.

    """
    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
    kwargs = dict(interpolation="nearest")
    plot.nice_imshow(ax[0], a1, **kwargs)
    plot.nice_imshow(ax[1], a2, **kwargs)
    plot.nice_imshow(ax[2], a2 - a1, **kwargs)
    ax[2].set_title("image 2 - image 1")
    fig.set_size_inches(18, 7)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0)
    return fig, ax


def normalize(image, clip_pmin=0, clip_pmax=100, offset_pmin=0, scale_pmax=84):
    """Apply cleaning and move images to the order of magnitude 1

    This happens before any curves are applied. The parameters should be
    tuned so that the curve to be applied in the future has the best
    effect.

    For an ashinh curve, the data should be transformed so that the most
    important part of their dynamic range falls around 0.0-1.0

    Steps:

    1. First clip using clip_pmin and clip_pmax as cutoff percentiles.
    Everything below pmin is set to the pmin percentile flux value.
    Everything above pmax is set to the pmax percentile flux value.

    2. Subtract offset using the offset percentile. Everything below the
    p'th percentile is now forced to zero, and the p'th percentile is
    now the zero point. Useful to clip off a uniform background.

    3. Rescale the data so that all points between the new zero point
    and the scale_pmax percentile run from 0 to 1. Everything above
    scale_pmax will still be greater than 1, and therefore ends up in
    the log regime. The idea is that the most important part of the
    dynamic range can be placed in the linear regime, by choosing these
    values wisely.

    Returns
    -------
    modified image : array of floats

    """
    if clip_pmax <= clip_pmin:
        raise ValueError(f"Invalid clip percentiles ({clip_pmin} and {clip_pmax})")
    if scale_pmax <= offset_pmin:
        raise ValueError(
            f"Invalid offsets and scale percentiles ({offset_pmin} and {scale_pmax})"
        )

    # all percentiles used here
    cmin = np.nanpercentile(image, clip_pmin)
    cmax = np.nanpercentile(image, clip_pmax)
    offset = np.nanpercentile(image, offset_pmin)
    scale = np.nanpercentile(image, scale_pmax) - offset

    # apply clip
    new_image = np.maximum(image, cmin)
    new_image = np.minimum(new_image, cmax)

    # apply offset, zero clip, and scale
    new_image = np.maximum(new_image - offset, 0) / scale
    return new_image
