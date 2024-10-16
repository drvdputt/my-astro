"""Some very common things to do with images, including plotting and comparing.

The functions here deal with monochromatic image data (simple 2d
arrays). Anything related to combining images into colors, should go
into rgb.py.

"""

from myastro import plot
from matplotlib import pyplot as plt
import numpy as np

def plot_residual(a1, a2):
    """Show two arrays and their difference.

    Layout is reasonable for square images.

    Note: this is put in `image`, rather than `plot`, because it creates
    it's own layout. The stuff in `plot`, needs to operate on a single
    axes ideally.

    """
    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
    kwargs = dict(interpolation='nearest')
    plot.nice_imshow(ax[0], a1, **kwargs)
    plot.nice_imshow(ax[1], a2, **kwargs)
    plot.nice_imshow(ax[2], a2 - a1, **kwargs)
    ax[2].set_title("image 2 - image 1")
    fig.set_size_inches(18, 7)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0)
    return fig, ax

def normalize(
    channel_array, clip_pmin=0, clip_pmax=100, scale_pmin=16, scale_pmax=84, offset_p=0
):
    """Apply cleaning and move images to the order of magnitude 1

    Steps:

    1. Clip using clip_pmin and clip_pmax as cutoff percentiles.
    Everything below pmin is set to the pmin percentile flux value.
    Everything above pmax is set to the pmax percentile flux value.

    2. Subtracts offset using the offset percentile. Everying below
    offset_p will be set to 0 (TODO: or negative? What is best?)

    3. Rescale the data so that all points between the scale_pmin and
    scale_pmax percentiles run from 0 to 1. Everything above scale_pmax
    will still be greater than 1. The idea is that the most important
    part of the dynamic range happens around 0.5.

    Returns
    -------
    modified image : array of floats

    """
    cmin = np.nanpercentile(channel_array, clip_pmin)
    cmax = np.nanpercentile(channel_array, clip_pmax)
    image = np.maximum(channel_array, cmin)
    image = np.minimum(image, cmax)

    width = np.nanpercentile(image, scale_pmax) - np.nanpercentile(image, scale_pmin)
    new_image = (image - np.nanpercentile(image, offset_p)) / width
    return new_image
