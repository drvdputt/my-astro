"""Some tools to quickly make rgb images the way I like it."""

import numpy as np
import matplotlib


def asinh_curve(image, scale, offset):
    """curve(x) = asinh(scale * (x - offset))

    *for the image*

    Which value is black? We want to curve to go through zero, and it's
    best to always set the black value to 0. The "offset" above is not
    included in the lower limit, so that a different part of the data
    can be brought into the linear regime near 0.

    Which value is maximum? curve(cutoff - offset). The plot command is
    then typically

    imshow(curve(image_array), vmin=0, vmax=curve(cutoff))

    To subtract a background our something similar, the offset should be
    tweaked instead of moving vmin away from 0.

    To brighten the low-flux areas, increase the scale value

    """
    return np.arcsinh(scale * (image - offset))


def scale_asinh_255(image, stretch=5, Q=8):
    """Apply asinh stretch, scales monochromatic image to 0 - 255 range.

    Assumes that the data have already been cleaned
    For the most predictable results, remove negative values from your
    data at the start, by clipping or applying a suitable bias yourself.

    If there are still negative values, they will be clipped (set to 0)
    here.

    Parameters
    ----------

    image: 2d array

    Returns
    -------
    uint8 2d array

    """
    new_image = image.copy()
    new_image = Q * np.arcsinh(Q * image / stretch)
    # configurable percentiles later?

    # ensure there are no negative values
    new_image[new_image < 0] = 0

    new_image = new_image / np.nanmax(new_image) * 255
    return new_image.astype("uint8")


def make_naive_rgb(image_r, image_g, image_b, stretch=5, Q=8):
    """Naive version with similar signature as lupton rgb function.

    Even though this is less informative, it looks better aesthetically
    sometimes.

    We try to recreate some of the same arguments for the curve, but the
    stretch here is deliberately NOT color preserving.

    """
    rgb = [scale_asinh_255(i, stretch, Q) for i in (image_r, image_g, image_b)]
    return np.stack(rgb, axis=-1)


def combine_arbitrary_colors(images, colors):
    """Make 3-channel rgb from many channels with different colors.

    Parameters
    ----------

    image_list: list of 2d arrays (same shape)
        Preferably, these input monochromatic images are already
        processed by something like image.normalize. For the best
        results, the images should be of the same order of magnitude,
        and should not have extreme outliers.

    color_list: list of colors
        Can be anything compatible with matplotlib.colors.to_rgba()

    Returns
    -------

    list of 3 float arrays

    Note: these return values still need to be processed by make_naive_rgb or make_lupton_rgb

    """
    nx, ny = images[0].shape
    shape = (3, nx, ny)
    rgb_totals = np.zeros(shape)
    for i in range(len(images)):
        rgba = matplotlib.colors.to_rgba(colors[i])
        for c in range(3):
            rgb_totals[c] += images[i] * rgba[c]

    # return as list for consistency with the rest of the code
    return [array for array in rgb_totals]


def palette(num_colors, cmap="hsv"):
    """Create colors based on colormap

    Returns
    -------

    list of RGBA 4-tuples

    """
    # colors to use for data -> rgb conversion
    cmap = matplotlib.cm.get_cmap(cmap)
    return [cmap(i / num_colors) for i in range(num_colors)]
