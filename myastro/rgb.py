"""Some tools to quickly make rgb images the way I like it."""

import numpy as np


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
