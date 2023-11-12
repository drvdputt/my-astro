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

def dumb_scaling(image):
    """Scales a monochromatic image to the 0 - 255 range

    Returns
    -------
    array of float

    """
    new_image = image.copy()
    new_image = np.sqrt(image)
    imin = np.nanpercentile(new_image, 0)
    span = np.nanpercentile(new_image, 100) - imin
    new_image = (new_image - imin) / span * 255
    return new_image


def normalize(
    image, clip_pmin=0, clip_pmax=100, scale_pmin=16, scale_pmax=84, offset_p=50
):
    """Bring images into a similar order of magnitude.

    Rescales using a width, determined given the scale percentile range

    Clips the image using the given percentiles

    Centers the image values by subtracting the offset percentile offset_p

    Returns
    -------
    modified image : array of floats

    """
    cmin = np.nanpercentile(image, clip_pmin)
    cmax = np.nanpercentile(image, clip_pmax)
    image = np.maximum(image, cmin)
    image = np.minimum(image, cmax)

    width = np.nanpercentile(image, scale_pmax) - np.nanpercentile(image, scale_pmin)
    new_image = (image - np.nanpercentile(image, offset_p)) / width
    return new_image
