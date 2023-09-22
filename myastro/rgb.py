"""Some tools to quickly make rgb images the way I like it."""

import numpy as np


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
