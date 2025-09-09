"""This a a place to put the whole thing (a bit too extensive for
myastro, at least until this repo+mystro get merged."""

import matplotlib
from astropy.nddata import CCDData
import reproject
import numpy as np
from myastro import image, rgb
from astropy.visualization import make_lupton_rgb


class QRGB:
    """
    Attributes
    ----------

    rgb_method: str
        Options are "naive" and "lupton"

    offsets: percentile to be subtracted from data
    """

    def __init__(self, images_array):
        """
        Parameters
        ----------

        images_array: 3D array
            All images to be used for the RGB image, packed in one
            single 3D cube. Indices are: image, iy, ix

        """
        self.images = images_array
        self.nc = self.images.shape[0]

        # normalization options
        self.clip_pmin = 0
        self.clip_pmax = 100
        self.scale_pmax = 84
        self.offsets = [0] * self.nc

        # colors to use for data -> rgb conversion
        cmap = matplotlib.cm.get_cmap("hsv")
        self.colors = [cmap(i / self.nc) for i in range(self.nc)]

        # rgb combination options
        self.rgb_method = "naive"
        self.stretch = 1
        self.Q = 1

    @classmethod
    def read(cls, fits_files, wcs_index: int = 0):
        """Quickly set up QRGB by reading multiple fits files and
        reprojecting them all"""

        ccddata = [CCDData.read(fn) for fn in fits_files]

        images = []
        for i in range(len(ccddata)):
            # skip reprojection for the one we are basing the WCS on
            if i == wcs_index:
                images.append(ccddata[i].data)
                continue

            image, _ = reproject.reproject_interp(
                ccddata[i],
                ccddata[wcs_index].wcs,
                shape_out=ccddata[wcs_index].shape,
                order="nearest-neighbor",
            )
            images.append(image)

        images_array = np.stack(images, axis=0)
        return cls(images_array)

    def get_normalized_images(self):
        """Clean and renormalize the input data

        Used internally mostly, but you can call it to check what your
        clip, scale, and offset options are doing.

        Returns
        -------

        List of 2d arrays, same length as self.images
        """
        kwargs = dict(
            clip_pmin=self.clip_pmin,
            clip_pmax=self.clip_pmax,
            scale_pmax=self.scale_pmax,
        )
        output = np.zeros(self.images.shape)
        for i in range(self.nc):
            output[i] = image.normalize(
                self.images[i], offset_pmin=self.offsets[i], **kwargs
            )

        return output

    def get_rgb(self):
        """Apply the full algorithm to make imshowable rgb array

        1. normalize images
        2. combine arbitrary colors into RGB
        3. convert RGB to integer normalized RGB, with a choice of method

        Returns
        -------

        uint8 array, shape (ny, nx, rgb), compatible with imshow

        """
        normalized_images = self.get_normalized_images()
        r, g, b = rgb.combine_arbitrary_colors(normalized_images, self.colors)

        # apply curve/stretch and combine RGB
        if self.rgb_method == "lupton":
            rgb_uint8 = make_lupton_rgb(r, g, b, stretch=self.stretch, Q=self.Q)
        elif self.rgb_method == "naive":
            rgb_uint8 = rgb.make_naive_rgb(r, g, b, stretch=self.stretch, Q=self.Q)
        else:
            raise "unsupported color method"

        return rgb_uint8
