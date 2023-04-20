from scipy import ndimage
import numpy as np
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord
from astropy import units as u


def rotate_image_and_wcs(
    image_array, celestial_wcs: WCS, rotate_angle, autocrop, manual_crop=None
):
    """Rotate image, remove empty space, adjust wcs

    Assumes that image is in a[y,x] format.

    Positive rotate_angle = clockwise rotation or array data. Array is
    expanded by numpy by default to fit the rotated image. That's why
    this method also crops to remove zero values.

    Code adjusted from PAHFITcube map collage.

    WCS adjustment: just rotate the PC matrix? Will only work if WCS
    uses PC matrix, and not one of the other rotation matrices.

    Sets CRVAL to the coordinates of the central pixel, which is useful
    because it is the center of rotation. The center of the new array is
    then also at CRVAL, so it becomes straightforward to adjust CRPIX.

    Returns
    -------

    """
    ## Step 1: rotation

    # rotation of data
    image_rot = ndimage.rotate(image_array, rotate_angle)
    print(f"{image_array.shape} --> {image_rot.shape}")
    # clean up after rotation. Cut off everything below smallest nonzero
    # value that was in the original array. The crop method below needs
    # out-of-bounds to be exactly zero, while the rotation above can
    # cause some artifacts.
    nonzero = image_array > 0
    if nonzero.any():
        vmin_nonzero = np.amin(image_array[nonzero])
    else:
        # avoid problems when everything is zero
        vmin_nonzero = 0
    image_rot[image_rot < vmin_nonzero] = 0

    # rotate the pc matrix
    radians = rotate_angle * np.pi / 180
    c = np.cos(radians)
    s = np.sin(radians)
    R = np.array([[c, -s], [s, c]])
    # R vs R.T was found through trial and error -_-.
    # pc_rot = R.dot(celestial_wcs.wcs.pc).dot(R.T)
    pc_rot = R.T.dot(celestial_wcs.wcs.pc)
    print(celestial_wcs.wcs.pc)
    print(pc_rot)

    # ndimage.rotate rotates around the central pixel. So if we set
    # CRVAL to that point, we just need to adjust CRPIX according to the
    # crop translation. Array center is y x, but one slice in the cube
    # i'm working with actually has x as the first index. As always with
    # FITS, ye olde trial and error.
    old_center = (image_array.shape[1] // 2, image_array.shape[0] // 2)
    center_of_rotation = celestial_wcs.array_index_to_world_values((old_center,))[0]
    # crpix is x y
    new_crpix = (image_rot.shape[1] // 2, image_rot.shape[0] // 2)

    # Save these values into a new WCS
    new_wcs = celestial_wcs.deepcopy()
    new_wcs.wcs.pc = pc_rot
    new_wcs.wcs.crval = center_of_rotation
    new_wcs.wcs.crpix = new_crpix

    # optional crop after rotation
    cropped = False
    if autocrop:
        keep_i = np.where(np.sum(np.square(image_rot), axis=1) != 0)[0]
        keep_j = np.where(np.sum(np.square(image_rot), axis=0) != 0)[0]
        if len(keep_i) > 2 and len(keep_j) > 2:
            min_i = keep_i[0]
            max_i = keep_i[-1]
            min_j = keep_j[0]
            max_j = keep_j[-1]
            cropped = True
            print("Suggested crop range: ", (min_i, max_i, min_j, max_j))
        else:
            print("Something weird with data! Skipping autocrop")
    elif manual_crop is not None:
        min_i, max_i, min_j, max_j = manual_crop
        cropped = True

    if cropped:
        image_rot = image_rot[min_i:max_i, min_j:max_j]

    # translation of coordinates
    if cropped:
        crop_translate_xy = np.array([-min_j, -min_i])
    else:
        crop_translate_xy = np.zeros(2)

    new_wcs.wcs.crpix = np.array(new_wcs.wcs.crpix) + crop_translate_xy
    # this is also yx
    new_wcs.array_shape = image_rot.shape

    return image_rot, new_wcs


def xy_span_arcsec(xy_shape, celestial_wcs):
    """Compute field of view size in arc seconds for x and y direction

    Distances are calculated using angular separation between (0, 0) and
    (nx - 1, 0) for x, and between (nx - 1, 0) and (nx - 1, ny - 1) for
    y.

    Parameters
    ----------

    xy_shape: nx, ny

    celestial_wcs: WCS used for converting x and y (in pixel units) to
    physical positions.

    Returns
    -------
    physical x span (arcsec)

    physical y span (arcsec)

    wcs

    """
    xmax = xy_shape[0] - 1
    ymax = xy_shape[1] - 1
    x_corners = [0, xmax, xmax, 0]
    y_corners = [0, 0, ymax, ymax]

    print(x_corners)
    print(y_corners)
    corners = pixel_to_skycoord(xp=x_corners, yp=y_corners, wcs=celestial_wcs)
    #  ^
    #  |
    #  y
    #  3 .... 2
    #
    #
    #  0 .... 1 x-->
    pixel_scale_x = corners[0].separation(corners[1]).to(u.arcsec)
    pixel_scale_y = corners[1].separation(corners[2]).to(u.arcsec)
    return pixel_scale_x, pixel_scale_y
