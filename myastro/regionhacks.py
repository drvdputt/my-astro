from astropy.coordinates import SkyCoord
from regions import (
    PolygonSkyRegion,
    Regions,
    PixCoord,
    CircleSkyRegion,
    RectangleSkyRegion,
    LineSkyRegion,
    PixelRegion,
)
import astropy
from astropy import units as u
import numpy as np
import myastro.plot
from photutils.aperture import SkyRectangularAperture, SkyCircularAperture


def find_center(r: PixelRegion):
    if hasattr(r, "center"):
        return r.center
    elif hasattr(r, "vertices"):
        x, y = r.vertices.xy
        return (np.mean(x), np.mean(y))


def circle_to_aperture(r: CircleSkyRegion):
    """Read DS9 region file and convert to photutils circular aperture

    Only works if file contains circle.

    Only the first region in the file is parsed.
    """
    return SkyCircularAperture(r.center, r=r.radius)


def rectangle_to_aperture(r: RectangleSkyRegion):
    """Read DS9 region file and convert to photutils rectangular aperture

    CAVEAT: is the SkyRectangularAperture theta the same as RectangleSkyRegion angle?

    From photutils doc: theta = The position angle (in angular units) of
    the rectangle “width” side. For a right-handed world coordinate
    system, the position angle increases counterclockwise from North
    (PA=0).

    -> width aligned with north then theta = 0
    -> theta is PA of width

    From regions doc: angle = The rotation angle of the rectangle,
    measured anti-clockwise. If set to zero (the default), the width
    axis is lined up with the longitude axis of the celestial
    coordinate.

    -> width aligned with west when angle = 0...
    -> angle is PA of height

    Add 90 degrees to fix this?

    """
    return SkyRectangularAperture(
        r.center, r.width, r.height, theta=r.angle + 90 * u.degree
    )


def skyregion_to_aperture_auto(r):
    if isinstance(r, CircleSkyRegion):
        return circle_to_aperture(r)
    elif isinstance(r, RectangleSkyRegion):
        return rectangle_to_aperture(r)
    else:
        return None


def combine_regions(fn_out, fn1, fn2, *args):
    # take the third line of each, and combine
    with open(fn1, "r") as f1, open(fn2, "r") as f2, open(fn_out, "w") as fout:
        fout.writelines(f1.readlines())
        fout.write(f2.readlines()[2])
        for fnN in args:
            with open(fnN, "r") as fN:
                fout.write(fN.readlines()[2])


def load_all_regions(ds9_reg_fn):
    regions = Regions.read(ds9_reg_fn)
    return [r for r in regions]


def draw_all_regions(ax, image_wcs, ds9_reg_files, reg_colors=None):
    """Region file can consist of multiple shapes"""
    for i, fn in enumerate(ds9_reg_files):
        regions = Regions.read(fn)
        # go over all region shapes in  the combined region
        for sub_region in regions:
            edgecolor = None if reg_colors is None else reg_colors[i]
            myastro.plot.region(ax, sub_region, image_wcs, edgecolor=edgecolor)


def make_rectangle_skycoord(center, w, h):
    """Returns skycoord containing corners of rectangle.

    Parameters
    ----------

    center: SkyCoord

    w: Quantity (angle)

    h: Quantity (angle)

    Returns
    -------
    SkyCoord of length 4, containing the bottom left, bottom right, top
    right, and top left coordinates

    """
    right = center.directional_offset_by(-90 * u.degree, w / 2)
    topright = right.directional_offset_by(0 * u.degree, h / 2)
    bottomright = right.directional_offset_by(180 * u.degree, h / 2)

    left = center.directional_offset_by(90 * u.degree, w / 2)
    topleft = left.directional_offset_by(0 * u.degree, h / 2)
    bottomleft = left.directional_offset_by(180 * u.degree, h / 2)
    return astropy.coordinates.concatenate([bottomleft, bottomright, topright, topleft])


def skycoord_to_region(skycoord):
    """
    Convert a list of sky coordinates to a PolygonSkyRegion

    Parameters
    ----------
    skycoord: SkyCoord

    Returns
    -------
    PolygonSkyRegion

    example output when written to disk

    # Region file format: DS9 version 4.1
    global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
    fk5
    polygon(0:47:29.866,-73:16:01.54,0:47:08.130,-73:15:54.38,0:47:08.535,-73:14:22.20,0:46:47.230,-73:14:15.16,0:46:47.529,-73:12:41.15,0:46:00.166,-73:12:26.89,0:46:00.115,-73:13:57.61,0:45:40.568,-73:13:52.20,0:45:40.570,-73:15:15.12,0:45:19.821,-73:15:08.77,0:45:19.756,-73:16:27.30,0:44:55.929,-73:16:16.08,0:44:55.160,-73:17:47.25,0:44:35.645,-73:17:38.89,0:44:34.941,-73:19:07.49,0:44:15.074,-73:18:57.03,0:44:13.789,-73:22:26.68,0:44:35.073,-73:22:33.72,0:44:34.601,-73:24:06.85,0:44:55.891,-73:24:14.62,0:44:55.314,-73:25:45.92,0:45:43.211,-73:26:02.84,0:45:43.194,-73:24:34.67,0:46:03.312,-73:24:41.20,0:46:03.266,-73:23:13.03,0:46:23.559,-73:23:19.57,0:46:23.588,-73:22:05.43,0:46:44.474,-73:22:28.82,0:46:47.982,-73:21:07.38,0:47:09.088,-73:21:29.90,0:47:14.115,-73:19:24.10,0:47:28.880,-73:19:30.18) # color=green
    """
    return PolygonSkyRegion(vertices=skycoord)


def make_rectangle_region(ra, dec, w, h):
    """Make a SkyRegion corresponding to a rectangular detector of a
    certain angular size.

    This probably only works for small width and height (in angle
    units). But no one is using wide angle lenses anyway.

    Parameters
    ----------
    ra: float
        RA of the center, decimal degrees

    dec: float
        DEC of the center, decimal degrees

    w: Quantity (angle)
        Width in any angle unit

    h: Quantity (angle)
        Height in any angle unit

    Returns
    -------
    PolygonSkyRegion

    """
    center = SkyCoord(ra=ra * u.degree, dec=dec * u.degree)
    corners = make_rectangle_skycoord(center, w, h)
    return skycoord_to_region(corners)


def make_compass_region(ra, dec, size):
    """Draw a compass on an image.

    Parameters
    ----------

    ra, dec: center of the compass in decimal degrees

    size_on_sky: size in arcsec

    wcs: the WCS used to convert the compass to pixel coordinates

    Returns
    -------

    Two LineSkyRegion objects (from regions package).

    center_north: line piece pointing north

    center_east: line piece pointing east

    These can be plotted using myastro.plot.region().

    """
    center = SkyCoord(ra, dec, unit="deg", frame="icrs")
    north = center.directional_offset_by(0 * u.degree, size * u.arcsec)
    east = center.directional_offset_by(90 * u.degree, size * u.arcsec)

    center_north = LineSkyRegion(center, north)
    center_east = LineSkyRegion(center, east)

    return center_north, center_east


def filter_sources_by_region(region_fn, ras, decs, image_wcs):
    """Region file can be in ra dec or in image coords.

    Image wcs needed because internal conversion to pixel coords happens

    ras: array

    decs: array

    returns: boolean mask

    """
    regions = Regions.read(region_fn)
    pix_regions = [
        r.to_pixel(image_wcs) if hasattr(r, "to_pixel") else r for r in regions
    ]

    skycoords = SkyCoord(ra=ras * u.degree, dec=decs * u.degree)
    pixcoords = PixCoord(*skycoords.to_pixel(image_wcs))
    mask = np.logical_or.reduce([ri.contains(pixcoords) for ri in pix_regions])
    return mask

    # keeping old code cause i'm not under version control
    # r2 = r.as_imagecoord(image_wcs.to_header())
    # myfilter = r2.get_filter()
    # coords = image_wcs.celestial.world_to_pixel_values(catalog['RA'], catalog['DEC'])
    # x, y = coords[0], coords[1]
    # mask = np.array([0 != myfilter.inside1(x[i], y[i]) for i in range(len(catalog))])
    # return catalog[mask]
