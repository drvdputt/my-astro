from astropy.coordinates import SkyCoord
from regions import PolygonSkyRegion, Regions, PixCoord
import astropy
from astropy import units as u
import numpy as np


def combine_regions(fn_out, fn1, fn2, *args):
    # take the third line of each, and combine
    with open(fn1, "r") as f1, open(fn2, "r") as f2, open(fn_out, "w") as fout:
        fout.writelines(f1.readlines())
        fout.write(f2.readlines()[2])
        for fnN in args:
            with open(fnN, "r") as fN:
                fout.write(fN.readlines()[2])


def draw_regions(ax, image_wcs, ds9_reg_files, reg_colors=None):
    """Region file can consist of multiple shapes"""
    # for i, fn in enumerate(ds9_reg_files):
    #     shapes = pyregion.open(fn)
    #     xy_shapes = shapes.as_imagecoord(header=image_wcs.celestial.to_header())
    #     patches, _ = xy_shapes.get_mpl_patches_texts()
    #     for p in patches:
    #         if reg_colors is not None:
    #             p.set_edgecolor(reg_colors[i])
    #         ax.add_patch(p)

    # alternate implementation with regions instead of pyregion, because
    # as_imagecoord is not working for a box exported from ds9
    for i, fn in enumerate(ds9_reg_files):
        regions = Regions.read(fn)
        # go over all region shapes in  the combined region
        for sub_region in regions:
            # convert to PixelRegion if a SkyRegion is given
            if hasattr(sub_region, "to_pixel"):
                pix_region = sub_region.to_pixel(image_wcs)
            else:
                pix_region = sub_region

            # use the built-in plot convenience
            pix_region.plot(
                ax=ax,
                # patch kwargs
                edgecolor=None if reg_colors is None else reg_colors[i],
            )


def make_rectangle_skycoord(center, w, h):
    """Returns skycoord containing corners"""
    right = center.directional_offset_by(-90 * u.degree, w / 2)
    topright = right.directional_offset_by(0 * u.degree, h / 2)
    bottomright = right.directional_offset_by(180 * u.degree, h / 2)

    left = center.directional_offset_by(90 * u.degree, w / 2)
    topleft = left.directional_offset_by(0 * u.degree, h / 2)
    bottomleft = left.directional_offset_by(180 * u.degree, h / 2)
    return astropy.coordinates.concatenate([bottomleft, bottomright, topright, topleft])


def skycoord_to_region(skycoord, fn):
    """example output:

    # Region file format: DS9 version 4.1
    global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
    fk5
    polygon(0:47:29.866,-73:16:01.54,0:47:08.130,-73:15:54.38,0:47:08.535,-73:14:22.20,0:46:47.230,-73:14:15.16,0:46:47.529,-73:12:41.15,0:46:00.166,-73:12:26.89,0:46:00.115,-73:13:57.61,0:45:40.568,-73:13:52.20,0:45:40.570,-73:15:15.12,0:45:19.821,-73:15:08.77,0:45:19.756,-73:16:27.30,0:44:55.929,-73:16:16.08,0:44:55.160,-73:17:47.25,0:44:35.645,-73:17:38.89,0:44:34.941,-73:19:07.49,0:44:15.074,-73:18:57.03,0:44:13.789,-73:22:26.68,0:44:35.073,-73:22:33.72,0:44:34.601,-73:24:06.85,0:44:55.891,-73:24:14.62,0:44:55.314,-73:25:45.92,0:45:43.211,-73:26:02.84,0:45:43.194,-73:24:34.67,0:46:03.312,-73:24:41.20,0:46:03.266,-73:23:13.03,0:46:23.559,-73:23:19.57,0:46:23.588,-73:22:05.43,0:46:44.474,-73:22:28.82,0:46:47.982,-73:21:07.38,0:47:09.088,-73:21:29.90,0:47:14.115,-73:19:24.10,0:47:28.880,-73:19:30.18) # color=green"""
    PolygonSkyRegion(vertices=skycoord).write(fn, overwrite=True)


def make_miri_rectangle(image_wcs, ra, dec):
    """
    Uses the above function for MIRI and our use case with ra and dec in decimal degrees.
    """
    center = SkyCoord(ra=ra * u.degree, dec=dec * u.degree)

    # dimension of miri imaging according to documentation
    mirix = 73.25 * u.arcsec
    miriy = 112.6 * u.arcsec

    fn = f"miri_{ra}_{dec}.reg"
    corners = make_rectangle_skycoord(center, mirix, miriy)
    skycoord_to_region(corners, fn)

    return fn


def make_sbc_rectangle(image_wcs, ra, dec):
    center = SkyCoord(ra=ra * u.degree, dec=dec * u.degree)
    sbcx = 31 * u.arcsec
    sbcy = 35 * u.arcsec
    fn = f"sbc_{ra:.3f}_{dec:.3f}.reg"
    corners = make_rectangle_skycoord(center, sbcx, sbcy)
    skycoord_to_region(corners, fn)
    return fn


def filter_sources_by_region(region_fn, catalog, image_wcs):
    """Region file can be in ra dec or in image coords.

    Catalog needs to have RA and DEC columns in decimal degrees"""
    regions = Regions.read(region_fn)
    pix_regions = [
        r.to_pixel(image_wcs) if hasattr(r, "to_pixel") else r for r in regions
    ]

    skycoords = SkyCoord(ra=catalog["RA"] * u.degree, dec=catalog["DEC"] * u.degree)
    pixcoords = PixCoord(*skycoords.to_pixel(image_wcs))
    mask = np.logical_or.reduce([ri.contains(pixcoords) for ri in pix_regions])
    return catalog[mask]

    # keeping old code cause i'm not under version control
    # r2 = r.as_imagecoord(image_wcs.to_header())
    # myfilter = r2.get_filter()
    # coords = image_wcs.celestial.world_to_pixel_values(catalog['RA'], catalog['DEC'])
    # x, y = coords[0], coords[1]
    # mask = np.array([0 != myfilter.inside1(x[i], y[i]) for i in range(len(catalog))])
    # return catalog[mask]
