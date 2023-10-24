"""Constants and regions corresponding to footprints of instruments"""

from myastro.regionhacks import make_rectangle_region, combine_regions
from astropy import units as u
from astropy.coordinates import SkyCoord

# dimensions of nircam imaging
nircamx = 129 * u.arcsec
nircamy = 129 * u.arcsec
nircamgap = 44 * u.arcsec

# dimensions of miri imaging according to documentation
mirix = 73.25 * u.arcsec
miriy = 112.6 * u.arcsec

# dimensions of acs/sbc fuv imaging
sbcx = 31 * u.arcsec
sbcy = 35 * u.arcsec


def make_nircam_both(ra, dec, pa):
    """pa: Quantity (angle)
        position angle of module b relative to module 1.

    I'm not doing a proper rotation of the regions yet. Both will be
    RA-DEC oriented. But B will be offset according to the module gap
    and the given position angle.

    0 = right, 90 = up, 180 = left, 270 = down

    Writes a file "nircam_both_ra_dec.reg" to disk, containing the two
    rectangles in sky coordinates

    """
    fna = f"nircam_a_{ra}_{dec}.reg"
    fnb = f"nircam_b_{ra}_{dec}.reg"
    fn = f"nircam_both_{ra}_{dec}.reg"

    reg_a = make_rectangle_region(ra, dec, nircamx, nircamy)
    reg_a.write(fna)

    center_a = SkyCoord(ra=ra * u.degree, dec=dec * u.degree)
    center_b = center_a.directional_offset_by(pa, nircamgap)

    reg_b = make_rectangle_region(center_b.ra, center_b.dec, nircamx, nircamy)
    reg_b.write(fnb)

    combine_regions(fn, fna, fnb)


def make_nircam_rectangle(ra, dec):
    """Returns file name"""
    fn = f"nircam_{ra}_{dec}.reg"
    polygon_sky_region = make_rectangle_region(ra, dec, nircamx, nircamy)
    polygon_sky_region.write(fn, overwrite=True)

    return fn


def make_miri_rectangle(ra, dec):
    """
    Uses the above function for MIRI and our use case with ra and dec in decimal degrees.
    """
    fn = f"miri_{ra}_{dec}.reg"
    polygon_sky_region = make_rectangle_region(ra, dec, mirix, miriy)
    polygon_sky_region.write(fn, overwrite=True)
    return fn


def make_sbc_rectangle(ra, dec):
    fn = f"sbc_{ra:.3f}_{dec:.3f}.reg"
    polygon_sky_region = make_rectangle_region(ra, dec, sbcx, sbcy)
    polygon_sky_region.write(fn, overwrite=True)
    return fn
