"""Constants and regions corresponding to footprints of instruments"""

from astropy.coordinates import SkyCoord
from myastro.regionhacks import make_rectangle_skycoord, skycoord_to_region
from astropy import units as u

# dimensions of miri imaging according to documentation
mirix = 73.25 * u.arcsec
miriy = 112.6 * u.arcsec

# dimensions of acs/sbc fuv imaging
sbcx = 31 * u.arcsec
sbcy = 35 * u.arcsec

def make_miri_rectangle(image_wcs, ra, dec):
    """
    Uses the above function for MIRI and our use case with ra and dec in decimal degrees.
    """
    center = SkyCoord(ra=ra * u.degree, dec=dec * u.degree)

    fn = f"miri_{ra}_{dec}.reg"
    corners = make_rectangle_skycoord(center, mirix, miriy)
    skycoord_to_region(corners, fn)

    return fn


def make_sbc_rectangle(image_wcs, ra, dec):
    center = SkyCoord(ra=ra * u.degree, dec=dec * u.degree)
    fn = f"sbc_{ra:.3f}_{dec:.3f}.reg"
    corners = make_rectangle_skycoord(center, sbcx, sbcy)
    skycoord_to_region(corners, fn)
    return fn


