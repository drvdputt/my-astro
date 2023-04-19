import numpy as np
from matplotlib.colors import LogNorm
import myastro.wcshacks
from regions import SkyRegion


def plot_s1d(ax, s, add_labels=True, offset=None, **kwargs):
    default_kwargs = dict(linewidth=0.5, drawstyle="steps")

    w = s.spectral_axis.value
    if offset is not None:
        w -= offset
    ax.plot(w, s.flux.value, **(default_kwargs | kwargs))

    if add_labels:
        ax.set_xlabel(f"wavelength ({s.spectral_axis.unit})")
        ax.set_ylabel(f"flux ({s.flux.unit})")


def region(
    ax,
    sky_region: SkyRegion,
    celestial_wcs,
    annotation_text=None,
    annotation_kwargs={},
    **kwargs,
):
    """Draw region onto image, starting from sky coordinates.

    If combined with the right WCS, the xy coordinates of the plotted
    region will nicely match the output of imshow, both with a normal
    Axes and with a WCSAxes.

    Parameters
    ----------

    region: SkyRegion object from the "regions" package

    celestial: WCS to convert to the right pixel coordinates

    text: add text at the center of the region

    """
    pix_region = sky_region.to_pixel(celestial_wcs)
    pix_region.plot(
        ax=ax,
        # same kwargs as patch
        **kwargs,
    )
    if annotation_text is not None:
        # ax.text(pix_region.center.x, pix_region.center.y, text, **text_kwargs)
        ax.annotate(
            annotation_text,
            (pix_region.center.x, pix_region.center.y),
            **annotation_kwargs,
        )


def nice_imshow(ax, a, log_color=False, **kwargs):
    pmin = 1
    pmax = 99
    default_kwargs = dict(
        origin="lower",
        vmin=np.nanpercentile(a, pmin),
        vmax=np.nanpercentile(a, pmax),
    )

    all_kwargs = default_kwargs | kwargs

    if log_color:
        vmin = np.amin(a[a > 0])
        all_kwargs["norm"] = LogNorm(vmin=vmin, vmax=all_kwargs.get("vmax", None))

    return ax.imshow(a, **all_kwargs)


def rotated_imshow(ax, image_array, celestial_wcs, rotate_angle, **imshow_kwargs):
    """Rotate image and wcs, display and return them."""
    image_rot, wcs_rot = myastro.wcshacks.rotate_image_and_wcs(
        image_array, celestial_wcs, rotate_angle, autocrop=True
    )
    return {
        "imshow": nice_imshow(ax, image_rot, **imshow_kwargs),
        "image": image_rot,
        "wcs": wcs_rot,
    }
