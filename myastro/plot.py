import numpy as np
from matplotlib.colors import LogNorm
import myastro.wcshacks
from regions import SkyRegion
from astropy import units as u
import math
from matplotlib import patheffects

# some useful kwarg collections

text_white_black_outline_kwargs = {
    "path_effects": [patheffects.withStroke(linewidth=2, foreground="k")],
}


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


from myastro.wcshacks import xy_span_arcsec
from matplotlib import ticker


def physical_ticklabels(
    ax,
    image_array,
    celestial_wcs,
    x_angle_values=None,
    y_angle_values=None,
    angle_unit=u.arcsec,
):
    """Modify the ticks and labels on an axis to show angular scale.

    Keep the image in pixel coordinates, but change the ticks and labels
    to that they match whole steps in an angular unit of choice.

    Choice of angular unit not implemented, just defaults to arcsec
    right now.

    Parameters
    ----------

    """
    ny, nx = image_array.shape
    angle_x, angle_y = xy_span_arcsec([nx, ny], celestial_wcs)
    print(f"Span is x: {angle_x} and y: {angle_y}")
    sx, sy = angle_x.to(angle_unit).value, angle_y.to(angle_unit).value

    # convert desired arcsec steps to tick locations
    angles_x = (
        np.array(range(0, int(math.ceil(sx))))
        if x_angle_values is None
        else x_angle_values
    )
    angles_y = (
        np.array(range(0, int(math.ceil(sy))))
        if y_angle_values is None
        else y_angle_values
    )
    xticks = angles_x / sx * nx
    yticks = angles_y / sy * ny
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    # ticker label format function that converts tick locations to
    # labels in arcsec units
    xformatter = ticker.FuncFormatter(lambda x, _: str(int(x / nx * sx)))
    yformatter = ticker.FuncFormatter(lambda y, _: str(int(y / ny * sy)))
    ax.xaxis.set_major_formatter(xformatter)
    ax.yaxis.set_major_formatter(yformatter)

    return {
        "xticks": xticks,
        "yticks": yticks,
        "xtickformatter": xformatter,
        "ytickformatter": yformatter,
    }


def nice_colorbar(fig, ax, mappable):
    """Colorbar nicely next to plot. Works well with imshow.

    https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph"""
    cax = fig.add_axes(
        [
            ax.get_position().x1 + 0.01,
            ax.get_position().y0,
            0.02,
            ax.get_position().height,
        ]
    )
    cb = fig.colorbar(mappable, cax=cax)  # Similar to fig.colorbar(im, cax = cax)
    return cb, cax
