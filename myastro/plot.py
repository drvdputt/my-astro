import numpy as np
from matplotlib.colors import LogNorm
import myastro.wcshacks
from regions import SkyRegion
from astropy import units as u
import math
from matplotlib import patheffects
from matplotlib import pyplot as plt
from matplotlib.colors import FuncNorm
from myastro import regionhacks

# some useful kwarg collections
text_white_black_outline_kwargs = {
    "path_effects": [patheffects.withStroke(linewidth=2, foreground="k")],
}


def s1d(ax, s, add_labels=True, offset=None, wav_unit=u.micron, **kwargs):
    default_kwargs = dict(linewidth=0.5, drawstyle="steps")

    w = s.spectral_axis.to(wav_unit).value
    if offset is not None:
        w -= offset
    ax.plot(w, s.flux.value, **(default_kwargs | kwargs))

    if add_labels:
        ax.set_xlabel(f"wavelength ({s.spectral_axis.unit:latex_inline})")
        ax.set_ylabel(f"flux ({s.flux.unit:latex_inline})")


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

    ax: Axes

    region: SkyRegion object from the "regions" package

    celestial: WCS to convert to the right pixel coordinates

    annotation_text: add text at the center of the region

    annotation_kwargs: arguments passed to annotate().
       e.g. annotation_kwargs={'color':'r'} to make the text red.

    """
    pix_region = sky_region.to_pixel(celestial_wcs)
    pix_region.plot(
        ax=ax,
        # same kwargs as patch
        **kwargs,
    )
    if annotation_text is not None:
        # ax.text(pix_region.center.x, pix_region.center.y, text, **text_kwargs)
        center = regionhacks.find_center(pix_region)
        ax.annotate(
            annotation_text,
            center,
            **annotation_kwargs,
        )


def make_asinh_FuncNorm(scale, offset, cutoff):
    def forward(x):
        np.arcsinh(scale * (x - offset))

    def reverse(y):
        np.sinh(y) / scale + offset

    return FuncNorm(functions=(forward, reverse), vmin=0, vmax=forward(cutoff))


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


# create custom 2D colorbar
# here's a good example https://stackoverflow.com/questions/49871436/scatterplot-with-continuous-bivariate-color-palette-in-python
def bivariate_imshow(
    ax, Z1, Z2, cmap1=plt.cm.Blues, cmap2=plt.cm.Reds, cax=None, **imshow_kwargs
):
    """Take two 2D arrays of the same size, two colormaps, and do imshow for the average color

    ax : axes for the data plot

    cax : axes for the 2D colorbar

    Z1, Z2 : 2D arrays
        data arrays to combine into one plot

    cmap1, cmap2 : color map object e.g. plt.cm.Reds
        color maps to mix
    """
    # Rescale values to fit into colormap range (0->255)

    def average_color(data1, data2):
        d1 = np.array(
            255 * (data1 - data1.min()) / (data1.max() - data1.min()), dtype=int
        )
        d2 = np.array(
            255 * (data2 - data2.min()) / (data2.max() - data2.min()), dtype=int
        )
        c1 = cmap1(d1)
        c2 = cmap2(d2)
        # Color for each point is average of two colors
        return np.sum([c1, c2], axis=0) / 2.0

    # do imshow
    ax.imshow(average_color(Z1, Z2), **imshow_kwargs)

    # add legend if cax is given
    if cax is not None:
        # arrays to construct legend
        C1 = np.linspace(np.amin(Z1), np.amax(Z1), 100)
        C2 = np.linspace(np.amin(Z2), np.amax(Z2), 100)
        CC1, CC2 = np.meshgrid(C1, C2)
        cax.imshow(average_color(CC1, CC2))


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
    x_offset=0,
    y_offset=0,
    angle_unit=u.arcsec,
):
    """Modify the ticks and labels on an axis to show angular scale.

    Keep the image in pixel coordinates, but change the ticks and labels
    to that they match whole steps in an angular unit of choice.

    Choice of angular unit not implemented, just defaults to arcsec
    right now.

    x_angle_values: ticks of choice (not RA/DEC, needs to be along the axes!)

    x_offset: shift visual zero point by subtracting this value from the tick labels

    y_offset: idem for y

    Warning: do this before setting xlim, if you want to keep things
    intuitive. The ticks are place from the 0,0 point of the image, not
    the current view.

    Parameters
    ----------

    """
    ny, nx = image_array.shape
    span_x, span_y = xy_span_arcsec([nx, ny], celestial_wcs)
    print(f"Span is x: {span_x} and y: {span_y}")
    sx, sy = span_x.to(angle_unit).value, span_y.to(angle_unit).value
    print(sx, sy)

    # convert desired arcsec steps to tick locations in pixel units
    def angle_values_to_tick_locations(user_angles, span, n):
        """Tick locations means where to put the ticks in terms of pixel
        indices."""
        angles = (
            np.array(range(0, int(math.ceil(sx))))
            if user_angles is None
            else user_angles
        )
        tick_locations = angles / span * n
        return tick_locations

    xticks = angle_values_to_tick_locations(x_angle_values, sx, nx)
    yticks = angle_values_to_tick_locations(y_angle_values, sy, ny)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    # print("xticks (pixel)", xticks)
    # print("yticks (pixel)", yticks)

    # ticker label format function that converts tick locations to
    # labels in arcsec units
    xformatter = ticker.FuncFormatter(
        lambda x, _: str(math.floor(x / nx * sx - x_offset + 0.5))
    )
    yformatter = ticker.FuncFormatter(
        lambda y, _: str(math.floor(y / ny * sy - y_offset + 0.5))
    )
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

    Mappable is typically the image object returned by the imshow call.

    https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph

    """
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


def scatter_in_pixel_coords(
    ax,
    ras,
    decs,
    image_wcs,
    scatter_kwargs={},
    labels=None,
):
    """
    labels: text labels to point at each point

    """
    # convert RA and DEC to the right XY coordinates for the image
    coords = image_wcs.celestial.world_to_pixel_values(ras, decs)
    x, y = coords[0], coords[1]

    scatter = ax.scatter(x, y, **scatter_kwargs)

    if labels is not None:
        for i, l in enumerate(labels):
            ax.annotate(l, (x[i], y[i]))

    # choose a good zoom level
    # xmin = np.amin(x)
    # xmax = np.amax(x)
    # xw = x_fudge * (xmax - xmin)
    # xc = (xmax + xmin) / 2
    # ymax = np.amax(y)
    # ymin = np.amin(y)
    # yw = y_fudge * (ymax - ymin)
    # yc = (ymax + ymin) / 2
    # ax.set_xlim(xc - xw / 2, xc + xw / 2)
    # ax.set_ylim(yc - yw / 2, yc + yw / 2)
    return {"scatter": scatter, "x": x, "y": y}
