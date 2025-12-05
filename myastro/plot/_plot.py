"""General plotting utilities"""

import numpy as np
from matplotlib import patheffects
from matplotlib.colors import FuncNorm
from matplotlib import ticker

# some useful kwarg collections
text_white_black_outline_kwargs = {
    "path_effects": [patheffects.withStroke(linewidth=2, foreground="k")],
}


def make_asinh_FuncNorm(scale, offset, cutoff):
    def forward(x):
        np.arcsinh(scale * (x - offset))

    def reverse(y):
        np.sinh(y) / scale + offset

    return FuncNorm(functions=(forward, reverse), vmin=0, vmax=forward(cutoff))


def nice_ticks(ax):
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(
        which="both", axis="both", top=True, bottom=True, left=True, right=True
    )


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
