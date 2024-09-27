"""Some very common things to do with images, including plotting and comparing."""

from myastro import plot
from matplotlib import pyplot as plt

def plot_residual(a1, a2):
    """Show two arrays and their difference.

    Layout is reasonable for square images.

    Note: this is put in `image`, rather than `plot`, because it creates
    it's own layout. The stuff in `plot`, needs to operate on a single
    axes ideally.

    """
    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
    kwargs = dict(interpolation='nearest')
    plot.nice_imshow(ax[0], a1, **kwargs)
    plot.nice_imshow(ax[1], a2, **kwargs)
    plot.nice_imshow(ax[2], a2 - a1, **kwargs)
    ax[2].set_title("image 2 - image 1")
    fig.set_size_inches(18, 7)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0)
    return fig, ax
