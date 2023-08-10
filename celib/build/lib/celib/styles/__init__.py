# -*- coding: utf-8 -*-
"""
styles
======
Styling for CE charts.

Adds CE colours to the list of matplotlib colors.

**Requires matplotlib version >= 2.0.0 as a dependency**

Usage:

    # Apply CE default chart styles
    import celib
    celib.styles.apply_defaults()

    # Apply a custom colour sequence (here, red -> green -> blue)
    import matplotlib.pyplot as plt
    rgb_cycle = celib.styles.create_cycler(color=['red', 'green', 'blue'])
    plt.rc('axes', prop_cycle=rgb_cycle)

    # Use `celib.styles.interpolate_colours()` to create a six-shade gradient
    # from white to blue, before applying as above
    shades = celib.styles.interpolate_colours(palette=((1, 1, 1), (0, 0, 1)), n=6)
    shades_cycle = celib.styles.create_cycler(color=shades)
    plt.rc('axes', prop_cycle=shades_cycle)

Notes
-----
On calling function, the names of the CE colours are stored to variable
`ce_colours`.

"""

import sys
import warnings

import matplotlib as mpl
from matplotlib import cycler
import matplotlib.pyplot as plt
import numpy as np

import celib.styles.styles

ce_colours = {
        'ce-pink-main': '#c5446e',
        'ce-green-main': '#aab71d',
        'ce-teal-main': '#49c9c5',
        'ce-cyan-1': '#009fe3',
        'ce-accent-2': '#909090',
        'ce-accent-3': '#0b1f2c',
        'ce-pink-subtle': '#f7e3ea',
        'ce-teal-subtle': '#d9f4f3',
        'ce-green-subtle': '#f2f4dc',
        'ce-light-cyan': '#66c5ee',
        'ce-light-cyan-2': '#99d9f4',
        }

mpl.colors.get_named_colors_mapping().update(ce_colours)


def use(style):
    """Apply a CE-specific style sheet.

    Parameter
    ---------
    style : str or list-like (of str)
        str - name of the CE style to apply
        list-like - name(s) of the CE style(s) to apply

    Notes
    -----
    Valid styles are stored in '.mplstyle' files in celib/styles/styles/. See,
    for example, 'ce-default'.

    """
    if type(style) is str:
        style = [style]

    plt.style.use([celib.styles.styles._style_files[s] for s in style])


def apply_defaults():
    """Apply default CE chart styles."""
    use(['ce-default', 'ce-colours'])


def create_cycler(**kwargs):
    """Create a `matplotlib` `Cycler` object from the keyword arguments.

    Typical usage
    -------------
    >>> import matplotlib.pyplot as plt
    >>> import celib

    # Alternate between light and dark blue from the CE branding guidelines
    # NB For RGB colours, values should lie in [0, 1] - need to rescale from
    #    the more common integer representation
    >>> cycle = celib.styles.create_cycler(color=[(0, 167/255, 231/255),
                                                  (0, 99/255, 152/255)],
                                           linestyle=['-', '--'])
    >>> plt.rc('axes', prop_cycle=cycle)

    Other examples
    --------------
    >>> import celib

    # Generate a two-colour cycler
    >>> cycle = celib.styles.create_cycler(color=['red', 'blue'])
    >>> list(cycle)
    [{'color': 'red'}, {'color': 'blue'}]

    # Generate a three-colour cycler that switches between dashed and dotted
    # lines after each colour cycle
    >>> celib.styles.create_cycler(color=['red', 'white', 'blue'],
                                   linestyle=['-', '--'])
    >>> list(cycle)
    [{'color': 'red', 'linestyle': '-'},
     {'color': 'white', 'linestyle': '-'},
     {'color': 'blue', 'linestyle': '-'},
     {'color': 'red', 'linestyle': '--'},  # <- Restart colour cycle with a new line style
     {'color': 'white', 'linestyle': '--'},
     {'color': 'blue', 'linestyle': '--'}]

    Notes
    -----
    * To work properly, this function requires Python 3.6 or higher. Earlier
      versions do not preserve the order of keyword arguments, which will lead
      to an arbitrary ordering of the cycles. There will be a warning if the
      Python version is not recent enough. (Not relevant in the one-argument
      case, of course.)
    * This function builds the `Cycler` such that the first argument makes up
      the inner cycle (the cycle that changes most rapidly), working out to the
      last argument, which defines the outermost cycle. That is, for a cycle
      defined as:

      >>> cycle = create_cycler(color=['red', 'blue'], linestyle=['-', '--])
      >>> list(cycle)

      # Note how the colours cycle through before the linestyle changes
      [{'color': 'red', 'linestyle': '-'},
       {'color': 'blue', 'linestyle': '-'},
       {'color': 'red', 'linestyle': '--'},
       {'color': 'blue', 'linestyle': '--'}]

    """
    if sys.version_info.major < 3 or sys.version_info.minor < 6:
        warnings.warn(
            'Python 3.6 or higher required: '
            'Cycle order not guaranteed to be preserved correctly.')

    cycle = None

    for attribute, values in reversed(list(kwargs.items())):
        if cycle is None:
            cycle = cycler(attribute, values)
        else:
            cycle *= cycler(attribute, values)

    return cycle


def interpolate_colours(*, palette=None, n=3):
    """Linearly interpolate `n` shades between the two colours in `palette`.

    Parameters
    ----------
    palette : 2-element list/tuple, defaults to [RGB(0, 167, 231), RGB(0, 99, 152)]
              (as in the CE brand guidelines)
        The start and end colours to interpolate between. Must be a set of RGB
        values.
    n : int
        The number of colours to generate

    Returns
    -------
    shades : list
        `n`-length colour palette

    Notes
    -----
    RGB values in Python are generally assumed to lie in the interval
    [0, 1]. If intending to use 'typical' RGB values (in the range [0, 255]) in
    Python, remember to divide by 255.

    """
    if palette is None:
        palette = [(0, 167, 231), (0, 99, 152)]

    shades = []
    for alpha in np.linspace(start=0, stop=1, num=n):
        shades.append(tuple([a + ((b - a) * alpha) for a, b, in zip(*palette)]))

    return shades