import numpy as np
import matplotlib.pyplot as plt
import itertools
import sys

def draw_line_with_uncertainty(
    axis,
    measurement_grid,
    x,
    y,
    slice=None,
    reduce=None,
    line_style=None,
    fill_color=True,
    label=True,
    ):
    # set label=None to plot an unlabelled line
    if label is True:
        label = y

    if not line_style:
        line_style='-'

    if fill_color is True:
        fill_color='g'

    if not reduce:
        reduce = []

    if slice:
        for var, pos in slice.iteritems():
            measurement_grid = measurement_grid.slice_at(var, pos)

    y_raw = getattr(measurement_grid, y)

    # We need to reduce everything to one dimension in order to
    # plot.  Warn if this means we end up reducing over dimensions
    # whose reduction wasn't requested.
    independent_var = set([x])
    forced_reduce_vars = set(measurement_grid.domain.independent_variables) - independent_var
    if set(reduce) != forced_reduce_vars:
        sys.stderr.write("WARNING: Asked to reduce {} but forced to reduce {}.\n".format(
            reduce, list(forced_reduce_vars)))

        reduce = list(forced_reduce_vars)
        

    if reduce:
        # To reduce over all but one dimension, permute the one we
        # want to keep to the 0th position and then flatten everything
        # else out along the 1st dimension.

        reduce_indexes = sorted(measurement_grid.domain.rank_of(r) for r in reduce)
        x_index = measurement_grid.domain.rank_of(x)

        y_raw = y_raw.transpose(tuple([x_index] + reduce_indexes))
        y_raw = y_raw.reshape((y_raw.shape[0], -1))

        y_mean = y_raw.mean(axis=-1)
        y_std = y_raw.std(axis=-1)
    else:
        y_mean = y_raw
        y_std = np.zeros_like(y_mean)

    x_extent = measurement_grid.domain.extent[x]

    assert y_mean.ndim == 1, "Too many dimensions: slice along one of {}".format(
        [var for var in measurement_grid.domain.independent_variables if var not in [x, reduce]])

    axis.plot(x_extent, y_mean, line_style, label=label)
    if fill_color is not None:
        axis.fill_between(x_extent, y_mean-y_std, y_mean+y_std, alpha=0.2, color=fill_color)


def draw_many_lines_with_uncertainty(
    axis,
    measurement_grid,
    each,
    x,
    y,
    slice=None,
    reduce=None,
    fill_colors=None,
    line_styles=None,
    label_format="{}={}",
    ):

    if not fill_colors:
        fill_colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

    if not line_styles:
        line_styles = ['-', '--', '-.', ':']
    
    line_styles = itertools.cycle(line_styles)
    fill_colors = itertools.cycle(fill_colors)

    each_levels = measurement_grid.domain.extent[each]
    for line_style, fill_color, each_position in zip(line_styles, fill_colors, each_levels):
        each_slice = dict(slice)
        each_slice[each] = each_position
        draw_line_with_uncertainty(
            axis,
            measurement_grid,
            x=x,
            y=y,
            slice=each_slice,
            reduce=reduce,
            line_style=line_style,
            fill_color=fill_color,
            label=label_format.format(each, each_position) if label_format else None)

