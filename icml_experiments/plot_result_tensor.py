import numpy as np
import argparse
import cPickle as pickle
import matplotlib.pyplot as plt
import itertools


import plot_utils_tensor as plot_utils


def key_value_list(string):
    try:
        args = {}
        for arg in string.split(','):
            name, value = arg.split('=')
            args[name] = value
        return args
    except:
        raise argparse.ArgumentTypeError("I don't understand '{}'".format(string))

def csv_list(type):
    def typed_csv_list(string):
        try:
            return map(type, string.split(','))
        except:
            raise argparse.ArgumentTypeError("I don't understand '{}'".format(string))
    return typed_csv_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Slice, reduce and visualize a single MeasurementGrid.')
    parser.add_argument('-f', '--in_file',
            help='file with experiment result',
            required=True)
    parser.add_argument('-l', '--log_scale',
            help='use log scale',
            action='store_const',
            const=True,
            default=False)
    parser.add_argument('-q', '--quiet',
            help="don't show the created plot",
            default=False,
            required=False,
            action='store_const',
            const=True)
    parser.add_argument('-p', '--plot_file',
            help='out plot',
            default=None,
            required=False)
    parser.add_argument('-r', '--reduce',
            help='axis to use for error bars',
            type=csv_list(str),
            default=[],
            required=False)
    parser.add_argument('-x', '--x_axis',
            help='independent variable',
            required=True)
    parser.add_argument('-y', '--y_axis',
            help='dependent variable',
            required=True)
    parser.add_argument('-s', '--slice',
            help='where to slice the experiment tensor',
            type=key_value_list,
            default={},
            required=False)
    parser.add_argument('-e', '--each',
            help="draw a line for each level of the specified variables",
            required=False)
    parser.add_argument('--line_styles',
            help="line styles to plot with",
            required=False,
            default=['-', '--', '-.', ':'])
    parser.add_argument('--fill_colors',
            help="colors to use for filling in uncertainty bands",
            required=False,
            default=['r', 'g', 'b', 'c', 'm', 'y', 'k'])
    parser.add_argument('--title',
            help="title for the plot",
            required=False)
    parser.add_argument('--x_lim',
            help="x limits for the plot",
            default=None,
            type=csv_list(float),
            required=False)
    parser.add_argument('--y_lim',
            help="y limits for the plot",
            default=None,
            type=csv_list(float),
            required=False)
    parser.add_argument('--legend_loc',
            help="location for the legend",
            default="best",
            required=False)
    args = parser.parse_args()

    if not args.title:
        args.title = "{} vs {}".format(args.y_axis, args.x_axis)

    experiment_result = pickle.load(open(args.in_file))
    measurements = experiment_result['measurements']

    fig = plt.figure()
    ax = fig.gca()

    if args.each:
        plot_utils.draw_many_lines_with_uncertainty(
            ax,
            measurements,
            each=args.each,
            x=args.x_axis,
            y=args.y_axis,
            slice=args.slice,
            reduce=args.reduce,
            fill_colors=args.fill_colors,
            line_styles=args.line_styles,
            )
    else:
        plot_utils.draw_line_with_uncertainty(
            ax,
            measurements,
            x=args.x_axis,
            y=args.y_axis,
            slice=args.slice,
            reduce=args.reduce,
            line_style=args.line_styles[0],
            fill_color=args.fill_colors[0],
            )

    ax.legend(loc=args.legend_loc)
    ax.set_title(args.title)

    if not args.x_lim:
        ax.set_xlim([
            np.min(measurements.domain.extent[args.x_axis]),
            np.max(measurements.domain.extent[args.x_axis])
            ])
    else:
        ax.set_xlim(args.x_lim)

    if args.log_scale:
        ax.set_xscale('log')

    if args.y_lim:
        ax.set_ylim(args.y_lim)
        
    if not args.quiet:
        plt.show()

    if args.plot_file:
        plt.savefig(args.plot_file)

