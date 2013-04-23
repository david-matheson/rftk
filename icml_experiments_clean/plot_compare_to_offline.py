import numpy as np
import argparse
import glob
import cPickle as pickle
import matplotlib
import matplotlib.pyplot as plt
import plot_utils


def load_measurements(file_name):
    with open(file_name) as pkl_file:
        data = pickle.load(pkl_file)
    return data['measurements']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare tree performance to the forest.')
    parser.add_argument('--offline',
        help="offline file",
        required=True)
    parser.add_argument('--online',
        help="online file",
        required=True)
    parser.add_argument('--saffari',
        help="saffari file",
        required=True)
    parser.add_argument('-q',
        help="suppress interactive display",
        action='store_true',
        required=False)
    parser.add_argument('-o', '--out',
        help="output file name",
        required=True)
    args = parser.parse_args()

    offline_measurements = load_measurements(args.offline)
    online_measurements = load_measurements(args.online)
    saffari_measurements = load_measurements(args.saffari)

    params = {
            'axes.labelsize' : 20,
            'font.size' : 20,
            'text.fontsize' : 20,
            'legend.fontsize': 20,
            'xtick.labelsize' : 15,
            'ytick.labelsize' : 15,
            }
    matplotlib.rcParams.update(params)


    fig = plt.figure()
    ax = fig.gca()

    off_line = plot_utils.draw_line_with_uncertainty(
        ax,
        offline_measurements,
        x='data_size',
        y='accuracy',
        reduce=['job_id'],
        fill_color='b',
        line_style='b-^',
        label="Offline"
        )

    saffari_line = plot_utils.draw_line_with_uncertainty(
        ax,
        saffari_measurements,
        x='data_size',
        y='accuracy',
        reduce=['job_id'],
        fill_color='r',
        line_style='r--h',
        label="Saffari et al. (2009)"
        )

    on_line = plot_utils.draw_line_with_uncertainty(
        ax,
        online_measurements,
        x='data_size',
        y='accuracy',
        reduce=['job_id'],
        fill_color='g',
        line_style='g-.D',
        label="Online"
        )

    x_extent = online_measurements.domain.extent['data_size']
    ax.set_xlim([x_extent.min(), x_extent.max()])

    ax.set_xscale('log')
    ax.set_xlabel('Data Size')
    ax.set_ylabel('Accuracy')
    
    handles, labels = ax.get_legend_handles_labels()
    handles = [handles[0], handles[2], handles[1]]
    labels = [labels[0], labels[2], labels[1]]
    ax.legend(handles, labels, loc="lower right")
    ax.set_title("USPS")

    plt.savefig(args.out)
    if not args.q:
        plt.show()
