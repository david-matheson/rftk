import numpy as np
import argparse
import cPickle as pickle
import glob

import plot_utils
import dist_utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot depth data')
    parser.add_argument('-c', '--config_file', help='experiment config file', required=True)
    parser.add_argument('-d', '--in_directory', help='directory of input measurements', required=True)
    parser.add_argument('-l', '--log_scale', type=int, help='use log scale', required=True)
    parser.add_argument('-s', '--plot_standard_deviation', type=int, default=1, help='whether to plot error bars')
    parser.add_argument('-p', '--plot_file', help='out plot', required=True)
    args = parser.parse_args()

    config = __import__(args.config_file)
    data_config = config.get_data_config()
    measurements = []

    for f in glob.glob("%s/%s*pkl" % (args.in_directory, config.get_experiment_name())):
        measurements.extend( pickle.load(open(f, "rb")))

    plot_utils.plot_depths(data_config.data_sizes, data_config.number_of_passes_through_data,
        measurements, plot_standard_deviation=(args.plot_standard_deviation != 0),
        log_scale=(args.log_scale != 0), plot_filename=args.plot_file)