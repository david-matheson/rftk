import numpy as np
import argparse
import cPickle as pickle
import glob

import plot_utils
import dist_utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sklearn accuracy on data')
    parser.add_argument('-c', '--config_file', help='experiment config file', required=True)
    parser.add_argument('-p', '--plot_file', help='out plot', required=True)
    args = parser.parse_args()

    config = __import__(args.config_file)
    data_config = config.get_data_config()
    measurements = []

    for f in glob.glob("experiment_data/%s*pkl" % (config.get_experiment_name())):
        measurements.extend( pickle.load(open(f, "rb")))

    plot_utils.plot_forest_and_tree_accuracy(data_config.bayes_accuracy,
        data_config.data_sizes, data_config.number_of_passes_through_data,
        measurements,
        plot_trees=True, plot_sklearn=False, log_scale=True, plot_filename=args.plot_file)