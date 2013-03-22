import numpy as np
import argparse
import os
import cPickle as pickle
from datetime import datetime

import rftk.forest_data as forest_data

import plot_utils
import dist_utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot accuracy')
    parser.add_argument('-o', '--online_rf_accuracy_file', help='Online accuracy pickle file', required=True)
    parser.add_argument('-r', '--rf_accuracy_file', default="", help='Online accuracy pickle file')
    parser.add_argument('-p', '--plot_file', help='out plot', required=True)
    args = parser.parse_args()

    (epoch_id, total_sample_list, online_forest_accuracy, online_tree_accuracy, bayes_accuracy) = pickle.load(open(args.online_rf_accuracy_file, "rb"))
    if args.rf_accuracy_file is not "":
        offline_forest_accuracy = pickle.load(open(args.rf_accuracy_file, "rb"))
    else:
        offline_forest_accuracy = None

    plot_utils.plot_forest_and_tree_accuracy(epoch_id, np.array(total_sample_list), online_forest_accuracy, online_tree_accuracy,
        bayes_accuracy, offline_forest_accuracy, args.plot_file)