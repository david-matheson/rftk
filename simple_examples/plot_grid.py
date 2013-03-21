import numpy as np
import argparse
import os
import cPickle as pickle
from datetime import datetime

import rftk.forest_data as forest_data
import rftk.predict as predict

import plot_utils
import dist_utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot accuracy')
    parser.add_argument('-i', '--in_folder', default="In file", help='', required=True)
    parser.add_argument('-f', '--out_forest_folder', default="", help='out file', required=True)
    parser.add_argument('-t', '--out_tree_folder', default="", help='out file', required=True)
    args = parser.parse_args()

    accuracy_file = open("%s/accuracies.pkl" % (args.in_folder), "rb")
    (epoch_id, total_sample_list, forest_accuracy, tree_accuracy, bayes_accuracy) = pickle.load(accuracy_file)

    data_file = open("%s/data.pkl" % (args.in_folder), "rb")
    (X_train, Y_train, X_test, Y_test) = pickle.load(data_file)

    previous_sample_count = 0
    for sample_count in total_sample_list:
        print "plotting sample count %d" % sample_count
        X_online_train_sample = X_train[previous_sample_count:sample_count, :]
        Y_online_train_sample = Y_train[previous_sample_count:sample_count]

        forest_pickle_filename = "%s/forest-%d.pkl" % (args.in_folder, sample_count)
        forest = forest_data.pickle_load_native_forest(forest_pickle_filename)
        predict_forest = predict.MatrixForestPredictor(forest)

        plot_utils.grid_plot(predict_forest, X_online_train_sample, Y_online_train_sample, X_test,
            "%s/grid-%d.png" % (args.out_forest_folder, sample_count))
        plot_utils.grid_plot(predict_forest, X_online_train_sample, Y_online_train_sample, X_test,
            "%s/grid-%d-scatter.png" % (args.out_forest_folder, sample_count), plot_scatter=True)

        for tree_id in range(forest.GetNumberOfTrees()):
            single_tree_forest_data = forest_data.Forest([forest.GetTree(tree_id)])
            single_tree_forest_predictor = predict.MatrixForestPredictor(single_tree_forest_data)
            plot_utils.grid_plot(single_tree_forest_predictor, X_online_train_sample, Y_online_train_sample, X_test,
                "%s/grid-%d-tree-%d.png" % (args.out_forest_folder, sample_count, tree_id))

        previous_sample_count = sample_count
