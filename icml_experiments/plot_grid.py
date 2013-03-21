import argparse
import cPickle as pickle

import rftk.forest_data as forest_data
import rftk.utils.forest as forest_utils
import rftk.utils.predict as predict_utils

import plot_utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot accuracy')
    parser.add_argument('-c', '--config_file', help='experiment config file', required=True)
    parser.add_argument('-f', '--in_forest_folder', default="In file", help='', required=True)
    parser.add_argument('-of', '--out_forest_folder', default="", help='out file', required=True)
    parser.add_argument('-ot', '--out_tree_folder', default="", help='out file', required=True)
    args = parser.parse_args()

    config = __import__(args.config_file)
    data_config = config.get_data_config()

    X_train, Y_train, X_test, Y_test = data_config.load_data(data_config.data_sizes[-1], 1)

    previous_data_size = 0
    for data_size in data_config.data_sizes:
        print "plotting sample count %d" % data_size
        X_online_train_sample = X_train[previous_data_size:data_size, :]
        Y_online_train_sample = Y_train[previous_data_size:data_size]

        forest_pickle_filename = "%s/forest-%d.pkl" % (args.in_forest_folder, data_size)
        forest = forest_utils.pickle_load_native_forest(forest_pickle_filename)
        predict_forest = predict_utils.MatrixForestPredictor(forest)

        plot_utils.grid_plot(predict_forest, X_online_train_sample, Y_online_train_sample, X_test,
            "%s/grid-%d.png" % (args.out_forest_folder, data_size))
        plot_utils.grid_plot(predict_forest, X_online_train_sample, Y_online_train_sample, X_test,
            "%s/grid-%d-scatter.png" % (args.out_forest_folder, data_size), plot_scatter=True)

        for tree_id in range(forest.GetNumberOfTrees()):
            single_tree_forest_data = forest_data.Forest([forest.GetTree(tree_id)])
            single_tree_forest_predictor = predict_utils.MatrixForestPredictor(single_tree_forest_data)
            plot_utils.grid_plot(single_tree_forest_predictor, X_online_train_sample, Y_online_train_sample, X_test,
                "%s/grid-%d-tree-%d.png" % (args.out_tree_folder, data_size, tree_id))

        previous_data_size = data_size
