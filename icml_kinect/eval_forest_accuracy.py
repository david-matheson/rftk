import numpy as np
import matplotlib.pyplot as pl
import cPickle as pickle
from datetime import datetime
import argparse
import os

import rftk.buffers as buffers
import rftk.forest_data as forest_data
import rftk.feature_extractors as feature_extractors
import rftk.best_split as best_splits
import rftk.predict as predict
import rftk.train as train

import utils as kinect_utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deterime forest accuracy on test set')
    parser.add_argument('-i', '--pose_files_input_path', type=str, required=True)
    parser.add_argument('-p', '--poses_to_use_file', type=str, required=True)
    parser.add_argument('-n', '--number_of_images', type=int, required=True)
    parser.add_argument('-f', '--forest_input_path', type=str, required=True)
    parser.add_argument('-l', '--list_of_forest_ids', type=str, required=True)
    parser.add_argument('-o', '--out_path', type=str, required=True)
    args = parser.parse_args()

    forest_ids = eval(args.list_of_forest_ids)
    print forest_ids

    poses_to_include_file = open(args.poses_to_use_file, 'r')
    pose_filenames = poses_to_include_file.read().split('\n')
    poses_to_include_file.close()

    depths_buffer, labels_buffer = kinect_utils.load_data(args.pose_files_input_path,
                                            pose_filenames[0:args.number_of_images])

    depths = buffers.as_numpy_array(depths_buffer)
    labels = buffers.as_numpy_array(labels_buffer)

    # Load forest
    for (pass_id, forest_id) in forest_ids:
        forest_pickle_filename = "%s/forest-%d-%d.pkl" % (args.forest_input_path, pass_id, forest_id)
        out_pickle_filename = "%s/accuracy-%d-%d.pkl" % (args.forest_input_path, pass_id, forest_id)
        forest = pickle.load(open(forest_pickle_filename, 'rb'))

        forest_stats = forest.GetForestStats()
        forest_stats.Print()

        # for tree_id in range(forest.GetNumberOfTrees()):
        #     print "path %d" % tree_id
        #     forest.GetTree(tree_id).mPath.Print()

        # for tree_id in range(forest.GetNumberOfTrees()):
        #     print "f_params %d" % tree_id
        #     forest.GetTree(tree_id).mFloatFeatureParams.Print()

        # for tree_id in range(forest.GetNumberOfTrees()):
        #     print "i_params %d" % tree_id
        #     forest.GetTree(tree_id).mIntFeatureParams.Print()

        #kinect_utils.plot_classification_imgs(args.out_path, depths, labels, forest)
        accuracy = kinect_utils.classification_accuracy(depths, labels, forest)
        print accuracy
        pickle.dump(accuracy, file(out_pickle_filename, 'wb'))
