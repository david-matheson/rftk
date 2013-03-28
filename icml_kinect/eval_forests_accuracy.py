import numpy as np
import matplotlib.pyplot as pl
import cPickle as pickle
import gzip
from datetime import datetime
import argparse
import os
import sys

import rftk.buffers as buffers
import rftk.forest_data as forest_data
import rftk.feature_extractors as feature_extractors
import rftk.best_split as best_splits
import rftk.predict as predict
import rftk.train as train

import utils as kinect_utils


def eval_accuracies(depths_buffer, labels_buffer, list_of_forest_and_accuracy_files, force_compute=False):
    accuracies = np.zeros(len(list_of_forest_and_accuracy_files))
    for i, (forest_file, accuracy_file) in enumerate(list_of_forest_and_accuracy_files):
        if not os.path.exists(accuracy_file) or force_compute:
            depths = buffers.as_numpy_array(depths_buffer)
            labels = buffers.as_numpy_array(labels_buffer)
            try:
                forest = pickle.load(open(forest_file, 'rb'))
            except:
                try:
                    forest = pickle.load(gzip.open(forest_file, 'rb'))
                    stats = forest.GetForestStats()
                    stats.Print()
                except:
                    print "Unexpected error:", sys.exc_info()[0]

            accuracy = kinect_utils.classification_accuracy(depths, labels, forest, 8)
            py_forest = forest_data.as_pyforest(forest)
            pickle.dump(accuracy, file(accuracy_file, 'wb'))
        accuracies[i] = pickle.load(file(accuracy_file, 'rb'))
    print accuracies
    return accuracies


def gen_paths(folder, runs):
    paths = []
    for (number_passes, number_data) in runs:
        forest_file = '%s/forest-%d-%d.pkl' % (folder, number_passes, number_data)
        accuracy_file = '%s/accuracy-%d-%d.pkl' % (folder, number_passes, number_data)
        paths.append((forest_file, accuracy_file))
    return paths

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot forests accuracy on test set')
    parser.add_argument('-i', '--pose_files_input_path', type=str, required=True)
    parser.add_argument('-p', '--poses_to_use_file', type=str, required=True)
    parser.add_argument('-n', '--number_of_images', type=int, required=True)
    parser.add_argument('-o', '--out_file', type=str, required=True)
    args = parser.parse_args()

    poses_to_include_file = open(args.poses_to_use_file, 'r')
    pose_filenames = poses_to_include_file.read().split('\n')
    poses_to_include_file.close()

    depths_buffer, labels_buffer = kinect_utils.load_data(args.pose_files_input_path,
                                                        pose_filenames[0:args.number_of_images])


    online_samples = [(0,100), (0, 200), (0, 500), (0, 1000), (0, 2000), (0, 5000),
                        (0, 10000), (0, 25000), (0, 50000), (0, 100000), (0, 250000), (0, 500000),
                        (0,973909),(1,973909),(2,973909)]

    online_iid_rftk3_0 = eval_accuracies(depths_buffer, labels_buffer,
                                    gen_paths('/media/data/projects/rftk-3/icml_kinect/experiment_data_online_iid/online-iid-tree-25-n-1000-m-10-splitrate-1.00-splitroot-10.00-evalperiod-5-maxdepth-500-2013-03-28-14-33-24.598709',
                                            online_samples),
                                    force_compute=False)

    online_iid_rftk_0 = eval_accuracies(depths_buffer, labels_buffer,
                                    gen_paths('/media/data/projects/rftk/icml_kinect/experiment_data_online_iid/online-iid-tree-25-n-1000-m-10-splitrate-1.00-splitroot-10.00-evalperiod-5-maxdepth-500-2013-03-28-14-37-50.788713',
                                            online_samples),
                                    force_compute=False)


    results = {'online_iid_rftk3_0':online_iid_rftk3_0, 'online_iid_rftk_0':online_iid_rftk_0 }
    print results
    pickle.dump(results, file(args.out_file, 'wb'))
