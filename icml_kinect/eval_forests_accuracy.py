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



def eval_accuracies(depths_buffer, labels_buffer, list_of_forest_and_accuracy_files):
    accuracies = np.zeros(len(list_of_forest_and_accuracy_files))
    for i, (forest_file, accuracy_file) in enumerate(list_of_forest_and_accuracy_files):
        if not os.path.exists(accuracy_file):
            depths = buffers.as_numpy_array(depths_buffer)
            labels = buffers.as_numpy_array(labels_buffer)
            forest = pickle.load(open(forest_file, 'rb'))
            accuracy = kinect_utils.classification_accuracy(depths, labels, forest, 8)
            pickle.dump(accuracy, file(accuracy_file, 'wb'))
        accuracies[i] = pickle.load(file(accuracy_file, 'rb'))
    print accuracies
    return accuracies



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

    depths_buffer, labels_buffer = load_data(args.pose_files_input_path,
                                            pose_filenames[0:args.number_of_images])

    # offline_accuracy = eval_accuracies(depths_buffer, labels_buffer,
    #                     [(  '/home/davidm/projects/rftk/icml_kinect/experiment_data_v4/offline-tree-25-n-25-2013-02-12-22-17-28.589374/forest-1-25.pkl',
    #                         '/home/davidm/projects/rftk/icml_kinect/experiment_data_v4/offline-tree-25-n-25-2013-02-12-22-17-28.589374/accuracy-1-25.pkl'),
    #                     (  '/home/davidm/projects/rftk/icml_kinect/experiment_data_v4/offline-tree-25-n-100-2013-02-12-22-47-33.357779/forest-1-100.pkl',
    #                         '/home/davidm/projects/rftk/icml_kinect/experiment_data_v4/offline-tree-25-n-100-2013-02-12-22-47-33.357779/accuracy-1-100.pkl'),
    #                     (  '/home/davidm/projects/rftk/icml_kinect/experiment_data_v4/offline-tree-25-n-200-2013-02-12-23-42-24.780385/forest-1-200.pkl',
    #                         '/home/davidm/projects/rftk/icml_kinect/experiment_data_v4/offline-tree-25-n-200-2013-02-12-23-42-24.780385/accuracy-1-200.pkl'),
    #                      (  '/home/davidm/projects/rftk/icml_kinect/experiment_data_v4/offline-tree-25-n-500-2013-02-13-01-48-35.758347/forest-1-500.pkl',
    #                         '/home/davidm/projects/rftk/icml_kinect/experiment_data_v4/offline-tree-25-n-500-2013-02-13-01-48-35.758347/accuracy-1-500.pkl'),
    #                      (  '/home/davidm/projects/rftk/icml_kinect/experiment_data_v4/offline-tree-25-n-1000-2013-02-12-22-18-43.251212/forest-1-1000.pkl',
    #                         '/home/davidm/projects/rftk/icml_kinect/experiment_data_v4/offline-tree-25-n-1000-2013-02-12-22-18-43.251212/accuracy-1-1000.pkl')
                        ])


    results = {}
    print results
    pickle.dump(results, file(args.out_file, 'wb'))
