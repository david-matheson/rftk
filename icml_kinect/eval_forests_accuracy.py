import numpy as np
import matplotlib.pyplot as pl
import cPickle as pickle
from datetime import datetime
import argparse
import os

import rftk.native.assert_util
import rftk.native.bootstrap
import rftk.native.buffers as buffers
import rftk.native.forest_data as forest_data
import rftk.native.features
import rftk.native.feature_extractors as feature_extractors
import rftk.native.best_split as best_splits
import rftk.native.predict as predict
import rftk.native.train as train


import rftk.utils.forest as forest_utils

import utils as kinect_utils



def load_data(pose_path, list_of_poses):
    concat = False
    for i, pose_filename in enumerate(list_of_poses):
        print "Loading %d - %s" % (i, pose_filename)

        # Load single pose depth and class labels
        depths = pickle.load(open("%s%s_depth.pkl" % (pose_path, pose_filename), 'rb'))
        labels = pickle.load(open("%s%s_classlabels.pkl" % (pose_path, pose_filename), 'rb'))

        depths = depths

        depths_buffer = buffers.as_tensor_buffer(depths)
        labels_buffer = buffers.as_tensor_buffer(labels)

        if concat:
            complete_depths_buffer.Append(depths_buffer)
            complete_labels_buffer.Append(labels_buffer)
        else:
            complete_depths_buffer = depths_buffer
            complete_labels_buffer = labels_buffer
            concat = True

    assert(complete_depths_buffer.GetL() == complete_labels_buffer.GetL())
    assert(complete_depths_buffer.GetM() == complete_labels_buffer.GetM())
    assert(complete_depths_buffer.GetN() == complete_labels_buffer.GetN())

    return complete_depths_buffer, complete_labels_buffer



def eval_accuracies(depths_buffer, labels_buffer, list_of_forest_and_accuracy_files):
    accuracies = np.zeros(len(list_of_forest_and_accuracy_files))
    for i, (forest_file, accuracy_file) in enumerate(list_of_forest_and_accuracy_files):
        if not os.path.exists(accuracy_file):
            depths = buffers.as_numpy_array(depths_buffer)
            labels = buffers.as_numpy_array(labels_buffer)
            forest = forest_utils.pickle_load_native_forest(forest_file)
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

    online_one_pass_accuracy = eval_accuracies(depths_buffer, labels_buffer,
                            [(  '/home/davidm/projects/rftk/icml_kinect/experiment_data_v5/online-tree-25-n-25-m-10-splitrate-1.20-splitroot-1000.00-evalperiod-100-2013-02-13-18-42-57.133029/forest-0-25.pkl',
                                '/home/davidm/projects/rftk/icml_kinect/experiment_data_v5/online-tree-25-n-25-m-10-splitrate-1.20-splitroot-1000.00-evalperiod-100-2013-02-13-18-42-57.133029/accuracy-0-25.pkl'),
                            (  '/home/davidm/projects/rftk/icml_kinect/experiment_data_v5/online-tree-25-n-100-m-10-splitrate-1.20-splitroot-1000.00-evalperiod-100-2013-02-13-18-43-35.062241/forest-0-100.pkl',
                                '/home/davidm/projects/rftk/icml_kinect/experiment_data_v5/online-tree-25-n-100-m-10-splitrate-1.20-splitroot-1000.00-evalperiod-100-2013-02-13-18-43-35.062241/accuracy-0-100.pkl'),
                            (  '/home/davidm/projects/rftk/icml_kinect/experiment_data_v5/online-tree-25-n-200-m-10-splitrate-1.20-splitroot-1000.00-evalperiod-100-2013-02-13-18-43-38.353385/forest-0-200.pkl',
                                '/home/davidm/projects/rftk/icml_kinect/experiment_data_v5/online-tree-25-n-200-m-10-splitrate-1.20-splitroot-1000.00-evalperiod-100-2013-02-13-18-43-38.353385/accuracy-0-200.pkl'),
                            (   '/home/davidm/projects/rftk/icml_kinect/experiment_data_v5/online-tree-25-n-500-m-10-splitrate-1.20-splitroot-1000.00-evalperiod-100-2013-02-13-18-43-40.748752/forest-0-500.pkl',
                                '/home/davidm/projects/rftk/icml_kinect/experiment_data_v5/online-tree-25-n-500-m-10-splitrate-1.20-splitroot-1000.00-evalperiod-100-2013-02-13-18-43-40.748752/accuracy-0-500.pkl'),
                            (   '/home/davidm/projects/rftk/icml_kinect/experiment_data_v5/online-tree-25-n-1000-m-10-splitrate-1.20-splitroot-1000.00-evalperiod-100-2013-02-13-18-43-43.905509/forest-0-1000.pkl',
                                '/home/davidm/projects/rftk/icml_kinect/experiment_data_v5/online-tree-25-n-1000-m-10-splitrate-1.20-splitroot-1000.00-evalperiod-100-2013-02-13-18-43-43.905509/accuracy-0-1000.pkl')
                            ])


    online_multi_pass_accuracy = eval_accuracies(depths_buffer, labels_buffer,
                            [(  '/home/davidm/projects/rftk/icml_kinect/experiment_data_v5/online-tree-25-n-25-m-10-splitrate-1.20-splitroot-1000.00-evalperiod-100-2013-02-13-18-42-57.133029/forest-9-25.pkl',
                                '/home/davidm/projects/rftk/icml_kinect/experiment_data_v5/online-tree-25-n-25-m-10-splitrate-1.20-splitroot-1000.00-evalperiod-100-2013-02-13-18-42-57.133029/accuracy-9-25.pkl'),
                            (  '/home/davidm/projects/rftk/icml_kinect/experiment_data_v5/online-tree-25-n-100-m-10-splitrate-1.20-splitroot-1000.00-evalperiod-100-2013-02-13-18-43-35.062241/forest-9-100.pkl',
                                '/home/davidm/projects/rftk/icml_kinect/experiment_data_v5/online-tree-25-n-100-m-10-splitrate-1.20-splitroot-1000.00-evalperiod-100-2013-02-13-18-43-35.062241/accuracy-9-100.pkl'),
                            (  '/home/davidm/projects/rftk/icml_kinect/experiment_data_v5/online-tree-25-n-200-m-10-splitrate-1.20-splitroot-1000.00-evalperiod-100-2013-02-13-18-43-38.353385/forest-9-200.pkl',
                                '/home/davidm/projects/rftk/icml_kinect/experiment_data_v5/online-tree-25-n-200-m-10-splitrate-1.20-splitroot-1000.00-evalperiod-100-2013-02-13-18-43-38.353385/accuracy-9-200.pkl'),
                            (   '/home/davidm/projects/rftk/icml_kinect/experiment_data_v5/online-tree-25-n-500-m-10-splitrate-1.20-splitroot-1000.00-evalperiod-100-2013-02-13-18-43-40.748752/forest-9-500.pkl',
                                '/home/davidm/projects/rftk/icml_kinect/experiment_data_v5/online-tree-25-n-500-m-10-splitrate-1.20-splitroot-1000.00-evalperiod-100-2013-02-13-18-43-40.748752/accuracy-9-500.pkl'),
                            (   '/home/davidm/projects/rftk/icml_kinect/experiment_data_v5/online-tree-25-n-1000-m-10-splitrate-1.20-splitroot-1000.00-evalperiod-100-2013-02-13-18-43-43.905509/forest-0-1000.pkl',
                                '/home/davidm/projects/rftk/icml_kinect/experiment_data_v5/online-tree-25-n-1000-m-10-splitrate-1.20-splitroot-1000.00-evalperiod-100-2013-02-13-18-43-43.905509/accuracy-0-1000.pkl')
                            ])

    alpha_online_one_pass_accuracy = eval_accuracies(depths_buffer, labels_buffer,
                            [(  '/home/davidm/projects/rftk/icml_kinect/experiment_data_v5/online-alphabeta-tree-25-n-25-m-10-min_samples_split-2000-evalperiod-100-2013-02-13-21-16-21.011147/forest-0-25.pkl',
                                '/home/davidm/projects/rftk/icml_kinect/experiment_data_v5/online-alphabeta-tree-25-n-25-m-10-min_samples_split-2000-evalperiod-100-2013-02-13-21-16-21.011147/accuracy-0-25.pkl'),
                            (  '/home/davidm/projects/rftk/icml_kinect/experiment_data_v5/online-alphabeta-tree-25-n-100-m-10-min_samples_split-2000-evalperiod-100-2013-02-13-21-17-16.618084/forest-0-100.pkl',
                                '/home/davidm/projects/rftk/icml_kinect/experiment_data_v5/online-alphabeta-tree-25-n-100-m-10-min_samples_split-2000-evalperiod-100-2013-02-13-21-17-16.618084/accuracy-0-100.pkl'),
                            (  '/home/davidm/projects/rftk/icml_kinect/experiment_data_v5/online-alphabeta-tree-25-n-200-m-10-min_samples_split-2000-evalperiod-100-2013-02-13-21-36-30.393757/forest-0-200.pkl',
                                '/home/davidm/projects/rftk/icml_kinect/experiment_data_v5/online-alphabeta-tree-25-n-200-m-10-min_samples_split-2000-evalperiod-100-2013-02-13-21-36-30.393757/accuracy-0-200.pkl'),
                            (  '/home/davidm/projects/rftk/icml_kinect/experiment_data_v5/online-alphabeta-tree-25-n-200-m-10-min_samples_split-2000-evalperiod-100-2013-02-13-21-36-30.393757/forest-0-200.pkl',
                                '/home/davidm/projects/rftk/icml_kinect/experiment_data_v5/online-alphabeta-tree-25-n-200-m-10-min_samples_split-2000-evalperiod-100-2013-02-13-21-36-30.393757/accuracy-0-200.pkl'),
                            (  '/home/davidm/projects/rftk/icml_kinect/experiment_data_v5/online-alphabeta-tree-25-n-200-m-10-min_samples_split-2000-evalperiod-100-2013-02-13-21-36-30.393757/forest-0-200.pkl',
                                '/home/davidm/projects/rftk/icml_kinect/experiment_data_v5/online-alphabeta-tree-25-n-200-m-10-min_samples_split-2000-evalperiod-100-2013-02-13-21-36-30.393757/accuracy-0-200.pkl')
                            ])

    alpha_online_multi_pass_accuracy = eval_accuracies(depths_buffer, labels_buffer,
                            [(  '/home/davidm/projects/rftk/icml_kinect/experiment_data_v5/online-alphabeta-tree-25-n-25-m-10-min_samples_split-2000-evalperiod-100-2013-02-13-21-16-21.011147/forest-9-25.pkl',
                                '/home/davidm/projects/rftk/icml_kinect/experiment_data_v5/online-alphabeta-tree-25-n-25-m-10-min_samples_split-2000-evalperiod-100-2013-02-13-21-16-21.011147/accuracy-9-25.pkl'),
                            (  '/home/davidm/projects/rftk/icml_kinect/experiment_data_v5/online-alphabeta-tree-25-n-100-m-10-min_samples_split-2000-evalperiod-100-2013-02-13-21-17-16.618084/forest-9-100.pkl',
                                '/home/davidm/projects/rftk/icml_kinect/experiment_data_v5/online-alphabeta-tree-25-n-100-m-10-min_samples_split-2000-evalperiod-100-2013-02-13-21-17-16.618084/accuracy-9-100.pkl'),
                            (  '/home/davidm/projects/rftk/icml_kinect/experiment_data_v5/online-alphabeta-tree-25-n-200-m-10-min_samples_split-2000-evalperiod-100-2013-02-13-21-36-30.393757/forest-9-200.pkl',
                                '/home/davidm/projects/rftk/icml_kinect/experiment_data_v5/online-alphabeta-tree-25-n-200-m-10-min_samples_split-2000-evalperiod-100-2013-02-13-21-36-30.393757/accuracy-9-200.pkl'),
                            (  '/home/davidm/projects/rftk/icml_kinect/experiment_data_v5/online-alphabeta-tree-25-n-200-m-10-min_samples_split-2000-evalperiod-100-2013-02-13-21-36-30.393757/forest-9-200.pkl',
                                '/home/davidm/projects/rftk/icml_kinect/experiment_data_v5/online-alphabeta-tree-25-n-200-m-10-min_samples_split-2000-evalperiod-100-2013-02-13-21-36-30.393757/accuracy-9-200.pkl'),
                            (  '/home/davidm/projects/rftk/icml_kinect/experiment_data_v5/online-alphabeta-tree-25-n-200-m-10-min_samples_split-2000-evalperiod-100-2013-02-13-21-36-30.393757/forest-9-200.pkl',
                                '/home/davidm/projects/rftk/icml_kinect/experiment_data_v5/online-alphabeta-tree-25-n-200-m-10-min_samples_split-2000-evalperiod-100-2013-02-13-21-36-30.393757/accuracy-9-200.pkl')
                            ])

    offline_accuracy = eval_accuracies(depths_buffer, labels_buffer,
                            [(  '/home/davidm/projects/rftk/icml_kinect/experiment_data_v4/offline-tree-25-n-25-2013-02-12-22-17-28.589374/forest-1-25.pkl',
                                '/home/davidm/projects/rftk/icml_kinect/experiment_data_v4/offline-tree-25-n-25-2013-02-12-22-17-28.589374/accuracy-1-25.pkl'),
                            (  '/home/davidm/projects/rftk/icml_kinect/experiment_data_v4/offline-tree-25-n-100-2013-02-12-22-47-33.357779/forest-1-100.pkl',
                                '/home/davidm/projects/rftk/icml_kinect/experiment_data_v4/offline-tree-25-n-100-2013-02-12-22-47-33.357779/accuracy-1-100.pkl'),
                            (  '/home/davidm/projects/rftk/icml_kinect/experiment_data_v4/offline-tree-25-n-200-2013-02-12-23-42-24.780385/forest-1-200.pkl',
                                '/home/davidm/projects/rftk/icml_kinect/experiment_data_v4/offline-tree-25-n-200-2013-02-12-23-42-24.780385/accuracy-1-200.pkl'),
                             (  '/home/davidm/projects/rftk/icml_kinect/experiment_data_v4/offline-tree-25-n-500-2013-02-13-01-48-35.758347/forest-1-500.pkl',
                                '/home/davidm/projects/rftk/icml_kinect/experiment_data_v4/offline-tree-25-n-500-2013-02-13-01-48-35.758347/accuracy-1-500.pkl'),
                             (  '/home/davidm/projects/rftk/icml_kinect/experiment_data_v4/offline-tree-25-n-1000-2013-02-12-22-18-43.251212/forest-1-1000.pkl',
                                '/home/davidm/projects/rftk/icml_kinect/experiment_data_v4/offline-tree-25-n-1000-2013-02-12-22-18-43.251212/accuracy-1-1000.pkl')
                            ])

    online_one_pass_accuracy_no_depth = eval_accuracies(depths_buffer, labels_buffer,
                            [(  '/home/davidm/projects/rftk/icml_kinect/experiment_data_v5_fix_frontier/online-tree-25-n-1000-m-100-splitrate-1.00-splitroot-200.00-evalperiod-10-2013-03-14-02-05-12.588879/forest-0-25.pkl',
                                '/home/davidm/projects/rftk/icml_kinect/experiment_data_v5_fix_frontier/online-tree-25-n-1000-m-100-splitrate-1.00-splitroot-200.00-evalperiod-10-2013-03-14-02-05-12.588879/accuracy-0-25.pkl'),
                            (  '/home/davidm/projects/rftk/icml_kinect/experiment_data_v5_fix_frontier/online-tree-25-n-1000-m-100-splitrate-1.00-splitroot-200.00-evalperiod-10-2013-03-14-02-05-12.588879/forest-0-100.pkl',
                                '/home/davidm/projects/rftk/icml_kinect/experiment_data_v5_fix_frontier/online-tree-25-n-1000-m-100-splitrate-1.00-splitroot-200.00-evalperiod-10-2013-03-14-02-05-12.588879/accuracy-0-100.pkl'),
                            (  '/home/davidm/projects/rftk/icml_kinect/experiment_data_v5_fix_frontier/online-tree-25-n-1000-m-100-splitrate-1.00-splitroot-200.00-evalperiod-10-2013-03-14-02-05-12.588879/forest-0-200.pkl',
                                '/home/davidm/projects/rftk/icml_kinect/experiment_data_v5_fix_frontier/online-tree-25-n-1000-m-100-splitrate-1.00-splitroot-200.00-evalperiod-10-2013-03-14-02-05-12.588879/accuracy-0-200.pkl'),
                            (   '/home/davidm/projects/rftk/icml_kinect/experiment_data_v5_fix_frontier/online-tree-25-n-1000-m-100-splitrate-1.00-splitroot-200.00-evalperiod-10-2013-03-14-02-05-12.588879/forest-0-500.pkl',
                                '/home/davidm/projects/rftk/icml_kinect/experiment_data_v5_fix_frontier/online-tree-25-n-1000-m-100-splitrate-1.00-splitroot-200.00-evalperiod-10-2013-03-14-02-05-12.588879/accuracy-0-500.pkl'),
                            (   '/home/davidm/projects/rftk/icml_kinect/experiment_data_v5_fix_frontier/online-tree-25-n-1000-m-100-splitrate-1.00-splitroot-200.00-evalperiod-10-2013-03-14-02-05-12.588879/forest-0-1000.pkl',
                                '/home/davidm/projects/rftk/icml_kinect/experiment_data_v5_fix_frontier/online-tree-25-n-1000-m-100-splitrate-1.00-splitroot-200.00-evalperiod-10-2013-03-14-02-05-12.588879/accuracy-0-1000.pkl')
                            ])



    online_35_pass_accuracy_no_depth = eval_accuracies(depths_buffer, labels_buffer,
                            [(  '/home/davidm/projects/rftk/icml_kinect/experiment_data_v5_fix_frontier/online-tree-25-n-1000-m-100-splitrate-1.00-splitroot-200.00-evalperiod-10-2013-03-14-02-05-12.588879/forest-35-25.pkl',
                                '/home/davidm/projects/rftk/icml_kinect/experiment_data_v5_fix_frontier/online-tree-25-n-1000-m-100-splitrate-1.00-splitroot-200.00-evalperiod-10-2013-03-14-02-05-12.588879/accuracy-35-25.pkl'),
                            (  '/home/davidm/projects/rftk/icml_kinect/experiment_data_v5_fix_frontier/online-tree-25-n-1000-m-100-splitrate-1.00-splitroot-200.00-evalperiod-10-2013-03-14-02-05-12.588879/forest-35-100.pkl',
                                '/home/davidm/projects/rftk/icml_kinect/experiment_data_v5_fix_frontier/online-tree-25-n-1000-m-100-splitrate-1.00-splitroot-200.00-evalperiod-10-2013-03-14-02-05-12.588879/accuracy-35-100.pkl'),
                            (  '/home/davidm/projects/rftk/icml_kinect/experiment_data_v5_fix_frontier/online-tree-25-n-1000-m-100-splitrate-1.00-splitroot-200.00-evalperiod-10-2013-03-14-02-05-12.588879/forest-35-200.pkl',
                                '/home/davidm/projects/rftk/icml_kinect/experiment_data_v5_fix_frontier/online-tree-25-n-1000-m-100-splitrate-1.00-splitroot-200.00-evalperiod-10-2013-03-14-02-05-12.588879/accuracy-35-200.pkl'),
                            (   '/home/davidm/projects/rftk/icml_kinect/experiment_data_v5_fix_frontier/online-tree-25-n-1000-m-100-splitrate-1.00-splitroot-200.00-evalperiod-10-2013-03-14-02-05-12.588879/forest-35-500.pkl',
                                '/home/davidm/projects/rftk/icml_kinect/experiment_data_v5_fix_frontier/online-tree-25-n-1000-m-100-splitrate-1.00-splitroot-200.00-evalperiod-10-2013-03-14-02-05-12.588879/accuracy-35-500.pkl'),
                            (   '/home/davidm/projects/rftk/icml_kinect/experiment_data_v5_fix_frontier/online-tree-25-n-1000-m-100-splitrate-1.00-splitroot-200.00-evalperiod-10-2013-03-14-02-05-12.588879/forest-35-1000.pkl',
                                '/home/davidm/projects/rftk/icml_kinect/experiment_data_v5_fix_frontier/online-tree-25-n-1000-m-100-splitrate-1.00-splitroot-200.00-evalperiod-10-2013-03-14-02-05-12.588879/accuracy-35-1000.pkl')
                            ])



    results = {'number_of_data': np.array([25, 100, 200, 500, 1000]), 'online_one_pass':online_one_pass_accuracy, 'online_multi_pass':online_multi_pass_accuracy,
                'offline':offline_accuracy, 'alpha_online_one_pass_accuracy':alpha_online_one_pass_accuracy, 'alpha_online_multi_pass_accuracy':alpha_online_multi_pass_accuracy,
                'online_one_pass_accuracy_no_depth':online_one_pass_accuracy_no_depth, 'online_35_pass_accuracy_no_depth':online_35_pass_accuracy_no_depth}
    print results
    pickle.dump(results, file(args.out_file, 'wb'))
