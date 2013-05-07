#!../../../epd-7.2-2-rh3-x86_64/bin/python
'''
@author: David Matheson

Online training of kinect random forests
'''

import numpy as np
import matplotlib.pyplot as pl
import cPickle as pickle
import gzip
from datetime import datetime
import argparse
import os
import random

import rftk.buffers as buffers
import rftk.forest_data as forest_data
import rftk.feature_extractors as feature_extractors
import rftk.best_split as best_splits
import rftk.predict as predict
import rftk.train as train

import utils as kinect_utils


class KinectOnlineConfig(object):
    def __init__(self):
        self.number_of_pixels_per_image = 1000

    def configure_online_learner(self, number_of_trees, split_rate, number_datapoints_split_root, eval_split_period, max_depth):
        number_of_features = 2000
        number_of_thresholds = 10
        y_dim = kinect_utils.number_of_body_parts
        null_probability = 0
        impurity_probability = 0.5
        split_rate = split_rate
        number_of_data_to_split_root = number_datapoints_split_root
        number_of_data_to_force_split_root = 4 * number_datapoints_split_root
        min_impurity_gain = 0.01

        sigma_x = 75
        sigma_y = 75

        feature_extractor = feature_extractors.DepthScaledDepthDeltaFeatureExtractor(sigma_x, sigma_y, number_of_features, True)
        node_data_collector = train.TwoStreamRandomThresholdHistogramDataCollectorFactory(y_dim,
                                                                                            number_of_thresholds,
                                                                                            null_probability,
                                                                                            impurity_probability)

        class_infogain_best_split = best_splits.ClassInfoGainHistogramsBestSplit(y_dim,
                buffers.IMPURITY_HISTOGRAM_LEFT, buffers.IMPURITY_HISTOGRAM_RIGHT,
                buffers.YS_HISTOGRAM_LEFT, buffers.YS_HISTOGRAM_RIGHT)

        split_criteria = train.OnlineConsistentSplitCriteria(split_rate,
                                                            min_impurity_gain,
                                                            number_of_data_to_split_root,
                                                            number_of_data_to_force_split_root,
                                                            max_depth)

        max_number_of_nodes = 1000
        max_number_of_node_in_frontier = 1000

        extractor_list = [feature_extractor]
        train_config = train.TrainConfigParams(extractor_list,
                                                node_data_collector,
                                                class_infogain_best_split,
                                                split_criteria,
                                                number_of_trees,
                                                max_number_of_nodes)
        sampling_config = train.OnlineSamplingParams(False, 1.0, eval_split_period)
        online_learner = train.OnlineForestLearner(train_config, sampling_config, max_number_of_node_in_frontier)
        return online_learner




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build body part classification trees online')
    parser.add_argument('-i', '--pose_files_input_path', type=str, required=True)
    parser.add_argument('-p', '--poses_to_use_file', type=str, required=True)
    parser.add_argument('-n', '--number_of_images', type=int, required=True)
    parser.add_argument('-m', '--number_of_passes_through_data', type=int, required=True)
    parser.add_argument('-t', '--number_of_trees', type=int, required=True)
    parser.add_argument('-s', '--split_rate', type=float, required=True)
    parser.add_argument('-r', '--number_datapoints_split_root', type=float, required=True)
    parser.add_argument('-e', '--eval_split_period', type=int, required=True)
    parser.add_argument('-d', '--max_depth', type=int, required=True)
    args = parser.parse_args()

    poses_to_include_file = open(args.poses_to_use_file, 'r')
    pose_filenames = poses_to_include_file.read().split('\n')
    poses_to_include_file.close()
    pose_filenames = pose_filenames[0:args.number_of_images]


    online_run_folder = ("experiment_results/online-tree-%d-n-%d-m-%d-splitrate-%0.2f-splitroot-%0.2f-evalperiod-%d-maxdepth-%s-%s") % (
                            args.number_of_trees,
                            args.number_of_images,
                            args.number_of_passes_through_data,
                            args.split_rate,
                            args.number_datapoints_split_root,
                            args.eval_split_period,
                            args.max_depth,
                            str(datetime.now()).replace(':', '-').replace(' ', '-'))
    if not os.path.exists(online_run_folder):
        os.makedirs(online_run_folder)

    config = KinectOnlineConfig()
    online_learner = config.configure_online_learner( args.number_of_trees,
                                                      args.split_rate,
                                                      args.number_datapoints_split_root,
                                                      args.eval_split_period,
                                                      args.max_depth)

    print "Starting %s" % online_run_folder

    for pass_id in range(args.number_of_passes_through_data):
        random.shuffle(pose_filenames)

        for i, pose_filename in enumerate(pose_filenames):
            print "Processing %d - %d - %s - %s" % (pass_id, i, pose_filename, str(datetime.now()))

            # Load single pose depth and class labels

            # depth_pickle_file = "%s%s_depth.pkl" % (args.pose_files_input_path, pose_filename)
            # depths = pickle.load(open(depth_pickle_file,'rb'))
            depths = kinect_utils.load_depth("%s%s.exr" % (args.pose_files_input_path, pose_filename))

            class_labels_pickle_file = "%s%s_classlabels.pkl" % (args.pose_files_input_path, pose_filename)
            labels = pickle.load(open(class_labels_pickle_file,'rb'))
            pixel_indices, pixel_labels = kinect_utils.sample_pixels(depths, labels, config.number_of_pixels_per_image)

            # Randomly sample pixels and offset scales
            (number_of_datapoints, _) = pixel_indices.shape
            offset_scales = np.array(np.random.uniform(0.8, 1.2, (number_of_datapoints, 2)), dtype=np.float32)
            datapoint_indices = np.array(np.arange(number_of_datapoints), dtype=np.int32)

            # Package buffers for learner
            bufferCollection = buffers.BufferCollection()
            bufferCollection.AddFloat32Tensor3Buffer(buffers.DEPTH_IMAGES, buffers.as_tensor_buffer(depths))
            bufferCollection.AddFloat32MatrixBuffer(buffers.OFFSET_SCALES, buffers.as_matrix_buffer(offset_scales))
            bufferCollection.AddInt32MatrixBuffer(buffers.PIXEL_INDICES, buffers.as_matrix_buffer(pixel_indices))
            bufferCollection.AddInt32VectorBuffer(buffers.CLASS_LABELS, buffers.as_vector_buffer(pixel_labels))

            # Update learner
            online_learner.Train(bufferCollection, buffers.Int32Vector(datapoint_indices))

            #pickle forest and data used for training
            if (i+1) % 500 == 0:
            # if True:
                forest_pickle_filename = "%s/forest-%d-%d.pkl" % (online_run_folder, pass_id, i+1)
                pickle.dump(online_learner.GetForest(), gzip.open(forest_pickle_filename, 'wb'))

                # Print forest stats
                forestStats = online_learner.GetForest().GetForestStats()
                forestStats.Print()
