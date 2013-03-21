#!../../../epd-7.2-2-rh3-x86_64/bin/python
'''
@author: David Matheson

Offline training of kinect random forests
'''

import numpy as np
import matplotlib.pyplot as pl
import cPickle as pickle
from datetime import datetime
import argparse
import os

import rftk.native.asserts
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


class KinectOfflineConfig(object):
    def __init__(self):
        self.number_of_pixels_per_image = 1000

    def configure_offline_learner(self, number_of_trees):
        number_of_features = 5000
        y_dim = kinect_utils.number_of_body_parts
        min_impurity_gain = 0.01
        max_depth = 30
        min_samples_split = 10
        min_samples_leaf = 5
        sigma_x = 75
        sigma_y = 75

        feature_extractor = feature_extractors.DepthScaledDepthDeltaFeatureExtractor(sigma_x, sigma_y, number_of_features, True)
        node_data_collector = train.AllNodeDataCollectorFactory()
        class_infogain_best_split = best_splits.ClassInfoGainAllThresholdsBestSplit(1.0, 1, y_dim)

        # node_data_collector = train.TwoStreamRandomThresholdHistogramDataCollectorFactory(y_dim,
        #                                                                                     4,
        #                                                                                     0,
        #                                                                                     0.5)
        # class_infogain_best_split = best_splits.ClassInfoGainHistogramsBestSplit(y_dim,
        #         buffers.IMPURITY_HISTOGRAM_LEFT, buffers.IMPURITY_HISTOGRAM_RIGHT,
        #         buffers.YS_HISTOGRAM_LEFT, buffers.YS_HISTOGRAM_RIGHT)

        split_criteria = train.OfflineSplitCriteria(max_depth, min_impurity_gain,
                                                    min_samples_split, min_samples_leaf)
        extractor_list = [feature_extractor]
        train_config = train.TrainConfigParams(extractor_list,
                                                node_data_collector,
                                                class_infogain_best_split,
                                                split_criteria,
                                                number_of_trees,
                                                100000)
        depth_first_learner = train.DepthFirstParallelForestLearner(train_config)
        return depth_first_learner

    def get_sampling_config(self, number_of_datapoints):
        use_bootstrap = True
        return train.OfflineSamplingParams(number_of_datapoints, use_bootstrap)


def load_data(pose_path, list_of_poses, number_of_pixels_per_image):
    concat = False
    for i, pose_filename in enumerate(list_of_poses):
        print "Loading %d - %s" % (i, pose_filename)

        # Load single pose depth and class labels
        depths = pickle.load(open("%s%s_depth.pkl" % (pose_path, pose_filename), 'rb'))
        labels = pickle.load(open("%s%s_classlabels.pkl" % (pose_path, pose_filename), 'rb'))
        pixel_indices, pixel_labels = kinect_utils.sample_pixels(depths, labels, config.number_of_pixels_per_image)
        pixel_indices[:,0] = i

        depths_buffer = buffers.as_tensor_buffer(depths)
        pixel_labels_buffer = buffers.as_vector_buffer(pixel_labels)
        pixel_indices_buffer = buffers.as_matrix_buffer(pixel_indices)

        if concat:
            complete_depths_buffer.AppendSlice(depths_buffer)
            complete_pixel_labels_buffer.Append(pixel_labels_buffer)
            complete_pixel_indices_buffer.AppendVertical(pixel_indices_buffer)
        else:
            complete_depths_buffer = depths_buffer
            complete_pixel_labels_buffer = pixel_labels_buffer
            complete_pixel_indices_buffer = pixel_indices_buffer
            concat = True

    assert(complete_pixel_labels_buffer.GetN() == complete_pixel_indices_buffer.GetM())
    return complete_depths_buffer, complete_pixel_indices_buffer, complete_pixel_labels_buffer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build body part classification trees online')
    parser.add_argument('-i', '--pose_files_input_path', type=str, required=True)
    parser.add_argument('-p', '--poses_to_use_file', type=str, required=True)
    parser.add_argument('-n', '--number_of_images', type=int, required=True)
    parser.add_argument('-t', '--number_of_trees', type=int, required=True)
    args = parser.parse_args()

    offline_run_folder = ("experiment_data_v4/offline-tree-%d-n-%d-%s") % (
                            args.number_of_trees,
                            args.number_of_images,
                            str(datetime.now()).replace(':', '-').replace(' ', '-'))
    if not os.path.exists(offline_run_folder):
        os.makedirs(offline_run_folder)

    config = KinectOfflineConfig()

    poses_to_include_file = open(args.poses_to_use_file, 'r')
    pose_filenames = poses_to_include_file.read().split('\n')
    poses_to_include_file.close()

    depths_buffer, pixel_indices_buffer, pixel_labels_buffer = load_data(args.pose_files_input_path,
                                                                        pose_filenames[0:args.number_of_images],
                                                                        config.number_of_pixels_per_image)

    # Randomly offset scales
    number_of_datapoints = pixel_indices_buffer.GetM()
    offset_scales = np.array(np.random.uniform(0.8, 1.2, (number_of_datapoints, 2)), dtype=np.float32)
    datapoint_indices = np.array(np.arange(number_of_datapoints), dtype=np.int32)

    # Package buffers for learner
    bufferCollection = buffers.BufferCollection()
    bufferCollection.AddFloat32Tensor3Buffer(buffers.DEPTH_IMAGES, depths_buffer)
    bufferCollection.AddFloat32MatrixBuffer(buffers.OFFSET_SCALES, buffers.as_matrix_buffer(offset_scales))
    bufferCollection.AddInt32MatrixBuffer(buffers.PIXEL_INDICES, pixel_indices_buffer)
    bufferCollection.AddInt32VectorBuffer(buffers.CLASS_LABELS, pixel_labels_buffer)

    # Update learner
    offline_learner = config.configure_offline_learner(args.number_of_trees)
    sampling_config = config.get_sampling_config(number_of_datapoints)
    number_of_jobs = 2
    forest = offline_learner.Train(bufferCollection, buffers.Int32Vector(datapoint_indices), sampling_config, number_of_jobs)

    # Print forest stats
    forestStats = forest.GetForestStats()
    print forest.GetNumberOfTrees()
    forestStats.Print()

    #pickle forest and data used for training
    forest_pickle_filename = "%s/forest-1-%d.pkl" % (offline_run_folder, args.number_of_images)
    forest_utils.pickle_dump_native_forest(forest, forest_pickle_filename)
