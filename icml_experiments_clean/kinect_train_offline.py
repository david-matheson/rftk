#!../../../epd-7.2-2-rh3-x86_64/bin/python
'''
@author: David Matheson

Offline training of kinect random forests
'''

import numpy as np
import cPickle as pickle
import gzip
from datetime import datetime
import argparse
import os

import rftk.buffers as buffers
import rftk.feature_extractors as feature_extractors
import rftk.best_split as best_splits
import rftk.train as train

import kinect_utils as kinect_utils


class KinectOfflineConfig(object):
    def __init__(self):
        self.number_of_pixels_per_image = 1000

    def configure_offline_learner(self, number_of_trees):
        number_of_features = 2000
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
        split_criteria = train.OfflineSplitCriteria(max_depth, min_impurity_gain,
                                                    min_samples_split, min_samples_leaf)

        extractor_list = [feature_extractor]
        train_config = train.TrainConfigParams(extractor_list,
                                                node_data_collector,
                                                class_infogain_best_split,
                                                split_criteria,
                                                number_of_trees)
        depth_first_learner = train.DepthFirstParallelForestLearner(train_config)
        return depth_first_learner

    def get_sampling_config(self, number_of_datapoints):
        use_bootstrap = True
        return train.OfflineSamplingParams(number_of_datapoints, use_bootstrap)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build body part classification trees online')
    parser.add_argument('-p', '--train_poses', type=str, required=True)
    parser.add_argument('-n', '--number_of_samples', type=str, required=True)
    parser.add_argument('-t', '--number_of_trees', type=int, required=True)
    args = parser.parse_args()

    depths_buffer, pixel_indices_buffer, pixel_labels_buffer = kinect_utils.load_training_data(args.train_poses)

    number_of_datapoints = min(args.number_of_samples, pixel_labels_buffer.GetN())

    offline_run_folder = ("results/kinect-offline-n-%d-m-%d-tree-%d-%s") % (
                            depths_buffer.GetL(),
                            number_of_datapoints,
                            args.number_of_trees,
                            str(datetime.now()).replace(':', '-').replace(' ', '-'))
    if not os.path.exists(offline_run_folder):
        os.makedirs(offline_run_folder)

    config = KinectOfflineConfig()

    # Randomly offset scales
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
    forestStats.Print()

    #pickle forest
    forest_pickle_filename = "%s/forest-0-%d.pkl" % (offline_run_folder, number_of_datapoints)
    pickle.dump(forest, gzip.open(forest_pickle_filename, 'wb'))
