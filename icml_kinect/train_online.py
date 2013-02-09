#!../../../epd-7.2-2-rh3-x86_64/bin/python
'''
@author: David Matheson

Online training of kinect random forests
'''

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

import rftk.utils.buffer_converters as buffer_converters
import rftk.utils.forest as forest_utils

import utils as kinect_utils


class KinectOnlineConfig(object):
    def __init__(self):
        self.number_of_pixels_per_image = 1000

    def configure_online_learner(self, split_rate, number_datapoints_split_root):
        number_of_trees = 20
        number_of_features = 1000
        number_of_thresholds = 3
        y_dim = kinect_utils.number_of_body_parts
        null_probability = 0
        impurity_probability = 0.5
        split_rate = split_rate
        min_impurity_gain = 0.0
        number_of_data_to_split_root = number_datapoints_split_root
        number_of_data_to_force_split_root = 100 * number_datapoints_split_root

        sigma_x = 100
        sigma_y = 100

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
                                                            number_of_data_to_force_split_root)
        extractor_list = [feature_extractor]
        train_config = train.TrainConfigParams(extractor_list,
                                                node_data_collector,
                                                class_infogain_best_split,
                                                split_criteria,
                                                number_of_trees,
                                                100000)
        online_learner = train.OnlineForestLearner(train_config)
        return online_learner

    def get_sampling_config(self):
        return train.OnlineSamplingParams(False, 1.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build body part classification trees online')
    parser.add_argument('-i', '--pose_files_input_path', type=str, required=True)
    parser.add_argument('-s', '--split_rate', type=float, required=True)
    parser.add_argument('-r', '--number_datapoints_split_root', type=float, required=True)
    parser.add_argument('-p', '--poses_to_use_file', type=str, required=True)
    args = parser.parse_args()

    poses_to_include_file = open(args.poses_to_use_file, 'r')
    pose_filenames = poses_to_include_file.read().split('\n')
    poses_to_include_file.close()

    online_run_folder = ("experiment_data/online-run-splitrate-%0.2f-splitroot-%0.2f-%s") % (
                            args.split_rate,
                            args.number_datapoints_split_root,
                            str(datetime.now()).replace(':', '-').replace(' ', '-'))
    if not os.path.exists(online_run_folder):
        os.makedirs(online_run_folder)

    config = KinectOnlineConfig()
    online_learner = config.configure_online_learner(args.split_rate, args.number_datapoints_split_root)

    run_info = {'pose_filenames': [], 'pixel_indices': [], 'offset_scales': []}

    for i, pose_filename in enumerate(pose_filenames):
        print "Processing %d - %s" % (i, pose_filename)

        # Load single pose depth and class labels
        depths = np.clip(kinect_utils.load_depth("%s%s.exr" % (args.pose_files_input_path, pose_filename)), 0.0, 6.0)
        class_labels_pickle_file = "%s%s_classlabels.pkl" % (args.pose_files_input_path, pose_filename)
        if not os.path.exists(class_labels_pickle_file):
            class_labels_png = pl.imread("%s%s.png" % (args.pose_files_input_path, pose_filename))
            class_labels = kinect_utils.image_labels(class_labels_png)
            pickle.dump(class_labels, open(class_labels_pickle_file, 'wb'))
        labels = pickle.load(open(class_labels_pickle_file,'rb'))
        pixel_indices, pixel_labels = kinect_utils.sample_pixels(depths, labels, config.number_of_pixels_per_image)

        # Randomly sample pixels and offset scales
        (number_of_datapoints, _) = pixel_indices.shape
        offset_scales = np.array(np.random.uniform(0.8, 1.2,number_of_datapoints), dtype=np.float32)
        datapoint_indices = np.array(np.arange(number_of_datapoints), dtype=np.int32)

        # Package buffers for learner
        bufferCollection = buffers.BufferCollection()
        bufferCollection.AddFloat32Tensor3Buffer(buffers.DEPTH_IMAGES, buffer_converters.as_tensor_buffer(depths))
        bufferCollection.AddFloat32MatrixBuffer(buffers.OFFSET_SCALES, buffer_converters.as_matrix_buffer(offset_scales))
        bufferCollection.AddInt32MatrixBuffer(buffers.PIXEL_INDICES, buffer_converters.as_matrix_buffer(pixel_indices))
        bufferCollection.AddInt32VectorBuffer(buffers.CLASS_LABELS, buffer_converters.as_vector_buffer(pixel_labels))

        # Update learner
        online_learner.Train(bufferCollection, buffers.Int32Vector(datapoint_indices), config.get_sampling_config())

        #pickle forest and data used for training
        forest_pickle_filename = "%s/forest-%d.pkl" % (online_run_folder, i)
        forest_utils.pickle_dump_native_forest(online_learner.GetForest(), forest_pickle_filename)

        run_info['pose_filenames'].append(pose_filename)
        run_info['pixel_indices'].append(pixel_indices)
        run_info['offset_scales'].append(offset_scales)
        pickle.dump(run_info, open("%s/run_info.pkl" % (online_run_folder), 'wb'))

        # Print forest stats
        forestStats = online_learner.GetForest().GetForestStats()
        forestStats.Print()