#!../../../epd-7.2-2-rh3-x86_64/bin/python
'''
@author: David Matheson

Online training of kinect random forests
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


class KinectOnlineConfig(object):
    def __init__(self):
        self.number_of_pixels_per_image = 1000

    def configure_online_learner(self, number_of_trees, number_datapoints_split_root, eval_split_period, max_depth):
        number_of_features = 2000
        number_of_thresholds = 10
        y_dim = kinect_utils.number_of_body_parts
        null_probability = 0
        min_impurity_gain = 0.01

        sigma_x = 75
        sigma_y = 75

        feature_extractor = feature_extractors.DepthScaledDepthDeltaFeatureExtractor(sigma_x, sigma_y, number_of_features, True)

        node_data_collector = train.RandomThresholdHistogramDataCollectorFactory(y_dim,
                                                                                number_of_thresholds,
                                                                                null_probability)
        class_infogain_best_split = best_splits.ClassInfoGainHistogramsBestSplit(y_dim,
                buffers.HISTOGRAM_LEFT, buffers.HISTOGRAM_RIGHT, buffers.HISTOGRAM_LEFT, buffers.HISTOGRAM_RIGHT)

        split_criteria = train.OnlineAlphaBetaSplitCriteria(   max_depth,
                                                                    min_impurity_gain,
                                                                    number_datapoints_split_root)

        max_number_of_node_in_frontier = 1000000

        extractor_list = [feature_extractor]
        train_config = train.TrainConfigParams(extractor_list,
                                                node_data_collector,
                                                class_infogain_best_split,
                                                split_criteria,
                                                number_of_trees )
        sampling_config = train.OnlineSamplingParams(True, 1.0, eval_split_period)
        online_learner = train.OnlineForestLearner(train_config, sampling_config, max_number_of_node_in_frontier, max_depth)
        return online_learner




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build body part classification trees online')
    parser.add_argument('-p', '--train_poses', type=str, required=True)
    parser.add_argument('-m', '--number_of_passes_through_data', type=int, required=True)
    parser.add_argument('-t', '--number_of_trees', type=int, required=True)
    parser.add_argument('-r', '--number_datapoints_split_root', type=float, required=True)
    parser.add_argument('-e', '--eval_split_period', type=int, required=True)
    parser.add_argument('-d', '--max_depth', type=int, required=True)
    parser.add_argument('--list_of_sample_counts', type=str, required=True)
    args = parser.parse_args()

    depths_buffer, pixel_indices_buffer, pixel_labels_buffer = kinect_utils.load_training_data(args.train_poses)

    online_run_folder = ("results/kinect-saffari-n-%d-m-%d-tree-%d-splitroot-%0.2f-evalperiod-%d-maxdepth-%s-%s") % (
                            depths_buffer.GetL(),
                            args.number_of_passes_through_data,
                            args.number_of_trees,
                            args.number_datapoints_split_root,
                            args.eval_split_period,
                            args.max_depth,
                            str(datetime.now()).replace(':', '-').replace(' ', '-'))
    if not os.path.exists(online_run_folder):
        os.makedirs(online_run_folder)

    config = KinectOnlineConfig()
    online_learner = config.configure_online_learner( args.number_of_trees,
                                                      args.number_datapoints_split_root,
                                                      args.eval_split_period,
                                                      args.max_depth)

    print "Starting %s" % online_run_folder

    # Randomly offset scales
    number_of_datapoints = pixel_indices_buffer.GetM()
    offset_scales = np.array(np.random.uniform(0.8, 1.2, (number_of_datapoints, 2)), dtype=np.float32)

    # Package buffers for learner
    bufferCollection = buffers.BufferCollection()
    bufferCollection.AddFloat32Tensor3Buffer(buffers.DEPTH_IMAGES, depths_buffer)
    bufferCollection.AddFloat32MatrixBuffer(buffers.OFFSET_SCALES, buffers.as_matrix_buffer(offset_scales))
    bufferCollection.AddInt32MatrixBuffer(buffers.PIXEL_INDICES, pixel_indices_buffer)
    bufferCollection.AddInt32VectorBuffer(buffers.CLASS_LABELS, pixel_labels_buffer)

    # On the first pass through data learn for each sample counts
    list_of_sample_counts = eval(args.list_of_sample_counts)
    clipped_list_of_sample_counts = [min(s, pixel_labels_buffer.GetN()) for s in list_of_sample_counts]
    clipped_list_of_sample_ranges = zip([0] + clipped_list_of_sample_counts[:-1], clipped_list_of_sample_counts)
    print clipped_list_of_sample_ranges
    pass_id = 0
    for (start_index, end_index) in clipped_list_of_sample_ranges:
        datapoint_indices = np.array(np.arange(start_index, end_index), dtype=np.int32)
        online_learner.Train(bufferCollection, buffers.Int32Vector(datapoint_indices))

        #pickle forest and data used for training
        forest_pickle_filename = "%s/forest-%d-%d.pkl" % (online_run_folder, pass_id, end_index)
        pickle.dump(online_learner.GetForest(), gzip.open(forest_pickle_filename, 'wb'))

        # Print forest stats
        forestStats = online_learner.GetForest().GetForestStats()
        forestStats.Print()

    # For the rest of the passes use all of the data
    start_index = 0
    end_index = clipped_list_of_sample_counts[-1]
    for pass_id in range(1, args.number_of_passes_through_data):

        # Randomize the order
        perm = buffers.as_vector_buffer(np.array(np.random.permutation(pixel_labels_buffer.GetN()), dtype=np.int32))
        pixel_indices_buffer = pixel_indices_buffer.Slice(perm)
        pixel_labels_buffer = pixel_labels_buffer.Slice(perm)

        # Randomly offset scales
        number_of_datapoints = pixel_indices_buffer.GetM()
        offset_scales = np.array(np.random.uniform(0.8, 1.2, (number_of_datapoints, 2)), dtype=np.float32)

        # Package buffers for learner
        bufferCollection = buffers.BufferCollection()
        bufferCollection.AddFloat32Tensor3Buffer(buffers.DEPTH_IMAGES, depths_buffer)
        bufferCollection.AddFloat32MatrixBuffer(buffers.OFFSET_SCALES, buffers.as_matrix_buffer(offset_scales))
        bufferCollection.AddInt32MatrixBuffer(buffers.PIXEL_INDICES, pixel_indices_buffer)
        bufferCollection.AddInt32VectorBuffer(buffers.CLASS_LABELS, pixel_labels_buffer)

        datapoint_indices = np.array(np.arange(0, end_index), dtype=np.int32)
        online_learner.Train(bufferCollection, buffers.Int32Vector(datapoint_indices))

        #pickle forest and data used for training
        forest_pickle_filename = "%s/forest-%d-%d.pkl" % (online_run_folder, pass_id, end_index)
        pickle.dump(online_learner.GetForest(), gzip.open(forest_pickle_filename, 'wb'))

        # Print forest stats
        forestStats = online_learner.GetForest().GetForestStats()
        forestStats.Print()
