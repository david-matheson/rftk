#!../../../epd-7.2-2-rh3-x86_64/bin/python
'''
@author: David Matheson

Offline training of kinect random forests
'''

import numpy as np

import cPickle as pickle
import gzip
import argparse

from datetime import datetime
import os

import rftk.buffers as buffers
import rftk.forest_data as forest_data
import rftk.feature_extractors as feature_extractors
import rftk.best_split as best_splits
import rftk.predict as predict
import rftk.train as train

import utils as kinect_utils

class KinectRandomForestEvaluator:

    def __init__(self, train_datafile='train.pkl', test_datafile='test.pkl'):
        (self.train_depth, self.train_pixel_indices, self.train_pixel_labels) = pickle.load(gzip.open(train_datafile, 'rb'))
        (self.test_depth, self.test_pixel_labels) = pickle.load(gzip.open(test_datafile, 'rb'))
        self.number_of_datapoints = self.train_pixel_indices.GetM()
        self.offset_scales = np.array(np.random.uniform(0.8, 1.2, (self.number_of_datapoints, 2)), dtype=np.float32)


    def _configure_learner(self, number_of_trees, number_of_features,
                            max_depth, min_samples_split, min_samples_leaf, min_impurity_gain,
                            ux, uy, vx, vy, timeout_in_seconds):
        y_dim = kinect_utils.number_of_body_parts

        feature_extractor = feature_extractors.DepthScaledDepthDeltaFeatureExtractor(ux, uy, vx, vy, number_of_features, True)
        node_data_collector = train.AllNodeDataCollectorFactory()
        class_infogain_best_split = best_splits.ClassInfoGainAllThresholdsBestSplit(1.0, 1, y_dim)

        offline_split_criteria = train.OfflineSplitCriteria(max_depth, min_impurity_gain,
                                                    min_samples_split, min_samples_leaf)
        time_split_criteria = train.TimerSplitCriteria(offline_split_criteria, timeout_in_seconds)

        extractor_list = [feature_extractor]
        train_config = train.TrainConfigParams(extractor_list,
                                                node_data_collector,
                                                class_infogain_best_split,
                                                time_split_criteria,
                                                number_of_trees)
        depth_first_learner = train.DepthFirstParallelForestLearner(train_config)
        return depth_first_learner

    def _get_sampling_config(self, number_of_datapoints, use_bootstrap):
        return train.OfflineSamplingParams(number_of_datapoints, use_bootstrap)


    def evaluate(self, use_bootstrap, number_of_trees, number_of_features,
                    max_depth, min_samples_split, min_samples_leaf, min_impurity_gain,
                    ux, uy, vx, vy,
                    timeout_in_seconds=300, number_of_jobs=2):
        offline_run_folder = ("experiment_data_randomforest/offline-b-%r-t-%d-f-%d-d-%d-ms-%d-ml-%d-ig-%0.2f-ux-%0.2f-uy-%0.2f-vx-%0.2f-vy-%0.2f-%s") % (
                            use_bootstrap,
                            number_of_trees,
                            number_of_features,
                            max_depth,
                            min_samples_split,
                            min_samples_leaf,
                            min_impurity_gain,
                            ux, uy, vx, vy,
                            str(datetime.now()).replace(':', '-').replace(' ', '-'))
        if not os.path.exists(offline_run_folder):
            os.makedirs(offline_run_folder)

        # Create learner
        offline_learner = self._configure_learner( number_of_trees, number_of_features,
                                                    max_depth, min_samples_split, min_samples_leaf, min_impurity_gain,
                                                    ux, uy, vx, vy, timeout_in_seconds=timeout_in_seconds)

        # Package buffers for learner
        bufferCollection = buffers.BufferCollection()
        bufferCollection.AddFloat32Tensor3Buffer(buffers.DEPTH_IMAGES, self.train_depth)
        bufferCollection.AddFloat32MatrixBuffer(buffers.OFFSET_SCALES, buffers.as_matrix_buffer(self.offset_scales))
        bufferCollection.AddInt32MatrixBuffer(buffers.PIXEL_INDICES, self.train_pixel_indices)
        bufferCollection.AddInt32VectorBuffer(buffers.CLASS_LABELS, self.train_pixel_labels)
        datapoint_indices = np.array(np.arange(self.number_of_datapoints), dtype=np.int32)
        sampling_config = self._get_sampling_config(self.number_of_datapoints, use_bootstrap)

        # train and pickle forest
        print "Starting training", str(datetime.now())
        forest = offline_learner.Train(bufferCollection, buffers.Int32Vector(datapoint_indices), sampling_config, number_of_jobs)
        forest_pickle_filename = "%s/forest.pkl" % offline_run_folder
        pickle.dump(forest, gzip.open(forest_pickle_filename, 'wb'))

        # compute and pickle accuracy
        print "Starting testing", str(datetime.now())
        accuracy = kinect_utils.classification_accuracy(buffers.as_numpy_array(self.test_depth),
                                                        buffers.as_numpy_array(self.test_pixel_labels),
                                                        forest,
                                                        number_of_jobs=number_of_jobs)
        accuracy_pickle_filename = "%s/accuracy.pkl" % offline_run_folder
        pickle.dump(accuracy, file(accuracy_pickle_filename, 'wb'))
        print "Finish testing", str(datetime.now())
        return accuracy



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build body part classification trees online')
    parser.add_argument('--input_train', type=str, required=True)
    parser.add_argument('--input_test', type=str, required=True)
    parser.add_argument('-b', '--use_bootstrap', type=int, required=True)
    parser.add_argument('-t', '--number_of_trees', type=int, required=True)
    parser.add_argument('-f', '--number_of_features', type=int, required=True)
    parser.add_argument('-d', '--max_depth', type=int, required=True)
    parser.add_argument('-ms', '--min_samples_split', type=int, required=True)
    parser.add_argument('-ml', '--min_samples_leaf', type=int, required=True)
    parser.add_argument('-ig', '--min_impurity_gain', type=float, required=True)
    parser.add_argument('-ux', type=float, required=True)
    parser.add_argument('-uy', type=float, required=True)
    parser.add_argument('-vx', type=float, required=True)
    parser.add_argument('-vy', type=float, required=True)
    args = parser.parse_args()

    evaluator = KinectRandomForestEvaluator(train_datafile=args.input_train, test_datafile=args.input_test)
    accuracy = evaluator.evaluate(args.use_bootstrap==1, args.number_of_trees, args.number_of_features,
                                args.max_depth, args.min_samples_split, args.min_samples_leaf, args.min_impurity_gain,
                                args.ux, args.uy, args.vx, args.vy)
    print accuracy









