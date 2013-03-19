import os
import numpy as np
import argparse
import itertools
import random as random
import cPickle as pickle
from datetime import datetime

import rftk.native.assert_util
import rftk.native.bootstrap
import rftk.native.buffers as buffers
import rftk.native.forest_data  as forest_data
import rftk.native.features
import rftk.native.feature_extractors as feature_extractors
import rftk.native.best_split as best_splits
import rftk.native.predict as predict
import rftk.native.train as train

import rftk.utils.buffer_converters as buffer_converters
import rftk.utils.predict as predict_utils
import rftk.utils.forest as forest_utils

import dist_utils
import experiment_measurement as exp_measurement

def run_experiment(config):
    data_config = config.get_data_config()

    out_measurements_filename = ("experiment_data/%s-online-measurements-%s.pkl") % (
                          config.get_experiment_name(),
                          str(datetime.now()).replace(':', '-').replace(' ', '-'))

    measurements = []
    for number_of_samples, number_of_passes in itertools.product(data_config.data_sizes, data_config.number_of_passes_through_data):
        for run_id in range(data_config.number_of_runs):

            X_train, Y_train, X_test, Y_test = data_config.load_data(number_of_samples, number_of_passes)
            (x_m,x_dim) = X_train.shape
            y_dim = int(np.max(Y_train) + 1)

            data = buffers.BufferCollection()
            data.AddFloat32MatrixBuffer(buffers.X_FLOAT_DATA, buffer_converters.as_matrix_buffer(X_train))
            data.AddInt32VectorBuffer(buffers.CLASS_LABELS, buffers.Int32Vector(Y_train))
            indices = buffers.Int32Vector( np.array(np.arange(x_m), dtype=np.int32) )

            online_config = config.get_online_config()

            # feature_extractor = feature_extractors.RandomProjectionFeatureExtractor( max_features, x_dim, x_dim, True)
            feature_extractor = feature_extractors.Float32AxisAlignedFeatureExtractor( online_config.number_of_features, x_dim, True)
            # node_data_collector = train.AllNodeDataCollectorFactory()
            # class_infogain_best_split = best_splits.ClassInfoGainAllThresholdsBestSplit(1.0, 1, y_dim)
            if not online_config.use_two_streams:
                node_data_collector = train.RandomThresholdHistogramDataCollectorFactory(y_dim,
                                                                                        online_config.number_of_thresholds,
                                                                                        online_config.null_probability)
                class_infogain_best_split = best_splits.ClassInfoGainHistogramsBestSplit(y_dim,
                        buffers.HISTOGRAM_LEFT, buffers.HISTOGRAM_RIGHT, buffers.HISTOGRAM_LEFT, buffers.HISTOGRAM_RIGHT)
            else:
                node_data_collector = train.TwoStreamRandomThresholdHistogramDataCollectorFactory(y_dim,
                                                                                                    online_config.number_of_thresholds,
                                                                                                    online_config.null_probability,
                                                                                                    online_config.impurity_probability)
                class_infogain_best_split = best_splits.ClassInfoGainHistogramsBestSplit(y_dim,
                        buffers.IMPURITY_HISTOGRAM_LEFT, buffers.IMPURITY_HISTOGRAM_RIGHT,
                        buffers.YS_HISTOGRAM_LEFT, buffers.YS_HISTOGRAM_RIGHT)

            # self.split_criteria = train.OnlineAlphaBetaSplitCriteria(   max_depth,
            #                                                             min_impurity,
            #                                                             min_samples_split)
            max_tree_depth = 100
            split_criteria = train.OnlineConsistentSplitCriteria(  online_config.split_rate,
                                                                        online_config.min_impurity_gain,
                                                                        online_config.number_of_data_to_split_root,
                                                                        online_config.number_of_data_to_force_split_root,
                                                                        max_tree_depth)

            extractor_list = [feature_extractor]
            train_config = train.TrainConfigParams(extractor_list,
                                                    node_data_collector,
                                                    class_infogain_best_split,
                                                    split_criteria,
                                                    online_config.number_of_trees)
            sampling_config = train.OnlineSamplingParams(False, 1.0)

            # Train online forest
            online_learner = train.OnlineForestLearner(train_config, sampling_config, 100)
            online_learner.Train(data, indices)

            predict_forest = predict_utils.MatrixForestPredictor(online_learner.GetForest())
            y_probs = predict_forest.predict_proba(X_test)
            y_hat = y_probs.argmax(axis=1)
            # print Y_test[(Y_test == y_hat)]
            # print y_probs[(Y_test == y_hat), Y_test[(Y_test == y_hat)]]
            accurracy = np.mean(Y_test == y_hat)
            print "%d %d %d: %0.5f" % (number_of_samples, number_of_passes, run_id, accurracy)
            forest_measurement = exp_measurement.OnlineForestMeasurement(data_config, online_config,
                number_of_samples, number_of_passes, accurracy)
            measurements.append(forest_measurement)

            stat_measurement = exp_measurement.OnlineForestStatsMeasurement(data_config, online_config,
                number_of_samples, number_of_passes, online_learner.GetForest().GetForestStats())
            measurements.append(stat_measurement)

            # Print forest stats
            forestStats = online_learner.GetForest().GetForestStats()
            forestStats.Print()

            if online_config.measure_tree_accuracy:
              online_forest_data = online_learner.GetForest()
              for tree_id in range(online_forest_data.GetNumberOfTrees()):
                  single_tree_forest_data = forest_data.Forest([online_forest_data.GetTree(tree_id)])
                  single_tree_forest_predictor = predict_utils.MatrixForestPredictor(single_tree_forest_data)
                  y_probs = single_tree_forest_predictor.predict_proba(X_test)
                  accurracy = np.mean(Y_test == y_probs.argmax(axis=1))
                  tree_measurement = exp_measurement.OnlineTreeMeasurement(data_config, online_config,
                      number_of_samples, number_of_passes, accurracy, tree_id)
                  measurements.append(tree_measurement)

        pickle.dump(measurements, open(out_measurements_filename, "wb"))

    return measurements


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Online Random Forest Training')
    parser.add_argument('-c', '--config_file', help='experiment config file', required=True)
    args = parser.parse_args()

    config = __import__(args.config_file)

    measurements = run_experiment(config)

