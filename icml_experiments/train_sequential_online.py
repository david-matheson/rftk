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
import rftk.native.feature_extractors as feature_extractors
import rftk.native.best_split as best_splits
import rftk.native.predict as predict
import rftk.native.train as train

import rftk.utils.buffer_converters as buffer_converters
import rftk.utils.predict as predict_utils
import rftk.utils.forest as forest_utils

import dist_utils
import experiment_measurement as exp_measurement

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sequnetial Online Random Forest Training')
    parser.add_argument('-c', '--config_file', help='experiment config file', required=True)
    args = parser.parse_args()

    config = __import__(args.config_file)
    data_config = config.get_data_config()

    time_string = str(datetime.now()).replace(':', '-').replace(' ', '-')
    out_measurements_filename = ("experiment_data/%s-online-sequential-measurements-%s.pkl") % (
                          config.get_experiment_name(), time_string)
    forest_path = ("experiment_data/%s-online-sequential-forest-%s") % (
                          config.get_experiment_name(), time_string)
    if not os.path.exists(forest_path):
        os.makedirs(forest_path)

    # Load data
    X_train_full, Y_train_full, X_test, Y_test = data_config.load_data(data_config.data_sizes[-1], 1)
    (_,x_dim) = X_train_full.shape
    y_dim = int(np.max(Y_train_full) + 1)

    # Configure learner
    online_config = config.get_online_sequential_config()
    # feature_extractor = feature_extractors.RandomProjectionFeatureExtractor( max_features, x_dim, x_dim, True)
    feature_extractor = feature_extractors.AxisAlignedFeatureExtractor( online_config.number_of_features, x_dim, True)
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
    split_criteria = train.OnlineConsistentSplitCriteria(  online_config.split_rate,
                                                                online_config.min_impurity_gain,
                                                                online_config.number_of_data_to_split_root,
                                                                online_config.number_of_data_to_force_split_root)
    extractor_list = [feature_extractor]
    train_config = train.TrainConfigParams(extractor_list,
                                            node_data_collector,
                                            class_infogain_best_split,
                                            split_criteria,
                                            online_config.number_of_trees,
                                            1000)
    sampling_config = train.OnlineSamplingParams(False, 1.0)

    # Create online learner
    online_learner = train.OnlineForestLearner(train_config)

    measurements = []
    samples_seen = 0
    for number_of_samples in data_config.data_sizes:
        X_train = X_train_full[samples_seen:number_of_samples, :]
        Y_train = Y_train_full[samples_seen:number_of_samples]
        (x_m,x_dim) = X_train.shape
        samples_seen = number_of_samples

        # Setup rf buffers with data
        data = buffers.BufferCollection()
        data.AddFloat32MatrixBuffer(buffers.X_FLOAT_DATA, buffer_converters.as_matrix_buffer(X_train))
        data.AddInt32VectorBuffer(buffers.CLASS_LABELS, buffers.Int32Vector(Y_train))
        indices = buffers.Int32Vector( np.array(np.arange(x_m), dtype=np.int32) )

        # Train online forest
        online_learner.Train(data, indices, sampling_config)

        # print "predict"
        predict_forest = predict_utils.MatrixForestPredictor(online_learner.GetForest())
        y_probs = predict_forest.predict_proba(X_test)
        y_hat = y_probs.argmax(axis=1)
        accurracy = np.mean(Y_test == y_hat)
        print "%d: %0.5f" % (number_of_samples, accurracy)
        forest_measurement = exp_measurement.OnlineForestMeasurement(data_config, online_config,
            number_of_samples, 1, accurracy)
        measurements.append(forest_measurement)
        # print "predict done"

        if online_config.measure_tree_accuracy:
          online_forest_data = online_learner.GetForest()
          for tree_id in range(online_forest_data.GetNumberOfTrees()):
              single_tree_forest_data = forest_data.Forest([online_forest_data.GetTree(tree_id)])
              single_tree_forest_predictor = predict_utils.MatrixForestPredictor(single_tree_forest_data)
              y_probs = single_tree_forest_predictor.predict_proba(X_test)
              accurracy = np.mean(Y_test == y_probs.argmax(axis=1))
              tree_measurement = exp_measurement.OnlineTreeMeasurement(data_config, online_config,
                  number_of_samples, 1, accurracy, tree_id)
              measurements.append(tree_measurement)

          # pickle online forest for later evaluation
          forest_pickle_filename = "%s/forest-%d.pkl" % (forest_path, number_of_samples)
          forest_utils.pickle_dump_native_forest(online_learner.GetForest(), forest_pickle_filename)

    # pickle measurements for plotting
    pickle.dump(measurements, open(out_measurements_filename, "wb"))
