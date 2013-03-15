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
import experiment_measurement_tensor as exp_measurement
import experiment_utils_tensor as experiment_utils


from joblib import Parallel, delayed



def run_experiment(experiment_config, run_config):
    X_train, Y_train, X_test, Y_test = experiment_config.load_data(run_config.data_size, run_config.number_of_passes_through_data)
    (x_m,x_dim) = X_train.shape
    y_dim = int(np.max(Y_train) + 1)

    data = buffers.BufferCollection()
    data.AddFloat32MatrixBuffer(buffers.X_FLOAT_DATA, buffer_converters.as_matrix_buffer(X_train))
    data.AddInt32VectorBuffer(buffers.CLASS_LABELS, buffers.Int32Vector(Y_train))
    indices = buffers.Int32Vector( np.array(np.arange(x_m), dtype=np.int32) )

    # feature_extractor = feature_extractors.RandomProjectionFeatureExtractor( max_features, x_dim, x_dim, True)
    feature_extractor = feature_extractors.AxisAlignedFeatureExtractor( run_config.number_of_features, x_dim, True)
    # node_data_collector = train.AllNodeDataCollectorFactory()
    # class_infogain_best_split = best_splits.ClassInfoGainAllThresholdsBestSplit(1.0, 1, y_dim)
    if not run_config.use_two_streams:
        node_data_collector = train.RandomThresholdHistogramDataCollectorFactory(y_dim,
                                                                                run_config.number_of_thresholds,
                                                                                run_config.null_probability)
        class_infogain_best_split = best_splits.ClassInfoGainHistogramsBestSplit(y_dim,
                buffers.HISTOGRAM_LEFT, buffers.HISTOGRAM_RIGHT, buffers.HISTOGRAM_LEFT, buffers.HISTOGRAM_RIGHT)
    else:
        node_data_collector = train.TwoStreamRandomThresholdHistogramDataCollectorFactory(y_dim,
                                                                                            run_config.number_of_thresholds,
                                                                                            run_config.null_probability,
                                                                                            run_config.impurity_probability)
        class_infogain_best_split = best_splits.ClassInfoGainHistogramsBestSplit(y_dim,
                buffers.IMPURITY_HISTOGRAM_LEFT, buffers.IMPURITY_HISTOGRAM_RIGHT,
                buffers.YS_HISTOGRAM_LEFT, buffers.YS_HISTOGRAM_RIGHT)

    # self.split_criteria = train.OnlineAlphaBetaSplitCriteria(   max_depth,
    #                                                             min_impurity,
    #                                                             min_samples_split)
    max_tree_depth = 1000
    split_criteria = train.OnlineConsistentSplitCriteria(  run_config.split_rate,
                                                                run_config.min_impurity_gain,
                                                                run_config.number_of_data_to_split_root,
                                                                run_config.number_of_data_to_force_split_root,
                                                                max_tree_depth)

    extractor_list = [feature_extractor]
    train_config = train.TrainConfigParams(extractor_list,
                                            node_data_collector,
                                            class_infogain_best_split,
                                            split_criteria,
                                            run_config.number_of_trees,
                                            1000)
    sampling_config = train.OnlineSamplingParams(False, 1.0)

    # Train online forest
    online_learner = train.OnlineForestLearner(train_config, sampling_config, 10000)
    online_learner.Train(data, indices)

    predict_forest = predict_utils.MatrixForestPredictor(online_learner.GetForest())
    y_probs = predict_forest.predict_proba(X_test)
    y_hat = y_probs.argmax(axis=1)
    accuracy = np.mean(Y_test == y_hat)
    stats = online_learner.GetForest().GetForestStats()
    forest_measurement = exp_measurement.StatsMeasurement(
        accuracy=accuracy,
        min_depth=stats.mMinDepth,
        max_depth=stats.mMaxDepth,
        average_depth=stats.GetAverageDepth(),
        min_estimator_points=stats.mMinEstimatorPoints,
        max_estimator_points=stats.mMaxEstimatorPoints,
        average_estimator_points=stats.GetAverageEstimatorPoints(),
        total_estimator_points=stats.mTotalEstimatorPoints)

    if not experiment_config.measure_tree_accuracy:
        return forest_measurement, None

    tree_measurements = []
    online_forest_data = online_learner.GetForest()
    for tree_id in range(online_forest_data.GetNumberOfTrees()):
        single_tree_forest_data = forest_data.Forest([online_forest_data.GetTree(tree_id)])
        single_tree_forest_predictor = predict_utils.MatrixForestPredictor(single_tree_forest_data)
        y_probs = single_tree_forest_predictor.predict_proba(X_test)
        # print "A:", y_probs.argmax(axis=1)
        # print "B:", Y_test
        accuracy = np.mean(Y_test == y_probs.argmax(axis=1))
        tree_measurement = exp_measurement.AccuracyMeasurement(
            accuracy=accuracy)
        tree_measurements.append(tree_measurement)

    return forest_measurement, tree_measurements





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Online Random Forest Training')
    parser.add_argument('-c', '--config_file', help='experiment config file', required=True)
    args = parser.parse_args()

    config = __import__(args.config_file)
    experiment_config = config.get_experiment_config()
    run_config_template = config.get_online_config()

    out_measurements_filename = ("experiment_data/%s-online-measurements-{}-%s.pkl") % (
                          config.get_experiment_name(),
                          str(datetime.now()).replace(':', '-').replace(' ', '-'))

    configuration_domain = experiment_utils.ConfigurationDomain(run_config_template)

    def launch_job(configuration_domain, position):
        run_config = configuration_domain.configuration_at(position)
        forest_measurement, tree_measurement = run_experiment(experiment_config, run_config)
        return position, forest_measurement, tree_measurement

    job_results = Parallel(n_jobs=4, verbose=5)(
        delayed(launch_job)(configuration_domain, position)
        for position in reversed(list(iter(configuration_domain))))

    forest_measurement_grid = exp_measurement.MeasurementGrid(
        configuration_domain,
        exp_measurement.StatsMeasurement)

    # this is going to break if we do tree measurements AND multiple
    # numbers of trees
    if experiment_config.measure_tree_accuracy:
        tree_measurement_grids = []
        for i in xrange(run_config_template.number_of_trees):
            tree_measurement_grids.append(
                exp_measurement.MeasurementGrid(
                    configuration_domain,
                    exp_measurement.AccuracyMeasurement))

    for position, forest_measurement, tree_measurements in job_results:
        forest_measurement_grid.record_at(position, forest_measurement)

        if experiment_config.measure_tree_accuracy:
            for tree_measurement_grid, tree_measurement in zip(tree_measurement_grids, tree_measurements):
                tree_measurement_grid.record_at(position, tree_measurement)

    results = {
        'measurements': forest_measurement_grid,
        'experiment_config': experiment_config,
        'run_config': run_config_template,
    }
    pickle.dump(results, open(out_measurements_filename.format("forest"), "wb"))

    if experiment_config.measure_tree_accuracy:
        for i, tree_measurement_grid in enumerate(tree_measurement_grids):
            results = {
                'measurements': tree_measurement_grid,
                'experiment_config': experiment_config,
                'run_config': run_config_template,
            }
            pickle.dump(results, open(out_measurements_filename.format(
                "tree%05d" % i), "wb"))
