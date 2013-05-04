import os
import numpy as np
import argparse
import itertools
import random as random
import cPickle as pickle

import rftk.buffers as buffers
import rftk.forest_data as forest_data
import rftk.feature_extractors as feature_extractors
import rftk.best_split as best_splits
import rftk.predict as predict
import rftk.train as train

import experiment_utils.measurements
import experiment_utils.management

from joblib import Parallel, delayed


def run_experiment(experiment_config, run_config):
    X_train, Y_train, X_test, Y_test = experiment_config.load_data(
        run_config.data_size,
        run_config.number_of_passes_through_data)
    (x_m,x_dim) = X_train.shape
    y_dim = int(np.max(Y_train) + 1)

    data = buffers.BufferCollection()
    data.AddFloat32MatrixBuffer(buffers.X_FLOAT_DATA, buffers.as_matrix_buffer(X_train))
    data.AddInt32VectorBuffer(buffers.CLASS_LABELS, buffers.Int32Vector(Y_train))
    indices = buffers.Int32Vector( np.array(np.arange(x_m), dtype=np.int32) )

    feature_extractor = feature_extractors.Float32AxisAlignedFeatureExtractor(
        run_config.number_of_features,
        x_dim,
        False) # choose number of features from poisson?
    
    node_data_collector = train.RandomThresholdHistogramDataCollectorFactory(
        y_dim,
        run_config.number_of_thresholds,
        run_config.null_probability)
    
    class_infogain_best_split = best_splits.ClassInfoGainHistogramsBestSplit(
        y_dim,
        buffers.HISTOGRAM_LEFT,
        buffers.HISTOGRAM_RIGHT,
        buffers.HISTOGRAM_LEFT,
        buffers.HISTOGRAM_RIGHT)
    
    split_criteria = train.OnlineAlphaBetaSplitCriteria(
        run_config.max_depth,
        run_config.min_impurity_gain,
        run_config.min_samples_split)
    
    extractor_list = [feature_extractor]
    train_config = train.TrainConfigParams(
        extractor_list,
        node_data_collector,
        class_infogain_best_split,
        split_criteria,
        run_config.number_of_trees,
        100000)
    sampling_config = train.OnlineSamplingParams(True, 1.0)

    # Train online forest
    online_learner = train.OnlineForestLearner(train_config, sampling_config, 100000)
    online_learner.Train(data, indices)

    predict_forest = predict.MatrixForestPredictor(online_learner.GetForest())
    y_probs = predict_forest.predict_proba(X_test)
    y_hat = y_probs.argmax(axis=1)
    accuracy = np.mean(Y_test == y_hat)
    stats = online_learner.GetForest().GetForestStats()
    forest_measurement = experiment_utils.measurements.StatsMeasurement(
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
        single_tree_forest_predictor = predict.MatrixForestPredictor(single_tree_forest_data)
        y_probs = single_tree_forest_predictor.predict_proba(X_test)
        accuracy = np.mean(Y_test == y_probs.argmax(axis=1))
        tree_measurement = experiment_utils.measurements.AccuracyMeasurement(
            accuracy=accuracy)
        tree_measurements.append(tree_measurement)

    return forest_measurement, tree_measurements


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Online Random Forest Training')
    parser.add_argument('-c', '--config_file', help='experiment config file', required=True)
    parser.add_argument('-o', '--out', help='output file name', required=True)
    args = parser.parse_args()

    config = __import__(args.config_file, fromlist=[""])
    experiment_config = config.get_experiment_config()
    run_config_template = config.get_saffari_config()

    configuration_domain = experiment_utils.management.ConfigurationDomain(
        run_config_template)

    def launch_job(configuration_domain, position):
        run_config = configuration_domain.configuration_at(position)
        forest_measurement, tree_measurement = run_experiment(experiment_config, run_config)
        return position, forest_measurement, tree_measurement

    # beware: these jobs take a lot of memory
    job_results = Parallel(n_jobs=2, verbose=5)(
        delayed(launch_job)(configuration_domain, position)
        for position in list(iter(configuration_domain)))

    forest_measurement_grid = experiment_utils.management.MeasurementGrid(
        configuration_domain,
        experiment_utils.measurements.StatsMeasurement)

    # this is going to break if we do tree measurements AND multiple
    # numbers of trees
    if experiment_config.measure_tree_accuracy:
        tree_measurement_grids = []
        for i in xrange(run_config_template.number_of_trees):
            tree_measurement_grids.append(
                experiment_utils.management.MeasurementGrid(
                    configuration_domain,
                    experiment_utils.measurements.AccuracyMeasurement))

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
    pickle.dump(results, open(args.out.format("forest"), "wb"))

    if experiment_config.measure_tree_accuracy:
        with open(args.out.format("trees"), 'wb') as tree_results_file:
            tree_results = [
                {
                    'measurements': tree_measurement_grid,
                    'experiment_config': experiment_config,
                    'run_config': run_config_template,
                }
                for tree_measurement_grid in tree_measurement_grids
                ]
            pickle.dump(tree_results, tree_results_file)
