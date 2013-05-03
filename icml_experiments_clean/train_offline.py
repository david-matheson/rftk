import numpy as np
import argparse
import itertools
import cPickle as pickle
from datetime import datetime

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
    
    node_data_collector = train.AllNodeDataCollectorFactory()
    class_infogain_best_split = best_splits.ClassInfoGainAllThresholdsBestSplit(
        1.0, 1, y_dim)

    split_criteria = train.OfflineSplitCriteria(
        run_config.max_depth,
        run_config.min_impurity_gain,
        run_config.min_samples_split,
        run_config.min_samples_leaf)
    
    extractor_list = [feature_extractor]
    train_config = train.TrainConfigParams(
        extractor_list,
        node_data_collector,
        class_infogain_best_split,
        split_criteria,
        run_config.number_of_trees,
        100000)
    sampling_config = train.OfflineSamplingParams(x_m, run_config.use_bootstrap)

    depth_first_learner = train.DepthFirstParallelForestLearner(train_config)
    full_forest_data = depth_first_learner.Train(
        data,
        indices,
        sampling_config,
        run_config.number_of_jobs)
    
    forest = predict.MatrixForestPredictor(full_forest_data)

    y_probs = forest.predict_proba(X_test)
    y_hat = y_probs.argmax(axis=1)
    accuracy = np.mean(Y_test == y_hat)
    stats = full_forest_data.GetForestStats()

    forest_measurement = experiment_utils.measurements.StatsMeasurement(
        accuracy=accuracy,
        min_depth=stats.mMinDepth,
        max_depth=stats.mMaxDepth,
        average_depth=stats.GetAverageDepth(),
        min_estimator_points=stats.mMinEstimatorPoints,
        max_estimator_points=stats.mMaxEstimatorPoints,
        average_estimator_points=stats.GetAverageEstimatorPoints(),
        total_estimator_points=stats.mTotalEstimatorPoints)

    return forest_measurement

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Offline accuracy on data')
    parser.add_argument('-c', '--config_file', help='experiment config file', required=True)
    parser.add_argument('-o', '--out', help='output file name', required=True)
    args = parser.parse_args()

    config = __import__(args.config_file, fromlist=[""])
    experiment_config = config.get_experiment_config()
    run_config_template = config.get_offline_config()

    configuration_domain = experiment_utils.management.ConfigurationDomain(
        run_config_template)

    def launch_job(configuration_domain, position):
        run_config = configuration_domain.configuration_at(position)
        forest_measurement = run_experiment(experiment_config, run_config)
        return position, forest_measurement

    job_results = Parallel(n_jobs=4, verbose=5)(
        delayed(launch_job)(configuration_domain, position)
        for position in list(iter(configuration_domain)))

    forest_measurement_grid = experiment_utils.management.MeasurementGrid(
        configuration_domain,
        experiment_utils.measurements.StatsMeasurement)

    for position, forest_measurement in job_results:
        forest_measurement_grid.record_at(position, forest_measurement)

    results = {
        'measurements': forest_measurement_grid,
        'experiment_config': experiment_config,
        'run_config': run_config_template,
    }
    pickle.dump(results, open(args.out.format("forest"), "wb"))
