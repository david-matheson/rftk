import numpy as np
import argparse
import itertools
import cPickle as pickle
from datetime import datetime

import rftk.asserts
import rftk.bootstrap
import rftk.buffers as buffers
import rftk.forest_data as forest_data
import rftk.features
import rftk.feature_extractors as feature_extractors
import rftk.best_split as best_splits
import rftk.predict as predict
import rftk.train as train

import rftk.utils.predict as predict_utils

import experiment_measurement as exp_measurement


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sklearn accuracy on data')
    parser.add_argument('-c', '--config_file', help='experiment config file', required=True)
    args = parser.parse_args()

    config = __import__(args.config_file)
    data_config = config.get_data_config()
    offline_config = config.get_offline_config()

    out_measurements_filename = ("experiment_data/%s-offline-measurements-%s.pkl") % (
                          config.get_experiment_name(),
                          str(datetime.now()).replace(':', '-').replace(' ', '-'))

    measurements = []
    for number_of_samples, number_of_passes in itertools.product(data_config.data_sizes, data_config.number_of_passes_through_data):
        X_train, Y_train, X_test, Y_test = data_config.load_data(number_of_samples, number_of_passes)
        (x_m,x_n) = X_train.shape
        y_dim = int(np.max(Y_train) + 1)

        data = buffers.BufferCollection()
        data.AddFloat32MatrixBuffer(buffers.X_FLOAT_DATA, buffers.as_matrix_buffer(X_train))
        data.AddInt32VectorBuffer(buffers.CLASS_LABELS, buffers.as_vector_buffer(Y_train))
        sampling_config = train.OfflineSamplingParams(x_m, offline_config.use_bootstrap)
        indices = buffers.as_vector_buffer( np.array(np.arange(x_m), dtype=np.int32) )

        for run_id in range(data_config.number_of_runs):

            # feature_extractor = feature_extractors.RandomProjectionFeatureExtractor( offline_config.number_of_features, x_n, x_n)
            feature_extractor = feature_extractors.AxisAlignedFeatureExtractor( offline_config.number_of_features, x_n, True)
            node_data_collector = train.AllNodeDataCollectorFactory()
            class_infogain_best_split = best_splits.ClassInfoGainAllThresholdsBestSplit(1.0, 1, y_dim)
            # if not offline_config.use_two_streams:
            #     node_data_collector = train.RandomThresholdHistogramDataCollectorFactory(y_dim,
            #                                                                             offline_config.number_of_thresholds,
            #                                                                             offline_config.null_probability)
            #     class_infogain_best_split = best_splits.ClassInfoGainHistogramsBestSplit(y_dim,
            #             buffers.HISTOGRAM_LEFT, buffers.HISTOGRAM_RIGHT, buffers.HISTOGRAM_LEFT, buffers.HISTOGRAM_RIGHT)
            # else:
            #     node_data_collector = train.TwoStreamRandomThresholdHistogramDataCollectorFactory(y_dim,
            #                                                                                         offline_config.number_of_thresholds,
            #                                                                                         offline_config.null_probability,
            #                                                                                         offline_config.impurity_probability)
            #     class_infogain_best_split = best_splits.ClassInfoGainHistogramsBestSplit(y_dim,
            #             buffers.IMPURITY_HISTOGRAM_LEFT, buffers.IMPURITY_HISTOGRAM_RIGHT,
            #             buffers.YS_HISTOGRAM_LEFT, buffers.YS_HISTOGRAM_RIGHT)

            # self.split_criteria = train.OnlineAlphaBetaSplitCriteria(   max_depth,
            #                                                             min_impurity,
            #                                                             min_samples_split)
            # split_criteria = train.OnlineConsistentSplitCriteria(  offline_config.split_rate,
            #                                                             offline_config.min_impurity_gain,
            #                                                             offline_config.number_of_data_to_split_root,
            #                                                             offline_config.number_of_data_to_force_split_root)
            split_criteria = train.OfflineSplitCriteria(  offline_config.max_depth, offline_config.min_impurity_gain,
                                        offline_config.min_samples_split, offline_config.min_samples_leaf)
            extractor_list = [feature_extractor]
            train_config = train.TrainConfigParams(extractor_list,
                                                    node_data_collector,
                                                    class_infogain_best_split,
                                                    split_criteria,
                                                    offline_config.number_of_trees,
                                                    1000)
            depth_first_learner = train.DepthFirstParallelForestLearner(train_config)

            full_forest_data = depth_first_learner.Train(data, indices, sampling_config, offline_config.number_of_jobs)
            forest = predict_utils.MatrixForestPredictor(full_forest_data)

            y_probs = forest.predict_proba(X_test)
            y_hat = y_probs.argmax(axis=1)
            accurracy = np.mean(Y_test == y_hat)
            print "%d %d %d: %0.5f" % (number_of_samples, number_of_passes, run_id, accurracy)

            forest_measurement = exp_measurement.OfflineForestMeasurement(data_config, offline_config,
                number_of_samples, number_of_passes, accurracy)
            measurements.append(forest_measurement)

            stat_measurement = exp_measurement.OfflineForestStatsMeasurement(data_config, offline_config,
                number_of_samples, number_of_passes, full_forest_data.GetForestStats())
            measurements.append(stat_measurement)

        pickle.dump(measurements, open(out_measurements_filename, "wb"))
