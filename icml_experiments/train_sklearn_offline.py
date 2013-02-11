import numpy as np
import argparse
import itertools
import cPickle as pickle
from datetime import datetime

import sklearn.ensemble

import experiment_measurement as exp_measurement


def run_experiment(config):
    data_config = config.get_data_config()

    out_measurements_filename = ("experiment_data/%s-sklearn-offline-measurements-%s.pkl") % (
                          config.get_experiment_name(),
                          str(datetime.now()).replace(':', '-').replace(' ', '-'))

    measurements = []
    for number_of_samples, number_of_passes in itertools.product(data_config.data_sizes, data_config.number_of_passes_through_data):
        X_train, Y_train, X_test, Y_test = data_config.load_data(number_of_samples, number_of_passes)


        for run_id in range(data_config.number_of_runs):

            sklearn_config = config.get_sklearn_offline_config()
            forest = sklearn.ensemble.RandomForestClassifier(   criterion=sklearn_config.criterion,
                                                    max_features=sklearn_config.number_of_features,
                                                    n_estimators=sklearn_config.number_of_trees,
                                                    max_depth=sklearn_config.max_depth,
                                                    min_samples_split=sklearn_config.min_samples_split,
                                                    min_samples_leaf=sklearn_config.min_samples_leaf,
                                                    n_jobs=sklearn_config.number_of_jobs)
            forest.fit(X_train, Y_train)
            y_probs = forest.predict_proba(X_test)
            y_hat2 = forest.predict(X_test)
            y_hat = y_probs.argmax(axis=1)
            accurracy = np.mean(Y_test == y_hat2)
            print "%d %d %d: %0.5f" % (number_of_samples, number_of_passes, run_id, accurracy)
            forest_measurement = exp_measurement.SklearnForestMeasurement(data_config, sklearn_config,
                number_of_samples, number_of_passes, accurracy)
            measurements.append(forest_measurement)




        pickle.dump(measurements, open(out_measurements_filename, "wb"))

    return measurements


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sklearn accuracy on data')
    parser.add_argument('-c', '--config_file', help='experiment config file', required=True)
    args = parser.parse_args()

    config = __import__(args.config_file)

    measurements = run_experiment(config)
