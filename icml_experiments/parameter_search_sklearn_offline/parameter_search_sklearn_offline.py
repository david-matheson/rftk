import numpy as np

import train_sklearn_offline
import cPickle as pickle

import dist_utils

def load_sklearn_data(file):
    import sklearn.datasets as sklearn_datasets
    data = sklearn_datasets.load_svmlight_file(file)
    X = np.array(data[0].todense(),  dtype=np.float32)
    Y = np.array( data[1], dtype=np.int32 )
    print X.shape
    return X, Y

class DataConfig(object):
    def __init__(self, params):
        #self.pickle_data_file = "../source_data/mog_5class_100000.pkl"
        self.data_file_train = "../source_data/usps"
        self.data_file_test = "../source_data/usps.t"
        self.data_sizes = [7291] # all of usps = 7291
        self.number_of_passes_through_data = [1]
        self.number_of_runs = 1
        self.bayes_accuracy = 0.93 # scikits learn accuracy on usps

        (self.X_train_org, self.Y_train_org) = load_sklearn_data(self.data_file_train)
        (self.X_test_org, self.Y_test_org) = load_sklearn_data(self.data_file_test)

    def load_data(self, data_size, passes_through_data):
        (X_train, Y_train) = dist_utils.generate_multi_pass_dataset(self.X_train_org, self.Y_train_org, data_size, passes_through_data)
        return (X_train, Y_train, self.X_test_org, self.Y_test_org)

class SklearnOfflineConfig(object):
    def __init__(self, params):
        self.criterion = "entropy"
        self.number_of_trees = 100
        self.number_of_jobs = 2

        self.number_of_features = int(params['number_of_features'])
        self.max_depth = int(params['max_depth'])
        self.min_samples_split = int(params['min_samples_split'])

class ExperimentConfig(object):
    def __init__(self, params):
        self.data_config = DataConfig(params)
        self.sklearn_offline_config = SklearnOfflineConfig(params)
    
    def get_experiment_name(self):
        return "usps_parameter_search_sklearn"

    def get_data_config(self):
        return self.data_config

    def get_online_config(self):
        return self.online_config

    def get_sklearn_offline_config(self):
        return self.sklearn_offline_config


def main(job_id, params):

    experiment_config = ExperimentConfig(params)

    measurements = train_sklearn_offline.run_experiment(experiment_config)

    # spearmint minimizes the objective
    return -measurements[-1].accuracy


# Useful for debugging because spearmint doesn't tell you _where_ an error
# occurred.
if __name__ == "__main__":
    train_sklearn_offline.run_experiment(ExperimentConfig({'number_of_features':
        4, 'max_depth': 10, 'min_samples_split': 5}))
