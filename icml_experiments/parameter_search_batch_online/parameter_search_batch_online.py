import numpy as np

import train_batch_online
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

class OnlineConfig(object):
    def __init__(self, params):
        self.number_of_trees = 100
        self.use_two_streams = True
        self.measure_tree_accuracy = False

        # can't have numpy types here
        self.null_probability = float(params['null_probability'])
        self.min_impurity_gain = float(params['min_impurity_gain'])
        self.impurity_probability = float(params['impurity_probability'])
        self.split_rate = float(params['split_rate'])
        self.number_of_features = int(params['number_of_features'])
        self.number_of_thresholds = int(params['number_of_thresholds'])
        self.number_of_data_to_split_root = int(params['number_of_data_to_split_root'])
        self.number_of_data_to_force_split_root = int(params['number_of_data_to_force_split_root'])
        

class ExperimentConfig(object):
    def __init__(self, params):
        self.data_config = DataConfig(params)
        self.online_config = OnlineConfig(params)
    
    def get_experiment_name(self):
        return "usps_parameter_search"

    def get_data_config(self):
        return self.data_config

    def get_online_config(self):
        return self.online_config


def main(job_id, params):

    experiment_config = ExperimentConfig(params)

    measurements = train_batch_online.run_experiment(experiment_config)

    # spearmint minimizes the objective
    return -measurements[-1].accuracy

