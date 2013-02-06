import cPickle as pickle
import numpy as np
import random as random

import dist_utils

def get_experiment_name():
    return "5_class_mog"

def get_data_config():
    return DataConfig()

def get_online_config():
    return OnlineConfig()

def get_sklearn_offline_config():
    return SklearnOfflineConfig()

def get_online_sequential_config():
    return OnlineSequentialConfig()


# Sampling config should be the same for all experiments
class DataConfig(object):
    def __init__(self):
        self.pickle_data_file = "source_data/mog_5class_100000.pkl"
        self.data_sizes = [10, 20, 50, 100, 150, 250, 500, 1000, 2000, 5000, 10000, 25000]#, 50000, 100000]
        self.number_of_passes_through_data = [1,2,5]
        self.number_of_runs = 50
        self.bayes_accuracy = 0.666

    def load_data(self, data_size, passes_through_data):
        (X_train, Y_train, X_test, Y_test) = pickle.load(open(self.pickle_data_file, "rb"))
        (X_train, Y_train) = dist_utils.generate_multi_pass_dataset(X_train, Y_train, data_size, passes_through_data)
        return (X_train, Y_train, X_test, Y_test)


class OnlineConfig(object):
    def __init__(self):
        self.number_of_trees = 100
        self.number_of_features = 1
        self.number_of_thresholds = 10
        self.split_rate = 1.2
        self.number_of_data_to_split_root = 2
        self.number_of_data_to_force_split_root = 20
        self.use_two_streams = True
        self.null_probability = 0.5
        self.impurity_probability = 0.5
        self.min_impurity_gain = 0.001
        self.measure_tree_accuracy = True

class OnlineSequentialConfig(object):
    def __init__(self):
        self.number_of_trees = 100
        self.number_of_features = 1
        self.number_of_thresholds = 10
        self.split_rate = 1.2
        self.number_of_data_to_split_root = 2
        self.number_of_data_to_force_split_root = 20
        self.use_two_streams = True
        self.null_probability = 0.5
        self.impurity_probability = 0.5
        self.min_impurity_gain = 0.001
        self.measure_tree_accuracy = True
        self.use_bootstrap = True

class OfflineConfig(object):
    def __init__(self):
        self.number_of_trees = 100
        self.number_of_features = 1
        self.number_of_thresholds = 10
        self.max_depth = 1000
        self.min_samples_split = 100
        self.use_two_streams = False
        self.null_probability = 0.0
        # self.impurity_probability = 0.5
        self.min_impurity_gain = 0.001
        self.measure_tree_accuracy = False
        self.use_bootstrap = True
        self.number_of_jobs = 2


class SklearnOfflineConfig(object):
    def __init__(self):
        self.criterion = "entropy"
        self.number_of_trees = 100
        self.number_of_features = 1
        self.max_depth = 1000
        self.min_samples_split = 100
        self.number_of_jobs = 2
