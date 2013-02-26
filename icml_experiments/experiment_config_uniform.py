import cPickle as pickle
import numpy as np
import random as random
import numpy as np

import dist_utils

def get_experiment_name():
    return "epsilon_uniform"

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
        self.data_sizes = [50, 100, 250, 500, 1000, 2000, 5000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000]#, 1500000, 2000000]#, 2500000, 3000000, 3500000, 4000000, 4500000]
        self.number_of_passes_through_data = [1]
        self.number_of_runs = 1
        self.bayes_accuracy = 1.0
        self.test_size = 100000
        self.X_test = np.random.uniform(-1.0, 1.0, size=(self.test_size, 2))
        self.Y_test =  np.random.multinomial(1, [0.5001, 0.4999], size=self.test_size).argmax(axis=1)

    def load_data(self, data_size, passes_through_data):
        X_train = np.array( np.random.uniform(-1.0, 1.0, size=(data_size*passes_through_data, 2)), dtype=np.float32 )
        Y_train =  np.random.multinomial(1, [0.5001, 0.4999], size=data_size*passes_through_data).argmax(axis=1)
        return (X_train, Y_train, self.X_test, self.Y_test)


class OnlineConfig(object):
    def __init__(self):
        self.number_of_trees = 100
        self.number_of_features = 1
        self.number_of_thresholds = 10
        self.split_rate = 1.6
        self.number_of_data_to_split_root = 1
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
        self.split_rate = 2
        self.number_of_data_to_split_root = 1
        self.number_of_data_to_force_split_root = 20
        self.use_two_streams = True
        self.null_probability = 0.0
        self.impurity_probability = 0.5
        self.min_impurity_gain = 0.001
        self.measure_tree_accuracy = True
        self.use_bootstrap = False

class OfflineConfig(object):
    def __init__(self):
        self.number_of_trees = 100
        self.number_of_features = 1
        self.number_of_thresholds = 10
        self.max_depth = 1000
        self.min_samples_split = 100
        self.min_samples_leaf = 50
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
        self.min_samples_leaf = 50
        self.number_of_jobs = 2
