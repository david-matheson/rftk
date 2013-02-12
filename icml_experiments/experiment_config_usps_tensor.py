import cPickle as pickle
import numpy as np
import random as random

import dist_utils


def get_experiment_name():
    return "usps"

def get_experiment_config():
    return ExperimentConfig()

def get_online_config():
    return OnlineConfig()

def get_offline_config():
    return OfflineConfig()

def get_sklearn_offline_config():
    return SklearnOfflineConfig()

def load_sklearn_data(file):
    import sklearn.datasets as sklearn_datasets
    data = sklearn_datasets.load_svmlight_file(file)
    X = np.array(data[0].todense(),  dtype=np.float32)
    Y = np.array( data[1], dtype=np.int32 )
    print X.shape
    return X, Y


class ExperimentConfig(object):
    def __init__(self):
        self.measure_tree_accuracy = True

        data_config = DataConfig()
        (self.X_train_org, self.Y_train_org) = load_sklearn_data(data_config.data_file_train)
        (self.X_test_org, self.Y_test_org) = load_sklearn_data(data_config.data_file_test)

    def load_data(self, data_size, passes_through_data):
        (X_train, Y_train) = dist_utils.generate_multi_pass_dataset(self.X_train_org, self.Y_train_org, data_size, passes_through_data)
        return (X_train, Y_train, self.X_test_org, self.Y_test_org)



class DataConfig(object):
    def __init__(self):
        self.data_file_train = "source_data/usps"
        self.data_file_test = "source_data/usps.t"
        self.data_size = [20, 50, 100, 200, 500, 1000, 2000, 5000, 7291]
        #self.data_size = [20, 50, 100, 200]
        self.number_of_passes_through_data = [1,5,10,20]
        self.job_id = [1,2,3]
        self.bayes_accuracy = 1.0

class OnlineConfig(DataConfig):
    def __init__(self):
        DataConfig.__init__(self)

        self.number_of_trees = 100
        self.number_of_features = 8
        self.number_of_thresholds = 10
        self.split_rate = 1.1
        self.number_of_data_to_split_root = 10
        self.number_of_data_to_force_split_root = 1000
        self.use_two_streams = True
        self.null_probability = 0
        self.impurity_probability = 0.5
        self.min_impurity_gain = 0.1

# These haven't been set up yet, but should be re-enabled when their
# experiments are ready:

# class OnlineSequentialConfig(DataConfig):
#     def __init__(self):
#         DataConfig.__init__(self)

#         self.number_of_trees = 100
#         self.number_of_features = 1
#         self.number_of_thresholds = 10
#         self.split_rate = 1.2
#         self.number_of_data_to_split_root = 10
#         self.number_of_data_to_force_split_root = 1000
#         self.use_two_streams = True
#         self.null_probability = 0.5
#         self.impurity_probability = 0.5
#         self.min_impurity_gain = 0.1
#         self.measure_tree_accuracy = True

# class OfflineConfig(DataConfig):
#     def __init__(self):
#         DataConfig.__init__(self)

#         self.number_of_trees = 100
#         self.number_of_features = 8
#         self.number_of_thresholds = 10
#         self.max_depth = 1000
#         self.min_samples_split = 10
#         self.min_samples_leaf = 5
#         self.use_two_streams = False
#         self.null_probability = 0.0
#         # self.impurity_probability = 0.5
#         self.min_impurity_gain = 0.1
#         self.measure_tree_accuracy = False
#         self.use_bootstrap = True
#         self.number_of_jobs = 2

# class SklearnOfflineConfig(DataConfig):
#     def __init__(self):
#         DataConfig.__init__(self)

#         self.criterion = "entropy"
#         self.number_of_trees = 100
#         self.number_of_features = 8
#         self.max_depth = 1000
#         self.min_samples_split = 10
#         self.min_samples_leaf = 5
#         self.number_of_jobs = 2
