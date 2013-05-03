import cPickle as pickle
import numpy as np

import dist_utils

def get_experimen_name():
    return "5_class_mog"

def get_experiment_config():
    return ExperimentConfig()

def get_online_config():
    return OnlineConfig()

class ExperimentConfig(object):
    def __init__(self):
        self.measure_tree_accuracy = True

    def load_data(self, data_size, passes_through_data):
        data_config = DataConfig()
        (X_train, Y_train, X_test, Y_test) = pickle.load(open(data_config.pickle_data_file, "rb"))
        (X_train, Y_train) = dist_utils.generate_multi_pass_dataset(X_train, Y_train, data_size, passes_through_data)
        return (X_train, Y_train, X_test, Y_test)


class DataConfig(object):
    def __init__(self):
        self.pickle_data_file = "source_data/mog_5class_100000.pkl"
        self.data_size = map(int, np.exp(np.linspace(np.log(100), np.log(30000), 10)))
        self.number_of_passes_through_data = 1
        self.job_id = range(10)
        self.bayes_accuracy = 0.666

class OnlineConfig(DataConfig):
    def __init__(self):
        DataConfig.__init__(self)

        self.number_of_trees = 100
        self.number_of_features = 1
        self.number_of_thresholds = 10
        self.split_rate = 1.1
        self.number_of_data_to_split_root = 1
        self.number_of_data_to_force_split_root = 1000
        self.use_two_streams = True
        self.null_probability = 0
        self.impurity_probability = 0.5
        self.min_impurity_gain = 0.001

