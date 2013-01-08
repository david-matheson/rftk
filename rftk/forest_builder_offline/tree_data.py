import numpy as np


class TreeData:
    def __init__(self, max_number_of_nodes=1000000, max_int_params_dim=10, max_float_params_dim=10, y_dim=1):
        self.paths = np.negative( np.ones((max_number_of_nodes, 2), dtype=np.int32) ) #stores children node ids
        self.int_params = np.zeros((max_number_of_nodes, max_int_params_dim), dtype=np.int32) # stores feature type
        self.float_params = np.zeros((max_number_of_nodes, max_float_params_dim), dtype=np.float32) #stores parameters and threshold
        self.ys = np.zeros((max_number_of_nodes, y_dim), dtype=np.float32)

    def truncate(self, number_of_nodes):
        self.paths = self.paths[0:number_of_nodes+1]
        self.int_params = self.int_params[0:number_of_nodes+1]
        self.float_params = self.float_params[0:number_of_nodes+1]
        self.ys = self.ys[0:number_of_nodes+1]